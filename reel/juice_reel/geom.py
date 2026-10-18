"""Where a source pixel lands in the output, for any shot at any moment.

The renderer frames a shot with: cover-scale -> crop (fixed, panned or following a track) ->
optional zoom (static or eased push/pull, centered on the followed target). This module is the
single source of that math: render.py turns it into ffmpeg expressions, and overlays (callouts,
name tags) use it to pin graphics to the exact on-screen position of a tracked thing.
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from .spec import Shot, Spec


@lru_cache(maxsize=64)
def source_size(path: str) -> tuple[int, int]:
    """Display size (rotation applied), e.g. a portrait iPhone clip is 1080x1920, not 1920x1080."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:stream_side_data=rotation", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = st["width"], st["height"]
    rot = next((abs(int(sd.get("rotation", 0))) for sd in st.get("side_data_list", []) if "rotation" in sd), 0)
    return (h, w) if rot in (90, 270) else (w, h)


def live(shot: Shot) -> float:
    """Output seconds of moving picture (before any freeze)."""
    return (shot.end - shot.start) / shot.speed


def zoom_range(shot: Shot):
    if shot.zoom != 1.0:
        return shot.zoom
    return [1.0, 1 + shot.punch_in] if shot.punch_in else 1.0


def zoom_at(shot: Shot, tr: float, fps: int) -> float:
    z = zoom_range(shot)
    if not isinstance(z, (list, tuple)):
        return float(z)
    n = max(1, round(live(shot) * fps))
    on = min(round(tr * fps), n)
    z0, z1 = z
    return z0 + (z1 - z0) * (1 - (1 - min(on / n, 1)) ** 3)


def follow_points(shot: Shot, spec: Spec) -> tuple[list[float], list[float], list[float]]:
    """Output times and the followed target's point (source px) at those times."""
    path = spec.tracks["targets"][shot.follow]["path"]
    pt = np.array([r[0] for r in path])
    L = live(shot)
    n = max(3, int(L * 10) + 1)
    ts = list(np.linspace(0, L, n))
    xs, ys = [], []
    for t in ts:
        src_t = shot.start + t * shot.speed
        near = [r[1:5] for r, keep in zip(path, np.abs(pt - src_t) <= 0.08) if keep]
        x, y, w, h = np.array(near or [path[int(np.argmin(np.abs(pt - src_t)))][1:5]]).mean(0)
        xs.append(x + w / 2)
        ys.append(y + h / 2 + h * shot.follow_y)
    return ts, xs, ys


class Framing:
    """Precomputed framing for one shot in one output format."""

    def __init__(self, shot: Shot, spec: Spec):
        st = spec.style
        self.shot, self.W, self.H, self.fps = shot, st["width"], st["height"], st["fps"]
        sw, sh = source_size(str(spec.sources[shot.source]))
        self.sw, self.sh = sw, sh
        self.k = max(self.W / sw, self.H / sh)
        self.cw, self.ch = sw * self.k, sh * self.k
        self.y0 = (self.ch - self.H) / 2
        self.L = live(shot)
        if shot.follow:
            self.ts, xs, ys = follow_points(shot, spec)
            self.x0s = [min(max(x * self.k - self.W / 2, 0), self.cw - self.W) for x in xs]
            self.txs = [x * self.k - a for x, a in zip(xs, self.x0s)]
            self.tys = [y * self.k - self.y0 for y in ys]
        else:
            self.ts = None

    def x0(self, tr: float) -> float:
        if self.ts is not None:
            return float(np.interp(tr, self.ts, self.x0s))
        fx = self.shot.frame_x
        if isinstance(fx, (list, tuple)):
            a, b = fx
            s = min(max(tr / max(self.L, 0.01), 0), 1)
            cx = a + (b - a) * s * s * (3 - 2 * s)
        else:
            cx = fx
        return min(max(cx * self.cw - self.W / 2, 0), self.cw - self.W)

    def zoom_window(self, tr: float) -> tuple[float, float, float]:
        """(zoom, left, top) of the zoom window in crop pixels."""
        z = zoom_at(self.shot, tr, self.fps)
        if z == 1.0:
            return 1.0, 0.0, 0.0
        if self.ts is not None:
            cx, cy = float(np.interp(tr, self.ts, self.txs)), float(np.interp(tr, self.ts, self.tys))
        else:
            cx, cy = self.W / 2, self.H / 2
        left = min(max(cx - self.W / (2 * z), 0), self.W - self.W / z)
        top = min(max(cy - self.H / (2 * z), 0), self.H - self.H / z)
        return z, left, top

    def to_output(self, sx: float, sy: float, tr: float) -> tuple[float, float, float]:
        """Source pixel -> output pixel at shot time tr, plus the on-screen scale factor."""
        tr = min(tr, self.L)
        z, left, top = self.zoom_window(tr)
        x = (sx * self.k - self.x0(tr) - left) * z
        y = (sy * self.k - self.y0 - top) * z
        return x, y, self.k * z
