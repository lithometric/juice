"""Everything that sits on top of the footage, laid out on the output timeline.

Layers come from four places:
  shots[].fx      relative to the shot ("at", default "dur" = to the end of the shot)
  overlays        absolute output time ("from"/"to") or by shot index ("from_shot"/"to_shot");
                  for things that span cuts: meters HUD, captions, name tags, watermark, ticker, progress
  events          moments in the SOURCE footage ({"t": 27.08, ...}). juice finds where that moment
                  lands in the edit and fires the event's fx / sfx / meter change there, so re-cutting
                  never breaks timing. Shots marked "replay": true (hooks, flash-forwards) don't fire events.
  automatic       slow-motion shots get letterbox bars + a "SLOW-MO 0.5x" chip (style.auto_slowmo)

Meters are named numbers (health, balance, hype, rage, progress...) that start at "meters" values,
can drift per second, and step (eased) on events: {"meter": {"id": "gpt", "delta": -20}} or {"set": 0}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .components import COMPONENTS, Z, ease_out
from .geom import Framing
from .spec import Spec


@dataclass
class Layer:
    comp: str
    params: dict
    start: float
    end: float

    def active(self, t: float) -> bool:
        return self.start <= t < self.end


@dataclass
class Timeline:
    layers: list[Layer]
    sfx: list[tuple[float, str]]
    starts: list[float]
    durations: list[float]
    meters: dict = field(default_factory=dict)
    words: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.starts[-1] + self.durations[-1]


def _normalize(fx: dict) -> dict:
    """Accept the first-generation fx names/params."""
    f = dict(fx)
    if f.get("type") == "title" and "big" in f:
        f["text"], f["kicker"] = f.pop("big"), f.pop("small", None)
    return f


def source_to_output(spec: Spec, starts: list[float], t: float, source: str | None = None) -> list[float]:
    """Every output time where source moment t appears (skipping replay shots)."""
    out = []
    for shot, s0 in zip(spec.shots, starts):
        if shot.replay or (source and shot.source != source):
            continue
        if shot.start <= t <= shot.end:
            out.append(s0 + (t - shot.start) / shot.speed)
    return out


def build(spec: Spec) -> Timeline:
    fps = spec.fps
    durs = [spec.shot_frames(s) / fps for s in spec.shots]
    starts = list(np.cumsum([0.0] + durs[:-1]))
    total = starts[-1] + durs[-1]
    layers, sfx = [], []

    def add(fx: dict, start: float, end: float):
        fx = _normalize(fx)
        kind = fx.get("type")
        if kind == "sfx":
            sfx.append((start, fx["sound"]))
            return
        if kind not in COMPONENTS:
            raise ValueError(f"unknown component {kind!r} (have: {', '.join(sorted(COMPONENTS))})")
        if fx.get("sfx"):
            sfx.append((start, fx["sfx"]))
        layers.append(Layer(kind, fx, start, min(end, total)))

    for shot, s0, d in zip(spec.shots, starts, durs):
        for fx in shot.fx:
            at = s0 + float(fx.get("at", 0))
            add(fx, at, at + float(fx["dur"]) if "dur" in fx else s0 + d)
        if spec.style.get("auto_slowmo", True) and shot.speed < 1:
            add({"type": "letterbox"}, s0, s0 + d)
            add({"type": "chip", "text": f"SLOW-MO  {shot.speed:g}x", "fill": "danger", "ink": "paper",
                 "x": 0.95, "y": 0.1 if spec.style["width"] > spec.style["height"] else 0.2, "anchor": "right"},
                s0 + 0.1, s0 + d)

    for ov in spec.overlays:
        a = starts[ov["from_shot"]] if "from_shot" in ov else float(ov.get("from", 0))
        b = (starts[ov["to_shot"]] + durs[ov["to_shot"]]) if "to_shot" in ov else float(ov.get("to", total))
        add(ov, a, b)

    meter_steps: dict[str, list] = {k: [] for k in spec.meters}
    for ev in spec.events:
        times = [float(ev["at"])] if "at" in ev else source_to_output(spec, starts, float(ev["t"]), ev.get("source"))
        for i, at in enumerate(times):
            for fx in ev.get("fx", []):
                add(fx, at + float(fx.get("delay", 0)), at + float(fx.get("delay", 0)) + float(fx.get("dur", 1.2)))
            if ev.get("sfx"):
                sfx.append((at, ev["sfx"]))
            if ev.get("meter") and i == 0:
                m = ev["meter"]
                meter_steps.setdefault(m["id"], []).append((at, m))

    words = []
    if spec.transcripts:
        for key, path in spec.transcripts.items():
            data = json.loads(Path(path).read_text())
            for w in data.get("words", []):
                if w.get("type") != "word":
                    continue
                # A word belongs to a shot only if most of it is actually heard there: a cut that trims a
                # filler leaves a few ms of its onset, and that must not put the word on screen.
                for shot, s0 in zip(spec.shots, starts):
                    if shot.replay or shot.source != key:
                        continue
                    a, b = max(w["start"], shot.start), min(w["end"], shot.end)
                    if b - a <= 0 or (b - a) < 0.5 * max(1e-3, w["end"] - w["start"]):
                        continue
                    at = s0 + (a - shot.start) / shot.speed
                    words.append({"text": w["text"].strip(), "start": at, "end": at + (b - a) / shot.speed})
        words.sort(key=lambda w: w["start"])

    meters = {}
    for mid, init in spec.meters.items():
        start_v = init["value"] if isinstance(init, dict) else float(init)
        drift = init.get("drift", 0.0) if isinstance(init, dict) else 0.0
        steps = []
        v_base = start_v
        for at, m in sorted(meter_steps.get(mid, []), key=lambda x: x[0]):
            before = v_base + drift * at + sum(dv for _, dv in steps)
            dv = (m["set"] - before) if "set" in m else float(m.get("delta", 0))
            steps.append((at, dv))
        meters[mid] = (start_v, drift, steps)

    layers.sort(key=lambda L: (Z.get(L.comp, 25), L.start))
    return Timeline(layers=layers, sfx=sorted(sfx), starts=starts, durations=durs, meters=meters, words=words)


class Ctx:
    """What a component can ask about the moment it's drawing."""

    def __init__(self, spec: Spec, theme: dict, tl: Timeline):
        self.spec, self.theme, self.tl = spec, theme, tl
        self.W, self.H, self.fps = spec.style["width"], spec.style["height"], spec.fps
        self.u = min(self.W, self.H) / 1080
        self.total = tl.total
        self.words = tl.words
        self.t_out = 0.0
        self._framing: dict[int, Framing] = {}

    def meter(self, mid: str) -> float:
        if mid not in self.tl.meters:
            return 100.0
        v0, drift, steps = self.tl.meters[mid]
        v = v0 + drift * self.t_out
        for at, dv in steps:
            if self.t_out >= at:
                v += dv * ease_out((self.t_out - at) / 0.3)
        return max(0.0, min(100.0, v))

    def shot_at(self, t: float) -> int:
        i = int(np.searchsorted(self.tl.starts, t, side="right")) - 1
        return max(0, min(i, len(self.spec.shots) - 1))

    def target(self, name: str):
        """Tracked box (x, y, w, h) in output pixels right now, or None if hidden / off-screen."""
        tracks = self.spec.tracks
        if not tracks or name not in tracks["targets"]:
            return None
        i = self.shot_at(self.t_out)
        shot = self.spec.shots[i]
        from .geom import source_size
        if list(source_size(str(self.spec.sources[shot.source]))) != list(tracks["size"]):
            return None
        fr = self._framing.get(i) or self._framing.setdefault(i, Framing(shot, self.spec))
        tr = self.t_out - self.tl.starts[i]
        src_t = shot.start + min(tr, fr.L) * shot.speed
        path = tracks["targets"][name]["path"]
        times = tracks["targets"][name].setdefault("_t", np.array([r[0] for r in path]))
        if not times[0] <= src_t <= times[-1]:
            return None
        j = int(np.argmin(np.abs(times - src_t)))
        lo, hi = max(0, j - 3), min(len(path), j + 4)
        if not path[j][5]:
            return None
        x, y, w, h = np.array([r[1:5] for r in path[lo:hi]]).mean(0)
        ox, oy, s = fr.to_output(x, y, tr)
        bw, bh = w * s, h * s
        if ox + bw < 0 or ox > self.W or oy + bh < 0 or oy > self.H:
            return None
        return ox, oy, bw, bh


def draw_frame(ctx: Ctx, layers: list[Layer], t_out: float):
    from PIL import Image
    ctx.t_out = t_out
    canvas = Image.new("RGBA", (ctx.W, ctx.H), (0, 0, 0, 0))
    for L in layers:
        if L.active(t_out):
            COMPONENTS[L.comp](canvas, t_out - L.start, L.end - L.start, L.params, ctx)
    return canvas
