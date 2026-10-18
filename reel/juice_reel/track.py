"""Follow things through footage, and pin labels to them.

track: CSRT tracking of one box per target, re-anchored at hand-placed keyframes. Drift is
       removed by measuring the tracker's error when it reaches each keyframe and spreading
       the correction back across the segment, so the path is exact at every keyframe and
       smooth in between.
tag:   burn a name tag above each tracked box into a copy of the source. Tagging the source
       (before cutting) means tags stay attached through every crop, pan, zoom and shake.

Targets file (JSON):
  {"CLAUDE":  {"color": [217,119,87], "keys": [[6.0, 1000, 585, 50, 55], ...], "hide": [[27.6, 29.2]]},
   "CHATGPT": {...}}
Keys are [time_s, x, y, w, h] in source pixels; "hide" ranges suppress the tag (occluded/off-screen).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT, BLACK_CONDENSED = "/System/Library/Fonts/HelveticaNeue.ttc", 9

WORK_W = 1280  # track at this width for speed; results are scaled back to source pixels


def _probe(video: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video))
    w, h, fps = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps


def _new_tracker():
    return cv2.legacy.TrackerCSRT_create() if hasattr(cv2, "legacy") else cv2.TrackerCSRT_create()


def track(video: Path, targets: dict, start: float, end: float) -> dict:
    W, H, fps = _probe(video)
    k = WORK_W / W
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    frames_t, small = [], []
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        ok, f = cap.read()
        if not ok or t > end:
            break
        frames_t.append(t)
        small.append(cv2.resize(f, (WORK_W, int(H * k))))
    cap.release()
    ts = np.array(frames_t)

    out = {"fps": fps, "size": [W, H], "targets": {}}
    for name, spec in targets.items():
        keys = sorted(spec["keys"])
        idx = [int(np.argmin(np.abs(ts - kt))) for kt, *_ in keys]
        boxes = np.full((len(ts), 4), np.nan)
        for (ka, kb), (ia, ib) in zip(zip(keys, keys[1:] + [None]), zip(idx, idx[1:] + [len(ts) - 1])):
            box = np.array(ka[1:], float)
            boxes[ia] = box
            if kb is None and ia >= len(ts) - 1:
                break
            trk = _new_tracker()
            trk.init(small[ia], tuple(int(v * k) for v in box))
            last = box.copy()
            for i in range(ia + 1, ib + 1):
                ok, b = trk.update(small[i])
                if ok:
                    last = np.array(b, float) / k
                boxes[i] = last
            if kb is not None:
                # Tracker's error on arrival at the next keyframe, removed linearly over the segment.
                kbox = np.array(kb[1:], float)
                err = boxes[ib] - kbox
                n = max(1, ib - ia)
                for i in range(ia + 1, ib + 1):
                    f = (i - ia) / n
                    boxes[i] -= err * f
                    # If the tracker jumped to something else, trust the straight path between keyframes.
                    lerp = box + (kbox - box) * f
                    if np.hypot(*(boxes[i][:2] - lerp[:2])) > 2.5 * max(lerp[2], lerp[3]):
                        boxes[i] = lerp
        hide = spec.get("hide", [])
        path = []
        for t, b in zip(ts, boxes):
            vis = not np.isnan(b).any() and not any(a <= t <= z for a, z in hide)
            path.append([round(float(t), 4), *[round(float(v), 1) for v in np.nan_to_num(b)], int(vis)])
        out["targets"][name] = {"color": spec.get("color", [255, 255, 255]), "path": path}
    return out


def debug_sheet(video: Path, tracks: dict, times: list[float], out: Path, width: int = 640, cols: int = 4) -> Path:
    """Frames with each tracked box drawn on, to verify a track by eye."""
    cap = cv2.VideoCapture(str(video))
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            continue
        for name, tr in tracks["targets"].items():
            p = min(tr["path"], key=lambda r: abs(r[0] - t))
            x, y, w, h, vis = p[1:]
            c = tuple(int(v) for v in reversed(tr["color"]))
            cv2.rectangle(f, (int(x), int(y)), (int(x + w), int(y + h)), c if vis else (128, 128, 128), 4)
            cv2.putText(f, name if vis else name + " (hidden)", (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.4, c, 3)
        cv2.putText(f, f"{t:.2f}s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
        tiles.append(cv2.resize(f, (width, int(f.shape[0] * width / f.shape[1]))))
    cap.release()
    rows = [np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out), np.vstack(rows))
    return out


def _tag_image(name: str, color, size: int) -> Image.Image:
    font = ImageFont.truetype(FONT, size, index=BLACK_CONDENSED)
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    tw = d.textlength(name, font=font)
    padx, pady, tip = size * 0.45, size * 0.18, size * 0.35
    w, h = int(tw + padx * 2), int(size * 1.15 + pady * 2)
    img = Image.new("RGBA", (w + 16, int(h + tip) + 16), (0, 0, 0, 0))
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ds = ImageDraw.Draw(shadow)
    ds.rounded_rectangle([8, 10, 8 + w, 10 + h], radius=h // 2, fill=(0, 0, 0, 140))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(4)))
    di = ImageDraw.Draw(img)
    di.rounded_rectangle([8, 8, 8 + w, 8 + h], radius=h // 2, fill=tuple(color) + (255,))
    cx = 8 + w / 2
    di.polygon([(cx - tip, 8 + h - 1), (cx + tip, 8 + h - 1), (cx, 8 + h + tip)], fill=tuple(color) + (255,))
    di.text((8 + padx, 8 + pady - size * 0.05), name, font=font, fill=(255, 255, 255, 255))
    return img


def _smooth(path: list, win: int) -> np.ndarray:
    a = np.array([r[1:5] for r in path], float)
    if win <= 1:
        return a
    ker = np.ones(win) / win
    pad = win // 2
    return np.stack([np.convolve(np.pad(a[:, j], pad, mode="edge"), ker, mode="valid")[: len(a)] for j in range(4)], 1)


def tag(video: Path, tracks: dict, out: Path, size: int = 46, gap: float = 0.35, crf: int = 14) -> Path:
    """Burn a tag above each tracked box into a full-length copy of the source (same size, frame rate
    and timestamps, so a spec can swap it in without changing any shot times)."""
    W, H = tracks["size"]
    fps = tracks["fps"]
    smooth = {n: _smooth(tr["path"], 7) for n, tr in tracks["targets"].items()}
    times = {n: np.array([r[0] for r in tr["path"]]) for n, tr in tracks["targets"].items()}
    tag_imgs = {n: _tag_image(n, tr["color"], size) for n, tr in tracks["targets"].items()}

    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", str(crf),
                            "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    n_bytes = W * H * 3
    i = 0
    while True:
        buf = dec.stdout.read(n_bytes)
        if len(buf) < n_bytes:
            break
        t = i / fps
        frame = Image.frombuffer("RGB", (W, H), buf).convert("RGBA")
        for name, tr in tracks["targets"].items():
            if not times[name][0] <= t <= times[name][-1]:
                continue
            j = int(np.argmin(np.abs(times[name] - t)))
            if not tr["path"][j][5]:
                continue
            x, y, w, h = smooth[name][j]
            img = tag_imgs[name]
            # Tip of the tag sits a little above the top of the head box.
            px, py = int(x + w / 2 - img.width / 2), int(y - h * gap - img.height)
            frame.alpha_composite(img, (max(0, min(W - img.width, px)), max(0, py)))
        enc.stdin.write(frame.convert("RGB").tobytes())
        i += 1
    enc.stdin.close()
    enc.wait()
    dec.wait()
    return out
