"""Put recordings of one event on one clock: every camera and audio recorder lined up by its sound.

The clock is one file's own time (a separate audio recorder when there is one, otherwise the longest clip
with sound): a moment at clock time t is at (t - offset) / rate in each other file. `rate` carries clock
drift: a phone and a recorder disagree by 10-50 parts per million, which is a frame or more after half an
hour, so a long take is fitted at many points, not lined up once.

How a file is placed:
  1. a hint: matching start timecode (cameras jammed to one clock) or the phones' creation dates (to a second)
  2. coarse: cross-correlate the two syllable envelopes (10 ms frames), near the hint first, then anywhere
  3. fine: GCC-PHAT on up to 12 eight-second windows across the overlap (to a fraction of a sample at
     8 kHz), then a straight-line fit through them that drops the windows that disagree (offset + drift)
A file that shares no sound with the clock is placed through another file it does share sound with.

    juice sync footage/ -o footage/sync.json        # every clip in the folder
    groups = sync.align([a, b, rec], clock=rec)     # the same as data
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, correlate, sosfiltfilt

SR = 8000            # analysis rate: enough for speech and for sub-millisecond lags
HOP = 80             # envelope frame (10 ms)
WIN = 8.0            # fine-alignment window (s)
MEDIA = {".mov", ".mp4", ".m4v", ".mts", ".mxf", ".avi", ".mkv", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".aif",
         ".aiff", ".bwf"}


def _probe(path: Path) -> dict:
    entries = ("stream=codec_type,width,height,r_frame_rate,sample_rate:stream_tags=timecode"
               ":stream_side_data=rotation:format=duration:format_tags=timecode,time_reference,make,model,"
               "com.apple.quicktime.creationdate,com.apple.quicktime.make,com.apple.quicktime.model")
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    streams, tags = j.get("streams", []), j.get("format", {}).get("tags", {})
    v = next((s for s in streams if s.get("codec_type") == "video" and s.get("width")), None)
    info = {"duration": float(j.get("format", {}).get("duration") or 0), "video": v is not None,
            "audio": any(s.get("codec_type") == "audio" for s in streams)}
    if v:
        num, den = (v.get("r_frame_rate") or "30/1").split("/")
        rot = next((abs(int(sd.get("rotation", 0))) for sd in v.get("side_data_list", []) if "rotation" in sd), 0)
        w, h = v["width"], v["height"]
        info.update(fps=round(int(num) / int(den), 6), size=[h, w] if rot in (90, 270) else [w, h])
    tc = tags.get("timecode") or next((s["tags"]["timecode"] for s in streams if s.get("tags", {}).get("timecode")), None)
    if tc:
        info["timecode"] = tc
        info["tc_s"] = _tc_seconds(tc, info.get("fps", 30))
    elif tags.get("time_reference"):    # a broadcast WAV's start: samples since midnight
        sr = next((int(s.get("sample_rate", 48000)) for s in streams if s.get("codec_type") == "audio"), 48000)
        info["tc_s"] = int(tags["time_reference"]) / sr
    date = tags.get("com.apple.quicktime.creationdate")
    if date:   # the phone's clock at the start of the take, to the second
        info["date_s"] = dt.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%z").timestamp()
    # Which camera shot it: one camera's files share a track in an editor (they never overlap).
    model = tags.get("com.apple.quicktime.model") or tags.get("model")
    make = tags.get("com.apple.quicktime.make") or tags.get("make")
    info["camera"] = " ".join(x for x in (make, model) if x) or \
        (f"{info['size'][0]}x{info['size'][1]}@{info['fps']:g}" if v else "audio")
    return info


def _tc_seconds(tc: str, fps: float) -> float:
    h, m, s, f = (int(x) for x in tc.replace(";", ":").split(":"))
    return h * 3600 + m * 60 + s + f / round(fps)


def _audio(path: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).astype(np.float64)
    # Speech band only: hum, rumble and hiss differ from mic to mic, syllables don't.
    return sosfiltfilt(butter(4, [150, 3400], btype="band", fs=SR, output="sos"), x) if len(x) > 100 else x


def _envelope(x: np.ndarray) -> np.ndarray:
    """Syllable onsets at 100 Hz: rises in log energy, with each mic's level and distance taken out."""
    n = len(x) // HOP
    if n < 10:
        return np.zeros(max(n, 1))
    e = np.log(np.mean(x[:n * HOP].reshape(n, HOP) ** 2, axis=1) + 1e-10)
    e -= uniform_filter1d(e, 101)
    d = np.maximum(np.diff(e, prepend=e[0]), 0)
    return (d - d.mean()) / (d.std() + 1e-9)


def _coarse(ref: np.ndarray, clip: np.ndarray, hint: tuple[float, float] | None) -> tuple[float, float]:
    """(lag s, score): clip time + lag = ref time. score is the peak's height over the spread of the rest."""
    c = correlate(ref, clip, mode="full", method="fft")
    lags = (np.arange(len(c)) - (len(clip) - 1)) * HOP / SR
    # Only lags where the two overlap by at least 5 s (or most of the shorter one) are believable.
    need = min(5.0, 0.5 * min(len(ref), len(clip)) * HOP / SR)
    ov = np.minimum(lags + len(clip) * HOP / SR, len(ref) * HOP / SR) - np.maximum(lags, 0)
    c = np.where(ov >= need, c, -np.inf)

    def best(mask):
        cc = np.where(mask, c, -np.inf)
        k = int(np.argmax(cc))
        if not np.isfinite(cc[k]):
            return 0.0, 0.0
        rest = np.abs(lags - lags[k]) > 1.0
        vals = c[np.isfinite(c) & rest]
        if len(vals) < 10:
            return float(lags[k]), 0.0
        med, mad = np.median(vals), np.median(np.abs(vals - np.median(vals))) + 1e-9
        return float(lags[k]), float((cc[k] - med) / (1.4826 * mad))

    if hint:
        lag, score = best(np.abs(lags - hint[0]) <= hint[1])
        if score >= 12:
            return lag, score
    return best(np.ones_like(c, dtype=bool))


def _gcc(r: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """Lag (samples) of c inside r, and the peak's sharpness. Partly whitened (GCC-PHAT, beta 0.8)."""
    n = 1 << int(np.ceil(np.log2(len(r) + len(c))))
    X = np.fft.rfft(r, n) * np.conj(np.fft.rfft(c, n))
    cc = np.fft.irfft(X / (np.abs(X) ** 0.8 + 1e-12), n)[: len(r) - len(c) + 1]
    k = int(np.argmax(cc))
    frac = 0.0
    if 0 < k < len(cc) - 1:
        a, b, d = cc[k - 1], cc[k], cc[k + 1]
        den = a - 2 * b + d
        frac = 0.5 * (a - d) / den if den else 0.0
    sharp = float(cc[k] / (np.std(cc) + 1e-12))
    return k + frac, sharp


def _fine(ref: np.ndarray, clip: np.ndarray, lag0: float, margin: float = 0.5) -> list[tuple[float, float, float]]:
    """(clip time, ref time, sharpness) at windows spread over the overlap."""
    m, w = int(margin * SR), int(WIN * SR)
    lo = max(0.0, -lag0) + margin
    hi = min(len(clip) / SR, len(ref) / SR - lag0) - WIN - margin
    if hi <= lo:
        lo, hi = max(0.0, -lag0), max(0.0, -lag0)
    starts = np.linspace(lo, hi, max(2, min(12, int((hi - lo) / WIN) + 1)))
    loud = np.sqrt(np.mean(clip ** 2)) if len(clip) else 0
    out = []
    for t in starts:
        a = int(t * SR)
        cw = clip[a: a + w]
        r0 = round((t + lag0) * SR) - m
        if len(cw) < w // 2 or r0 < 0 or np.sqrt(np.mean(cw ** 2)) < 0.1 * loud:
            continue       # off the end, or near silence: nothing to line up
        rw = ref[r0: r0 + len(cw) + 2 * m]
        if len(rw) < len(cw) + m:
            continue
        k, sharp = _gcc(rw, cw)
        out.append((t + len(cw) / SR / 2, (r0 + k) / SR + len(cw) / SR / 2, sharp))
    return out


def _fit(points: list[tuple[float, float, float]]) -> dict | None:
    """Straight line ref = offset + rate * clip through the windows, dropping the ones that disagree."""
    pts = [p for p in points if p[2] >= 8]          # a clear peak; noise-only windows give ~4-6
    if not pts:
        return None
    x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    keep = np.ones(len(x), bool)
    for _ in range(4):
        if keep.sum() >= 3 and np.ptp(x[keep]) >= 60:   # drift needs a minute of spread to mean anything
            rate, off = np.polyfit(x[keep], y[keep], 1)
        else:
            rate, off = 1.0, float(np.median(y[keep] - x[keep]))
        res = y - (off + rate * x)
        new = np.abs(res) <= max(0.002, 3 * np.median(np.abs(res[keep])))
        if (new == keep).all():
            break
        keep = new
    if keep.sum() == 0:
        return None
    res = (y - (off + rate * x))[keep]
    return {"offset": float(off), "rate": float(rate), "windows": f"{int(keep.sum())}/{len(points)}",
            "agree": float(keep.sum() / len(points)), "residual_ms": round(float(np.abs(res).max()) * 1000, 2)}


def pair(ref: np.ndarray, clip: np.ndarray, hint: tuple[float, float] | None = None,
         env: tuple[np.ndarray, np.ndarray] | None = None) -> dict | None:
    """Where clip sits on ref's clock: {offset, rate, score, ...}, or None when they share no sound."""
    er, ec = env or (_envelope(ref), _envelope(clip))
    lag0, score = _coarse(er, ec, hint)
    if score < 8:
        return None
    fit = _fit(_fine(ref, clip, lag0))
    if not fit or fit["agree"] < 0.5 or abs(fit["offset"] - lag0) > 0.3:
        return None
    return {**fit, "score": round(score, 1)}


def _hint(a: dict, b: dict) -> tuple[float, float] | None:
    """Expected lag of b on a's clock from timecode (to a few frames) or creation dates (to a few seconds)."""
    if "tc_s" in a and "tc_s" in b:
        return b["tc_s"] - a["tc_s"], 2.0
    if "date_s" in a and "date_s" in b:
        return b["date_s"] - a["date_s"], 10.0
    return None


def align(paths: list[Path], clock: Path | None = None) -> dict:
    """Line up every file on one clock. Returns the sync.json data."""
    paths = [Path(p).resolve() for p in paths]
    info = {p: _probe(p) for p in paths}
    if clock is None:   # a recorder with no picture first, then the longest file with sound
        withsound = [p for p in paths if info[p]["audio"]] or paths
        clock = max(withsound, key=lambda p: (not info[p]["video"], info[p]["duration"]))
    clock = Path(clock).resolve()
    if clock not in info:
        paths.insert(0, clock)
        info[clock] = _probe(clock)
    audio = {p: _audio(p) for p in paths if info[p]["audio"]}
    env = {p: _envelope(x) for p, x in audio.items()}

    placed = {clock: {"offset": 0.0, "rate": 1.0, "method": "clock"}}
    todo = sorted((p for p in paths if p != clock), key=lambda p: -info[p]["duration"])
    for _ in range(len(todo)):          # files placed through another placed file need later passes
        progress = False
        for p in list(todo):
            res = None
            for q in sorted(placed, key=lambda q: q != clock):     # the clock first, then anything already placed
                if p in audio and q in audio:
                    r = pair(audio[q], audio[p], _hint(info[q], info[p]), (env[q], env[p]))
                    method = "audio"
                elif "tc_s" in info[p] and "tc_s" in info[q]:
                    r, method = {"offset": info[p]["tc_s"] - info[q]["tc_s"], "rate": 1.0}, "timecode"
                else:
                    r = None
                if r:
                    base = placed[q]   # compose: clock = base.offset + base.rate * (r.offset + r.rate * t)
                    res = {**r, "offset": base["offset"] + base["rate"] * r["offset"],
                           "rate": base["rate"] * r["rate"], "method": method,
                           **({"via": q.name} if q != clock else {})}
                    break
            if res:
                placed[p] = res
                todo.remove(p)
                progress = True
        if not progress:
            break

    clips = []
    for p in paths:
        i, r = info[p], placed.get(p)
        row = {"name": p.name, "path": str(p), "duration": round(i["duration"], 3), "video": i["video"],
               "audio": i["audio"], "camera": i["camera"],
               **{k: i[k] for k in ("size", "fps", "timecode", "tc_s") if k in i}}
        if r:
            row.update(offset=round(r["offset"], 5), rate=round(r["rate"], 9),
                       drift_ppm=round((r["rate"] - 1) * 1e6, 1), method=r["method"],
                       **{k: r[k] for k in ("score", "windows", "residual_ms", "via") if k in r})
        else:
            row["method"] = "unplaced"
        clips.append(row)
    on = [c for c in clips if "offset" in c]
    # One camera's files never overlap: each run of them is one angle, named after its first file.
    lanes: dict[str, list] = {}
    for c in sorted(on, key=lambda c: c["offset"]):
        runs = lanes.setdefault(c["camera"], [])
        run = next((r for r in runs if r[-1]["offset"] + r[-1]["rate"] * r[-1]["duration"] <= c["offset"] + 0.05), None)
        run.append(c) if run else runs.append([c])
    for runs in lanes.values():
        for r in runs:
            for c in r:
                c["angle"] = Path(r[0]["name"]).stem
    pic = [c for c in on if c["video"]] or on
    span = [min(c["offset"] for c in pic), max(c["offset"] + c["rate"] * c["duration"] for c in pic)]
    return {"clock": clock.name, "span": [round(span[0], 3), round(span[1], 3)], "clips": clips,
            "unplaced": [c["name"] for c in clips if c["method"] == "unplaced"]}


def files(inputs: list[str | Path]) -> list[Path]:
    out = []
    for x in map(Path, inputs):
        out += sorted(f for f in x.rglob("*") if f.suffix.lower() in MEDIA and not f.name.startswith(".")) \
            if x.is_dir() else [x]
    return out


def sheet(data: dict, out: Path) -> Path:
    """Every file as a bar on the shared clock with its sound drawn in, to see the alignment at a glance."""
    from PIL import Image, ImageDraw
    on = [c for c in data["clips"] if "offset" in c]
    t0 = min(c["offset"] for c in on)
    t1 = max(c["offset"] + c["rate"] * c["duration"] for c in on)
    W, row, left = 1600, 64, 260
    img = Image.new("RGB", (W, row * len(data["clips"]) + 40), (18, 18, 22))
    d = ImageDraw.Draw(img)
    k = (W - left - 20) / max(t1 - t0, 1e-3)
    step = next(s for s in (1, 5, 10, 30, 60, 300, 600, 1800, 3600) if (t1 - t0) / s <= 14)
    for s in np.arange(np.ceil(t0 / step) * step, t1, step):
        x = left + (s - t0) * k
        d.line([(x, 0), (x, img.height - 30)], fill=(48, 48, 56))
        d.text((x + 3, img.height - 26), f"{s:g}s", fill=(150, 150, 160))
    for i, c in enumerate(data["clips"]):
        y = i * row + 8
        d.text((10, y + 8), c["name"][:34], fill=(230, 230, 235))
        d.text((10, y + 28), f"{c['method']}" + (f"  {c['offset']:+.3f}s  {c['drift_ppm']:+.0f}ppm"
                                                 if "offset" in c else ""), fill=(150, 150, 160))
        if "offset" not in c:
            continue
        a, b = left + (c["offset"] - t0) * k, left + (c["offset"] + c["rate"] * c["duration"] - t0) * k
        d.rectangle([a, y, b, y + row - 16], fill=(40, 70, 110) if c["video"] else (90, 60, 30))
        if c["audio"]:
            x = _audio(Path(c["path"]))
            n = max(1, int(b - a))
            seg = np.abs(x[: len(x) // n * n]).reshape(n, -1).max(1) if len(x) >= n else np.abs(x)
            seg = seg / (seg.max() + 1e-9)
            mid, hh = y + (row - 16) / 2, (row - 20) / 2
            for j, v in enumerate(seg):
                d.line([(a + j, mid - v * hh), (a + j, mid + v * hh)], fill=(200, 200, 210))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def write(data: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out


def register(sub) -> None:
    p = sub.add_parser("sync", help="line up every camera and recorder of one event on one clock (sync.json)")
    p.add_argument("inputs", nargs="+", help="files or folders")
    p.add_argument("--clock", help="the file whose time the others are placed on (default: a recorder, else the longest)")
    p.add_argument("-o", "--out", default="sync.json")
    p.set_defaults(fn=_cmd)


def _cmd(a) -> None:
    data = align(files(a.inputs), Path(a.clock) if a.clock else None)
    out = write(data, Path(a.out))
    png = sheet(data, out.with_suffix(".png"))
    for c in data["clips"]:
        where = (f"{c['offset']:+10.3f}s  {c['drift_ppm']:+6.1f} ppm  {c['method']:9}"
                 f" {c.get('windows', ''):>6} {c.get('residual_ms', '')!s:>5}ms" if "offset" in c else "  not placed: no shared sound")
        print(f"{c['name']:32} {where}")
    print(out, png)
