"""Pick the angle for every moment of a multicam conversation, and write it as shots in reel.json.

    juice multicam footage/sync.json --transcript footage/transcripts/rec.json -o reel.json

Who is talking comes from the strongest evidence there is:
  1. separate mics: a sound file with a channel per lav. A word belongs to the channel loudest over its own
     floor, which also catches an off-camera voice and two people talking at once.
  2. the transcript's speaker labels (ElevenLabs diarization; local whisper labels everyone the same).
  3. mouths: every camera's faces at 10 fps (YuNet), each fitted with 68 landmarks (OpenCV's LBF model) for how
     far the lips are open and how fast that changes. On two real talking-head clips cut into a conversation,
     the talker's mouth scores +1 to +3 against its own normal and the listener's -0.1 to -0.5; the change
     between frames of a cropped mouth (no landmarks) barely told them apart. This also ties mics and labels
     to cameras: a voice belongs to the close-up whose mouth opens when it speaks; a voice none follows is
     off camera.
A camera with one big face is a close-up; anything else is a wide.

What it costs is decoding video, not the face models (lip landmarks are ~1 ms a frame). Cameras are read in
parallel on the media engine at 10 fps, skipping the frames nothing refers to: 40 s of 4K60 iPhone HEVC
decodes in 4.0 s instead of 6.9 s, three cameras at once in 6.9 s. With a mic per person or speaker labels the
sound already says who talks when, so only a few stretches of each voice are looked at (to tie voices to
cameras and find the faces), however long the shoot; only mouths-alone reads the whole range (27 s of three
4K60 cameras: 15.3 s before these changes, 6.8 s now). Every frame's faces are cached per file
(~/.cache/juice/faces), so running it again over the same footage decodes nothing (0.2 s).

Then the edit, in the order it applies:
  - cut to a speaker's close-up `lead` (0.08 s) before the first word of their phrase
  - two people at once, or a voice that's off camera: the wide
  - an interjection shorter than `min_shot` stays on whoever has the floor
  - a turn longer than `max_shot` cuts away to the wide (or the listener) at a pause, then comes back
Each shot carries a "note" saying why, which `juice export` turns into a marker in Resolve.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np

from .. import encode
from . import hooks
from .geom import source_size
from .spec import Group, SpecError, load_group

FPS = 10
WIDTH = 480
LBF_URL = "https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml"
LBF = Path.home() / ".cache" / "juice" / "models" / "lbfmodel.yaml"
CACHE = Path.home() / ".cache" / "juice" / "faces"
VERSION = 1           # bump when what a cached frame holds changes
_marks = None
_marks_lock = threading.Lock()   # one landmark model for every camera's thread


@dataclass
class Subject:
    """A face one camera keeps seeing, and how much its mouth moves at each sample of the grid."""
    angle: str
    x: float = 0.0                       # where it sits across the frame (0-1)
    sizes: list = field(default_factory=list)
    mouth: np.ndarray | None = None      # per sample: lip opening over mouth width, nan where unseen
    z: np.ndarray | None = None          # how open and how fast-moving, against its own normal (robust z)

    @property
    def size(self) -> float:
        return float(np.median(self.sizes)) if self.sizes else 0.0


def _frames(path: Path, l0: float, dur: float):
    """(local time, BGR frame) at FPS, decoded and scaled on the media engine where it can be. Frames no other
    frame refers to are never decoded (-skip_frame noref): at 10 fps out of 60 the samples land on the frames
    that are left, at most a frame or two from their exact time, and decoding takes half as long."""
    sw, sh = source_size(str(path))
    h = round(WIDTH * sh / sw)
    h += h % 2
    eng = encode.engine_scale(path, WIDTH, h)
    inputs, vf = (eng[0], [f"fps={FPS}", *eng[1]]) if eng else ([], [f"fps={FPS}", f"scale={WIDTH}:{h}"])
    cmd = ["ffmpeg", "-v", "error", *inputs, "-skip_frame", "noref", "-ss", f"{max(l0, 0):.3f}", "-t", f"{dur:.3f}",
           "-an", "-i", str(path),
           "-vf", ",".join(vf + ["format=bgr24"]), "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n, k = WIDTH * h * 3, 0
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield l0 + k / FPS, np.frombuffer(buf, np.uint8).reshape(h, WIDTH, 3)
        k += 1
    proc.wait()


def _facemark():
    """OpenCV's 68-point LBF landmark model (56 MB, fetched once into ~/.cache/juice/models). Call under
    _marks_lock: one model serves every camera's thread."""
    global _marks
    if _marks is None:
        import cv2
        if not LBF.exists():
            import urllib.request
            LBF.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(LBF_URL, LBF.with_suffix(".part"))
            LBF.with_suffix(".part").replace(LBF)
        _marks = cv2.face.createFacemarkLBF()
        _marks.loadModel(str(LBF))
    return _marks


def _openings(img: np.ndarray, boxes: list) -> list[float]:
    """For each face box, how far the lips are open: inner-lip gap over mouth width (68-point landmarks)."""
    import cv2
    if not boxes:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with _marks_lock:
        ok, pts = _facemark().fit(gray, np.array(boxes, np.int32))
    out = []
    for f in (pts if ok else [None] * len(boxes)):
        if f is None:
            out.append(np.nan)
            continue
        p = np.asarray(f).reshape(-1, 2)
        gap = np.mean([np.linalg.norm(p[61] - p[67]), np.linalg.norm(p[62] - p[66]), np.linalg.norm(p[63] - p[65])])
        out.append(float(gap / (np.linalg.norm(p[60] - p[64]) + 1e-6)))
    return out


def _z(x: np.ndarray) -> np.ndarray:
    seen = x[~np.isnan(x)]
    if len(seen) < 3:
        return np.full(len(x), np.nan)
    med = np.median(seen)
    return (x - med) / (1.4826 * np.median(np.abs(seen - med)) + 1e-6)


def _scan(path: Path, k0: int, k1: int) -> dict[int, list]:
    """Faces in samples k0..k1-1 of a file's own 10 fps clock: {k: [[center x, center y, height, lip opening],
    ...]}, positions as fractions of the frame. Samples past the end of the file come back empty."""
    out = {k: [] for k in range(k0, k1)}
    for lt, img in _frames(path, k0 / FPS, (k1 - k0) / FPS):
        k = round(lt * FPS)
        if k not in out:
            continue
        H, W = img.shape[:2]
        found = [r for r in hooks.landmarks(img) if r[14] >= 0.6]
        out[k] = [[round(float((r[0] + r[2] / 2) / W), 4), round(float((r[1] + r[3] / 2) / H), 4),
                   round(float(r[3] / H), 4), None if np.isnan(op) else round(op, 4)]
                  for r, op in zip(found, _openings(img, [r[:4] for r in found]))]
    return out


def _cache_file(path: Path) -> Path:
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}|{FPS}|{WIDTH}|{VERSION}"
    return CACHE / f"{path.stem}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"


def faces(path: Path, ks) -> dict[int, list]:
    """Faces at these samples of a file's own 10 fps clock, decoding only the ones no earlier run looked at.
    Runs of missing samples less than 2 s apart are read in one go: a fresh seek costs more than the gap."""
    cf = _cache_file(Path(path))
    try:
        data = {int(k): v for k, v in json.loads(cf.read_text()).items()}
    except (OSError, ValueError):
        data = {}
    missing = sorted(k for k in set(ks) if k not in data)
    runs = []
    for k in missing:
        if runs and k - runs[-1][1] <= 2 * FPS:
            runs[-1][1] = k + 1
        else:
            runs.append([k, k + 1])
    for k0, k1 in runs:
        data.update(_scan(Path(path), k0, k1))
    if runs:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cf.with_suffix(".part")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        tmp.replace(cf)
    return {k: data[k] for k in ks if k in data}


def look(group: Group, name: str, vt: np.ndarray) -> list[Subject]:
    """Every face one angle sees at the sample times vt (group clock, sorted), with how open its mouth is."""
    subjects: list[Subject] = []
    last: dict[int, tuple] = {}            # subject index -> center last seen
    seen: dict[int, dict] = {}             # subject index -> {sample index: lip opening}
    for tk in group.angles[name].takes:
        if not tk.video:
            continue
        idx = [i for i, t in enumerate(vt) if tk.covers[0] <= t < tk.covers[1]]
        ks = {i: round(tk.local(vt[i]) * FPS) for i in idx}
        found = faces(tk.path, set(ks.values()))
        for i in idx:
            taken = set()
            for cx, cy, h, op in found.get(ks[i], []):
                best = min((j for j in last if j not in taken),
                           key=lambda j: np.hypot(cx - last[j][0], cy - last[j][1]), default=None)
                if best is None or np.hypot(cx - last[best][0], cy - last[best][1]) > 0.12:
                    subjects.append(Subject(angle=name))
                    best = len(subjects) - 1
                taken.add(best)
                subjects[best].sizes.append(h)
                last[best] = (cx, cy)
                seen.setdefault(best, {})[i] = np.nan if op is None else op
    out = []
    step = np.r_[np.inf, np.diff(vt)] if len(vt) else np.array([])
    for j, s in enumerate(subjects):
        if len(s.sizes) < 0.2 * len(vt):           # a passer-by, or a face for a moment
            continue
        m = np.full(len(vt), np.nan)
        for i, v in seen.get(j, {}).items():
            m[i] = v
        if np.sum(~np.isnan(m)) < 10:
            continue
        # Open lips, and lips moving from one sample to the next: talking does both, a smile only the first.
        # (Only between neighbouring samples: across the gap between two looked-at stretches it means nothing.)
        moving = np.where(step < 1.5 / FPS, np.abs(np.diff(m, prepend=np.nan)), np.nan)
        s.mouth, s.z = m, (_z(m) + _z(moving)) / 2
        s.x = float(last[j][0])
        out.append(s)
    return out


def _channels(path: Path, grid: np.ndarray) -> np.ndarray | None:
    """Each channel's level on the grid (dB under its own loudest speech), when the sound file has a channel
    per mic. Calibrating each mic to its own speech, not its floor, keeps a hot lav from winning everything:
    a voice reaches ~0 on its own lav and sits 10-20 dB down on the others. A camera's stereo pair is not a mic
    per person: its two sides stay within a few dB of each other (an iPhone's: never 10 dB apart, median 2 dB),
    where each of two lavs leads by more than 6 dB whenever its own person talks."""
    info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                                      "stream=channels", "-of", "json", str(path)],
                                     capture_output=True, text=True, check=True).stdout or "{}")
    ch = (info.get("streams") or [{}])[0].get("channels", 1)
    if ch < 2:
        return None
    sr = 8000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(grid[0], 0):.3f}", "-t", f"{grid[-1] - grid[0] + 1:.3f}",
                          "-i", str(path), "-map", "0:a:0", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32).reshape(-1, ch)
    hop = sr // FPS
    n = min(len(grid), len(x) // hop)
    lv = 10 * np.log10(np.mean(x[: n * hop].reshape(n, hop, ch) ** 2, axis=1) + 1e-10)
    lv -= np.percentile(lv, 95, axis=0)
    speech = lv.max(1) > -30
    if not any(np.mean(d > 6) >= 0.1 and np.mean(d < -6) >= 0.1
               for i in range(ch) for j in range(i + 1, ch) for d in [(lv[:, i] - lv[:, j])[speech]]):
        return None                                   # one place heard twice (stereo), not a mic per person
    out = np.zeros((len(grid), ch))
    out[:n] = lv
    return out


def _phrases(words: list[dict], who: list) -> list[dict]:
    """Runs of words with the same voice and no pause over 0.35 s."""
    out = []
    for w, v in zip(words, who):
        if out and out[-1]["who"] == v and w["start"] - out[-1]["end"] <= 0.35:
            out[-1]["end"] = w["end"]
            out[-1]["words"] += 1
        else:
            out.append({"start": w["start"], "end": w["end"], "who": v, "words": 1})
    return out


def _mean(x: np.ndarray, sl) -> float:
    v = x[sl]
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else float("nan")


def _windows(words: list[dict], who: list, g0: float, per_voice: int = 6, length: float = 3.0) -> np.ndarray:
    """Sample times for a few stretches of each voice talking, its longest phrases first, at least 20 s apart:
    enough to see whose mouth opens with which voice, without reading the whole recording."""
    keep = []
    for v in sorted({x for x in who if x is not None}):
        picked = []
        for p in sorted((p for p in _phrases(words, who) if p["who"] == v), key=lambda p: p["start"] - p["end"]):
            if len(picked) == per_voice:
                break
            if all(abs(p["start"] - q) >= 20 for q in picked):
                picked.append(p["start"])
                keep.append((p["start"], min(p["end"], p["start"] + length)))
    ks = sorted({k for a, b in keep for k in range(round((a - g0) * FPS), round((b - g0) * FPS) + 1)})
    return g0 + np.array(ks, float) / FPS


def plan(group: Group, words: list[dict], g0: float | None = None, g1: float | None = None,
         min_shot: float = 1.5, max_shot: float = 8.0, lead: float = 0.08) -> dict:
    """The shots (group times, an angle each, and why) for g0..g1 of a synced group."""
    cams = [k for k, a in group.angles.items() if a.video]
    if not cams:
        raise SpecError("no camera in the multicam group")
    words = [w for w in words if w.get("type", "word") == "word"]
    g0 = max(g0 if g0 is not None else (words[0]["start"] - lead if words else 0.0), group.sound.covers[0],
             min(t.covers[0] for k in cams for t in group.angles[k].takes))
    g1 = min(g1 if g1 is not None else (words[-1]["end"] + 0.3 if words else 0.0), group.sound.covers[1],
             max(t.covers[1] for k in cams for t in group.angles[k].takes))
    words = [w for w in words if g0 <= w["start"] and w["end"] <= g1]
    grid = g0 + np.arange(int((g1 - g0) * FPS) + 1) / FPS

    def span(a, b):      # grid samples a word covers
        return slice(max(0, int((a - g0) * FPS)), max(1, int(np.ceil((b - g0) * FPS))))

    # Who said each word, from the sound when it can tell: the mic loudest against its own speech level, else
    # the transcript's label. Neither: the mouths decide, further down.
    lv = _channels(group.sound.path, grid)
    if lv is not None:
        who, both = [], []
        for w in words:
            m = lv[span(w["start"], w["end"])].max(0)      # the word's peak: its silent edges would drag a mean down
            order = np.argsort(m)[::-1]
            who.append(f"mic {order[0] + 1}")
            both.append(bool(m[order[1]] >= -10 and m[order[0]] - m[order[1]] < 6))   # two voices near full level
        evidence = "mics"
    elif len({w.get("speaker_id") for w in words}) > 1:
        who, both, evidence = [w.get("speaker_id") for w in words], [False] * len(words), "labels"
        for i in range(len(words) - 1):          # one label's word inside another's: two at once
            if words[i + 1]["start"] < words[i]["end"] - 0.15 and who[i] != who[i + 1]:
                both[i] = both[i + 1] = True
    else:
        who, both, evidence = [None] * len(words), [False] * len(words), "mouths"

    # The video to look at: a few stretches of each voice when the sound already says who talks when, the whole
    # range when only mouths can. Every camera at once.
    vt = _windows(words, who, g0) if evidence != "mouths" else grid
    if not len(vt):
        vt = grid[:: max(1, len(grid) // 200)]

    def vs(a, b):        # samples looked at between a and b
        i = int(np.searchsorted(vt, a - 1e-6))
        return slice(i, max(i, int(np.searchsorted(vt, b - 1e-6))))

    with ThreadPoolExecutor(max_workers=min(4, len(cams))) as pool:
        found = dict(zip(cams, pool.map(lambda k: look(group, k, vt), cams)))
    subjects = {}
    for k in cams:     # the people a camera is pointed at; faces far smaller are the room behind them
        big = max((f.size for f in found[k]), default=0)
        subjects[k] = [f for f in found[k] if f.size >= 0.5 * big]
    close = {k: f[0] for k, f in subjects.items() if len(f) == 1 and f[0].size >= 0.1}
    wides = [k for k in cams if k not in close]
    wide = max(wides, key=lambda k: len(subjects[k])) if wides else None

    if evidence == "mouths":   # each word to the close-up whose mouth is clearly the most active around it (0.7 s)
        from scipy.ndimage import uniform_filter1d
        smooth = {}
        for k, f in close.items():
            z = np.nan_to_num(f.z, nan=0.0)
            seen = uniform_filter1d((~np.isnan(f.z)).astype(float), 7)
            smooth[k] = np.where(seen > 0.3, uniform_filter1d(z, 7) / np.maximum(seen, 1e-3), np.nan)
        who = []
        for w in words:
            z = {k: _mean(v, vs(w["start"], w["end"])) for k, v in smooth.items()}
            z = {k: x for k, x in z.items() if not np.isnan(x)}
            top = sorted(z, key=z.get, reverse=True)
            sure = top and z[top[0]] >= 0.3 and (len(top) == 1 or z[top[0]] - z[top[1]] >= 0.3)
            who.append(top[0] if sure else None)

    # One word flipping inside another voice's run is noise (a cough, a plosive into the other lav).
    for i in range(1, len(words) - 1):
        if who[i - 1] == who[i + 1] != who[i] and words[i]["end"] - words[i]["start"] < 0.6:
            who[i] = who[i - 1]
        if both[i] and not (both[i - 1] or both[i + 1]):
            both[i] = False

    # Tie each voice to the face whose mouth opens when it speaks: its close-up, and where it sits in the wide.
    voices = sorted({v for v in who if v is not None})

    def tie(faces_by: dict) -> dict:
        scores = {}
        for v in voices:
            mine, theirs = np.zeros(len(vt), bool), np.zeros(len(vt), bool)
            for w, u in zip(words, who):
                (mine if u == v else theirs)[vs(w["start"], w["end"])] = True
            for key, f in faces_by.items():
                a, b = _mean(f.z, mine), (_mean(f.z, theirs) if theirs.any() else 0.0)
                if not np.isnan(a):
                    scores[(v, key)] = a - (0.0 if np.isnan(b) else b)
        out = {}
        for (v, key), sc in sorted(scores.items(), key=lambda x: -x[1]):    # one voice per face
            if sc > 0.3 and v not in out and key not in out.values():
                out[v] = key
        return out

    cast = tie(close) if evidence != "mouths" else {k: k for k in voices}
    in_wide = tie({i: f for i, f in enumerate(subjects[wide])}) if wide else {}

    phrases = _phrases(words, [f"{v}|both" if b else v for v, b in zip(who, both)])
    picks = []
    for p in phrases:
        v = p["who"]
        if v and v.endswith("|both"):
            picks.append((wide, f"two talking at once ({v[:-5]} and another): the wide"))
        elif v is not None and v in cast:
            picks.append((cast[v], f"{v}'s mouth moves with the words" if evidence == "mouths"
                          else f"{v} talking ({evidence}), their close-up {cast[v]}"))
        elif v is not None:
            picks.append((wide, f"{v} is off camera: the wide"))
        else:
            z = {k: _mean(f.z, vs(p["start"], p["end"])) for k, f in close.items()}
            z = {k: x for k, x in z.items() if not np.isnan(x)}
            top = sorted(z, key=z.get, reverse=True)
            if top and z[top[0]] >= 0.5 and (len(top) == 1 or z[top[0]] - z[top[1]] >= 0.3):
                picks.append((top[0], f"{top[0]}'s mouth moves with the words (z {z[top[0]]:.1f})"))
            else:
                picks.append((wide, "no clear speaker: the wide"))

    # Phrases that go to the same angle are one run; a run shorter than min_shot (an interjection) stays on
    # whoever has the floor; every other run is a cut just before its first word.
    runs = []
    for p, (angle, why) in zip(phrases, picks):
        angle = angle or (runs[-1]["angle"] if runs else (wide or cams[0]))
        if runs and runs[-1]["angle"] == angle:
            runs[-1]["end"] = p["end"]
        else:
            runs.append({"angle": angle, "start": p["start"], "end": p["end"], "note": why})
    shots = []
    for r in runs:
        at = max(g0, r["start"] - lead)
        if not shots:
            shots.append({**r, "start": g0})
        elif r["angle"] == shots[-1]["angle"] or r["end"] - at < min_shot:
            shots[-1]["end"] = r["end"]
        else:
            shots[-1]["end"] = at
            shots.append({**r, "start": at})
    if not shots:
        shots = [{"angle": wide or cams[0], "start": g0, "end": g1, "note": "nobody talks: the wide"}]
    shots[-1]["end"] = g1

    # A long turn cuts away at a pause and comes back, so no shot outstays max_shot.
    gaps = [(a["end"], b["start"]) for a, b in pairwise(words) if b["start"] - a["end"] >= 0.2]
    out = []
    for s in shots:
        while s["end"] - s["start"] > max_shot:
            away = wide if wide and wide != s["angle"] else next((k for k in close if k != s["angle"]), None)
            pause = next(((a, b) for a, b in gaps if s["start"] + 0.6 * max_shot <= a and b <= s["end"] - 2 * min_shot), None)
            if away is None or pause is None:
                break
            cut = pause[1] - lead
            back = next((b - lead for a, b in gaps if b - lead >= cut + min_shot + 0.5), cut + 2.5)
            back = min(back, cut + 4.0, s["end"] - min_shot)
            out.append({**s, "end": cut})
            out.append({"angle": away, "start": cut, "end": back,
                        "note": f"long turn: cut away to {'the wide' if away == wide else 'the listener'} at a pause"})
            s = {**s, "start": back, "note": s["note"] + " (back)"}
        out.append(s)

    # Every shot on an angle that's rolling then; the same angle twice in a row is one shot.
    final = []
    for s in out:
        if not group.angles[s["angle"]].take(s["start"], s["end"]):
            alt = next((k for k in ([wide] if wide else []) + cams if group.angles[k].take(s["start"], s["end"])), None)
            if alt is None:
                raise SpecError(f"no camera rolls over all of {s['start']:.2f}-{s['end']:.2f}s")
            s = {**s, "angle": alt, "note": s["note"] + f" ({s['angle']} isn't rolling: {alt})"}
        if final and final[-1]["angle"] == s["angle"]:
            final[-1]["end"] = s["end"]
        else:
            final.append(s)
    # Frame each shot on the face that's talking in it (in the wide, the face of the voice heard most), so a
    # portrait cut from landscape cameras keeps the talker in frame.
    for s in final:
        faces_here = subjects[s["angle"]]
        if not faces_here:
            continue
        if len(faces_here) == 1:
            s["frame_x"] = round(faces_here[0].x, 3)
            continue
        heard = [v for w, v in zip(words, who) if v and s["start"] <= w["start"] < s["end"]]
        top = max(set(heard), key=heard.count) if heard else None
        if s["angle"] == wide and top in in_wide:
            talk = faces_here[in_wide[top]]
        else:
            talk = max(faces_here, key=lambda f: np.nan_to_num(_mean(f.z, vs(s["start"], s["end"])), nan=-9.0))
        s["frame_x"] = round(talk.x, 3)
    return {"range": [round(g0, 3), round(g1, 3)], "evidence": evidence, "wide": wide,
            "close": {k: {"x": round(f.x, 2), "face": round(f.size, 2)} for k, f in close.items()},
            "cast": cast, "faces": {k: len(v) for k, v in subjects.items()},
            "video_looked_at_s": round(len(vt) / FPS, 1),
            "shots": [{"angle": s["angle"], "start": round(s["start"], 3), "end": round(s["end"], 3),
                       **({"frame_x": s["frame_x"]} if "frame_x" in s else {}), "note": s["note"]} for s in final]}


def write_reel(result: dict, sync: Path, transcript: Path | None, out: Path, key: str = "pod",
               orientation: str = "landscape") -> Path:
    """Put the shots in reel.json: a new one, or the shots (and multicam group) of an existing one."""
    out = Path(out).resolve()
    raw = json.loads(out.read_text()) if out.exists() else {"orientation": orientation, "length": "long"}
    rel = lambda p: Path(p).resolve().relative_to(out.parent).as_posix() \
        if Path(p).resolve().is_relative_to(out.parent) else str(Path(p).resolve())
    raw.setdefault("multicam", {})[key] = rel(sync)
    raw["original_audio"] = True
    if transcript:
        raw.setdefault("transcripts", {})[key] = rel(transcript)
    raw["shots"] = [{"source": key, **s} for s in result["shots"]]
    out.write_text(json.dumps(raw, indent=2))
    return out


def register(sub) -> None:
    p = sub.add_parser("multicam", help="pick the angle for every moment of a synced conversation -> reel.json shots")
    p.add_argument("sync", help="sync.json from `juice sync`")
    p.add_argument("--transcript", help="word-level transcript of the sound file (default: transcribe it)")
    p.add_argument("--audio", help="which file or angle is the sound (default: sync.json's clock)")
    p.add_argument("--from", dest="start", type=float); p.add_argument("--to", dest="end", type=float)
    p.add_argument("--min-shot", type=float, default=1.5); p.add_argument("--max-shot", type=float, default=8.0)
    p.add_argument("--name", default="pod", help="the group's source name in reel.json")
    p.add_argument("-o", "--out", default="reel.json")
    p.set_defaults(fn=_cmd)


def transcript(media: Path) -> Path:
    """The sound file's words: ElevenLabs (with speaker labels) when there's a key, else local whisper."""
    from ..media import transcribe, whisper
    try:
        transcribe.api_key()
    except SystemExit:
        return whisper.transcribe(media, media.parent / "transcripts")
    return transcribe.transcribe(media, media.parent / "transcripts")


def _cmd(a) -> None:
    sync = Path(a.sync).resolve()
    group = load_group(sync.parent, {"sync": sync.name, **({"audio": a.audio} if a.audio else {})})
    tp = Path(a.transcript) if a.transcript else transcript(group.sound.path)
    words = json.loads(Path(tp).read_text()).get("words", [])
    res = plan(group, words, a.start, a.end, a.min_shot, a.max_shot)
    out = write_reel(res, sync, tp, Path(a.out), a.name)
    for s in res["shots"]:
        print(f"{s['start']:9.2f}-{s['end']:9.2f}  {s['angle']:14} {s['note']}")
    print(json.dumps({k: v for k, v in res.items() if k != "shots"}, indent=1))
    print(out)
