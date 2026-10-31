"""Hooks: rank the strongest openings across a pile of clips, and check that an edit opens on a
triple hook.

Triple hook: in the first ~1.5s three hooks land together, on the same idea.
  spoken  the first words start almost immediately, and not on a warm-up ("okay so...")
  text    on-screen words frame it by ~0.5s and read in one glance (one short line)
  visual  frame one is worth looking at: a face, motion, something happening

`candidates` scores every transcript phrase on the signals a hook tends to have (stands on its own,
short, not a filler start, not the interviewer's question, a laugh right after, energy, a face on
screen). It's a shortlist for the editor, not the decision: the wording still has to be funny,
surprising or wrong-sounding, which needs judgment.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..media import transcribe

FILLERS = {"um", "uh", "umm", "uhh", "okay", "ok", "so", "like", "well", "yeah", "hmm", "mm", "alright", "right"}
TEXT_COMPS = {"chip", "caption", "slam", "stamp", "title", "versus", "vs", "lower_third", "callout", "captions",
              "endcard"}
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
YUNET = Path.home() / ".cache" / "juice" / "models" / "face_detection_yunet_2023mar.onnx"
HOOK_DEFAULTS = {"check": True, "window": 1.5, "speech_by": 0.4, "text_by": 0.5, "max_words": 10,
                 "require_spoken": "auto"}


def _norm(w: str) -> str:
    return re.sub(r"[^a-z']", "", w.lower())


# --- frames, faces, audio -------------------------------------------------------------------------

def _frame(media: Path, t: float, width: int = 360) -> np.ndarray | None:
    """One BGR frame at time t (rotation applied), or None past the end."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(media), "-frames:v", "1",
                        "-vf", f"scale={width}:-2", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                       capture_output=True)
    if not r.stdout:
        return None
    h = len(r.stdout) // (width * 3)
    return np.frombuffer(r.stdout[: width * 3 * h], np.uint8).reshape(h, width, 3)


_detector = None
_lock = threading.Lock()   # one YuNet net, not thread-safe: input size and forward pass must not interleave


def faces(img: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Face boxes (x, y, w, h) with YuNet; [] if the model can't be fetched (offline)."""
    with _lock:
        return _faces(img)


def _faces(img: np.ndarray) -> list[tuple[float, float, float, float]]:
    global _detector
    import cv2
    if _detector is None:
        if not YUNET.exists():
            try:
                YUNET.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(YUNET_URL, YUNET)
            except OSError:
                _detector = False
        if _detector is None:
            for quiet in (getattr(cv2, "setLogLevel", None), getattr(getattr(cv2, "utils", None), "logging", None)):
                try:
                    quiet(2) if callable(quiet) else quiet.setLogLevel(quiet.LOG_LEVEL_ERROR)
                    break
                except Exception:
                    continue
            _detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.7)
    if _detector is False:
        return []
    _detector.setInputSize((img.shape[1], img.shape[0]))
    _, found = _detector.detect(img)
    return [] if found is None else [tuple(float(v) for v in f[:4]) for f in found]


def _audio(media: Path, sr: int = 16000) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media), "-map", "0:a:0", "-ac", "1", "-ar", str(sr),
                          "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def _db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)) if len(x) else -120.0


# --- candidates -----------------------------------------------------------------------------------

def score(f: dict) -> tuple[float, list[str]]:
    """Weighted hook score from phrase features, with the reasons that moved it. Pure: easy to tune/test."""
    s, why = 0.0, []
    d = f["dur"]
    if 0.8 <= d <= 4.0:
        s += 2.0
        why.append("short")
    elif d < 0.8:
        s += 0.5
    else:
        s += max(0.0, 2.0 - (d - 4.0) * 0.6)
    if f["complete"]:
        s += 1.5
        why.append("stands alone")
    if f["filler_start"]:
        s -= 1.5
        why.append("filler start")
    if f["interviewer_question"]:
        s -= 4.0
        why.append("interviewer question")
    elif f.get("interviewer"):
        s -= 2.5                                      # the hook comes from the person on camera
        why.append("interviewer line")
    if f["laugh_after"]:
        s += 2.5
        why.append("laugh after")
    if f["reaction_after"]:
        s += 1.0
        why.append("reaction after")
    s += max(-1.5, min(1.5, f["energy_db"] / 4))      # dB above the clip's median speech level
    if f["energy_db"] > 3:
        why.append("energetic")
    if f["face"] > 0:
        s += 1.0 + min(1.0, f["face"] * 10)           # bigger face = closer, more watchable
        why.append("face on screen")
    s += min(1.0, f["motion"] / 10)
    return round(s, 2), why


def _speaker_roles(phrases) -> str | None:
    """The interviewer: the speaker who asks the most questions (None if nobody clearly does)."""
    q = {}
    for _, _, spk, txt in phrases:
        q[spk] = q.get(spk, 0) + txt.count("?")
    if len(q) < 2:
        return None
    spk, n = max(q.items(), key=lambda kv: kv[1])
    return spk if n >= 1 else None


def clip_candidates(media: Path, transcripts_dir: Path | None = None) -> list[dict]:
    media = Path(media)
    tpath = transcribe.transcribe(media, transcripts_dir or media.parent / "transcripts")
    data = json.loads(Path(tpath).read_text())
    events = [w for w in data.get("words", []) if w.get("type") == "audio_event"]
    phrases = transcribe.phrases(tpath, gap=0.5)
    interviewer = _speaker_roles(phrases)
    audio = _audio(media)
    sr = 16000
    speech = [audio[int(a * sr):int(b * sr)] for a, b, _, _ in phrases]
    med = float(np.median([_db(x) for x in speech])) if speech else -40.0

    def one(i):
        a, b, spk, txt = phrases[i]
        if not re.search(r"[a-zA-Z]", re.sub(r"\[[^\]]*\]", "", txt)):
            return None
        words = [_norm(w) for w in txt.split()]
        nxt = phrases[i + 1] if i + 1 < len(phrases) else None
        img = _frame(media, a + min(0.3, (b - a) / 2))
        img2 = _frame(media, a + min(0.8, (b - a)))
        area = 0.0
        if img is not None:
            fs = faces(img)
            if fs:
                area = max(w * h for _, _, w, h in fs) / (img.shape[0] * img.shape[1])
        motion = float(np.mean(np.abs(img.astype(int) - img2.astype(int)))) if img is not None and img2 is not None \
            and img.shape == img2.shape else 0.0
        f = {
            "dur": b - a,
            "complete": bool(re.search(r"[.!?]\W*$", txt.strip())) and not txt.strip().endswith("-"),
            "filler_start": bool(words) and words[0] in FILLERS,
            "interviewer": spk == interviewer,
            "interviewer_question": spk == interviewer and "?" in txt,
            "laugh_after": any("laugh" in e["text"].lower() and a - 0.3 <= e["start"] <= b + 2.0 for e in events),
            "reaction_after": bool(nxt) and nxt[2] != spk and nxt[0] - b < 1.5 and len(nxt[3].split()) <= 4
                              and "?" not in nxt[3],
            "energy_db": _db(audio[int(a * sr):int(b * sr)]) - med,
            "face": area,
            "motion": motion,
        }
        sc, why = score(f)
        return {"clip": media.name, "start": round(a, 2), "end": round(b, 2), "speaker": spk, "text": txt,
                "score": sc, "why": why, "frame_t": round(a + min(0.3, (b - a) / 2), 2)}

    with ThreadPoolExecutor(max_workers=6) as pool:
        out = [c for c in pool.map(one, range(len(phrases))) if c]
    return out


def candidates(clips: list[Path], top: int = 10) -> list[dict]:
    allc = []
    for c in clips:
        allc += clip_candidates(Path(c))
    return sorted(allc, key=lambda c: -c["score"])[:top]


def sheet(clips_by_name: dict[str, Path], cands: list[dict], out: Path, cols: int = 5) -> Path:
    """Contact sheet of the ranked candidates' frames, labeled with rank, score and the line."""
    from PIL import Image, ImageDraw
    tiles = []
    for i, c in enumerate(cands):
        img = _frame(clips_by_name[c["clip"]], c["frame_t"], 300)
        if img is None:
            continue
        im = Image.fromarray(img[:, :, ::-1].copy())
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, im.width, 44], fill=(0, 0, 0))
        d.text((6, 4), f"#{i + 1}  {c['score']}  {c['clip']} {c['start']:.2f}s", fill=(255, 220, 0))
        d.text((6, 24), c["text"][:46], fill=(255, 255, 255))
        tiles.append(im)
    if not tiles:
        raise SystemExit("no candidates")
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    s = Image.new("RGB", (w * min(cols, len(tiles)), h * rows), (18, 18, 22))
    for i, im in enumerate(tiles):
        s.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
    s.save(out)
    return out


# --- triple-hook check ----------------------------------------------------------------------------

def _layer_words(L) -> int:
    p = L.params
    if L.comp == "captions":
        return int(p.get("max_words", 3))
    keys = ("text", "sub", "kicker", "left", "right", "premise", "name", "role", "label", "title", "tagline")
    return sum(len(str(p[k]).split()) for k in keys if p.get(k))


def hook_text(spec, tl, cfg: dict) -> tuple[dict, list[str]]:
    """Text hook: something to read by text_by, and short enough to read in one glance."""
    window, by, maxw = cfg["window"], cfg["text_by"], cfg["max_words"]
    early = [L for L in tl.layers if L.comp in TEXT_COMPS and L.start <= by and L.end > 0]
    shot0 = spec.shots[0].text
    words0 = sum(len(l.split()) for l in shot0.lines) if shot0 else 0
    first = min([L.start for L in early] + ([0.0] if shot0 else []), default=None)
    # Heaviest reading load at any moment in the window.
    load = 0
    for t in np.arange(0, window + 1e-6, 0.1):
        active = [L for L in tl.layers if L.comp in TEXT_COMPS and L.start <= t < L.end]
        load = max(load, sum(_layer_words(L) for L in active) + (words0 if t < tl.durations[0] else 0))
    probs = []
    if first is None:
        probs.append(f"hook: no on-screen text by {by}s (text hook: frame the video in one short line)")
    if load > maxw:
        probs.append(f"hook: {load} words on screen in the first {window}s; the text hook should read in one "
                     f"glance (≤ {maxw} words, one line). Bring extra labels in after the hook")
    return {"text_at": first, "words": load}, probs


def _transcript_for_source(spec, key: str) -> Path | None:
    src = Path(spec.sources[key])
    if key in spec.transcripts:
        return Path(spec.transcripts[key])
    for d in (spec.path.parent / "transcripts", src.parent / "transcripts"):
        p = d / f"{src.stem}.json"
        if p.exists():
            return p
    return None


def hook_spoken(spec, cfg: dict) -> tuple[dict, list[str]]:
    """Spoken hook: first word of the first shot starts by speech_by, and isn't a filler."""
    need = cfg["require_spoken"]
    if need == "auto":
        need = spec.original_audio
    shot = spec.shots[0]
    tp = _transcript_for_source(spec, shot.source)
    if not need:
        return {"first_word_at": None, "first_word": None, "note": "no spoken hook expected"}, []
    if tp is None:
        return {"first_word_at": None, "note": "no transcript for the first shot; spoken hook not checked"}, []
    words = [w for w in json.loads(tp.read_text()).get("words", []) if w.get("type") == "word"]
    inside = [w for w in words if w["end"] > shot.start + 0.03 and w["start"] < shot.end]
    if not inside:
        return {"first_word_at": None}, [f"hook: nothing is said in the first shot; open on a line"]
    w0 = inside[0]
    at = max(0.0, (w0["start"] - shot.start) / shot.speed)
    probs = []
    if at > cfg["speech_by"]:
        probs.append(f"hook: first words start at {at:.2f}s; the spoken hook should start by {cfg['speech_by']}s "
                     f"(trim the shot's start to just before \"{w0['text'].strip()}\")")
    if _norm(w0["text"]) in FILLERS:
        probs.append(f"hook: the video opens on a filler (\"{w0['text'].strip()}\"); start on the line itself")
    if w0["start"] < shot.start - 0.02:
        probs.append(f"hook: the first shot starts mid-word (\"{w0['text'].strip()}\")")
    return {"first_word_at": round(at, 2), "first_word": w0["text"].strip()}, probs


def hook_visual(spec, out: Path, cfg: dict) -> tuple[dict, list[str]]:
    """Visual hook: frame one has detail, a face or motion in the first window."""
    times = [0.0, 0.25, 0.5, 1.0, cfg["window"]]
    imgs = [_frame(out, t, 270) for t in times]
    imgs = [i for i in imgs if i is not None]
    if not imgs:
        return {}, []
    detail = float(np.std(imgs[0]))
    motion = max((float(np.mean(np.abs(a.astype(int) - b.astype(int)))) for a, b in zip(imgs, imgs[1:])
                  if a.shape == b.shape), default=0.0)
    big = _frame(out, 0.25, 480)
    fs = faces(big) if big is not None else []
    face = max((w * h for _, _, w, h in fs), default=0) / (big.shape[0] * big.shape[1]) if fs else 0.0
    shot = spec.shots[0]
    moves = shot.zoom != 1.0 or shot.punch_in or shot.shake or isinstance(shot.frame_x, list) or shot.follow
    probs = []
    if detail < 12:
        probs.append("hook: the opening frame is nearly empty; open on the subject")
    if motion < 2.0 and not moves and not fs:
        probs.append("hook: the opening is static with no face; add motion (a push-in, a cut by ~1s) or open on a face")
    return {"detail": round(detail, 1), "motion": round(motion, 2), "faces": len(fs), "face_area": round(face, 3)}, probs


def triple_hook(spec, tl, out: Path) -> tuple[dict, list[str]]:
    cfg = {**HOOK_DEFAULTS, **spec.style.get("hook", {})}
    if not cfg["check"]:
        return {"checked": False}, []
    report, probs = {}, []
    for name, (r, p) in (("spoken", hook_spoken(spec, cfg)), ("text", hook_text(spec, tl, cfg)),
                         ("visual", hook_visual(spec, out, cfg))):
        report[name] = r
        probs += p
    report["triple"] = not probs
    return report, probs
