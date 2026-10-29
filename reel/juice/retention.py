"""Retention: predict how much of an edit people watch, and where they leave.

The model is survival analysis: every viewer is "alive" until they swipe. R(t) = exp(-∫₀ᵗ h(τ)dτ)
is the share still watching at t, and h is the hazard (swipe rate). The score is the average
percentage viewed, AVP = (1/T)∫₀ᵀ R(t)dt, the engagement metric of Wu, Rizoiu & Xie (ICWSM 2018,
"Beyond Views"), which they showed is stable per video and tracks quality, unlike view counts.

Baseline hazard: a Weibull with shape k < 1 (hazard falls with time: the swipe-off is front-loaded,
the "3-second cliff"), H₀(t) = (t/λ)^k, calibrated so a neutral edit holds 70% at 3s and 45% at 30s
(creator benchmarks: under ~70% 3-second retention reach collapses; 60-70% to the end of a 30-60s
clip is excellent; Wistia 2024/25: ~50% engagement for videos under a minute).

Per-moment multipliers on that hazard, each tied to evidence (strength noted):
  hook       first 3s: text on screen by 0.5s, speech by 0.4s, motion/face in frame one, and a
             question (information-gap curiosity, Loewenstein 1994). strong / strong / medium / theory
  captions   active word-level captions while speech is on screen. Facebook: +12% view time;
             Verizon/Publicis 2019: 80% more likely to finish with captions.             strong
  change     seconds since the last visual change (cut, graphic in/out, push-in). Kim et al. 2014:
             61% of interaction peaks sit on visual transitions; creator practice: something new
             every 1-3s.                                                                 medium
  silence    dead air in a talking video (≥ 0.5s between words) and filler words.         medium
  repeat     a phrase said again within 20s (restarts, "as I said").                      medium
  pace       speaking rate. Guo, Kim & Rubin 2014 (6.9M MOOC sessions): engagement rises with
             words per minute; slow delivery loses people.                               strong
  length     handled by the baseline: longer edits accumulate more hazard (Guo 2014, Kim 2014,
             Wistia: engagement falls with length).                                      strong
  progress   a visible progress bar ("end in sight").                                     weak

Constants are calibrated defaults, not measured on this creator's analytics; the ranking of
moments (where the hazard peaks) is more trustworthy than the absolute number. Feed real retention
curves back into BASE / WEIGHTS when they exist.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .spec import Spec
from .timeline import Timeline, build

BASE = {"k": 0.35, "lam": 57.0}
WEIGHTS = {
    "hook_text": (0.80, 1.30),      # (present, missing) multipliers over the first 3s
    "hook_speech": (0.90, 1.20),
    "hook_visual": (0.90, 1.10),
    "hook_question": 0.92,
    "captions": 0.88,
    "change_free_s": 2.0,           # seconds of no visual change before hazard starts climbing
    "change_per_s": 0.14,           # +14% hazard per further second, capped
    "change_cap": 1.7,
    "silence_min_s": 0.5,
    "silence_per_s": 0.6,           # +60% hazard per second of dead air beyond the minimum
    "silence_cap": 2.2,
    "filler": 1.3,
    "repeat": 1.35,
    "pace": {"slow": 1.22, "ok": 1.05, "good": 1.0, "fast": 0.93},   # <120, <160, <200, 200+ wpm
    "progress": 0.97,
}
FILLERS = {"um", "uh", "umm", "uhh", "erm", "hmm", "mm"}
STOP = {"the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "at", "is", "it", "i", "you", "we",
        "they", "he", "she", "that", "this", "was", "were", "be", "are", "am", "do", "did", "not", "for", "with",
        "my", "your", "me", "them", "what", "who", "there", "here", "like", "just", "very", "really", "want", "know"}
CHANGE_COMPS = {"title", "slam", "stamp", "chip", "lower_third", "versus", "vs", "flash", "endcard",
                "callout", "letterbox", "caption"}
TEXT_COMPS = {"title", "slam", "stamp", "chip", "lower_third", "versus", "vs", "caption", "captions",
              "callout", "endcard"}


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9%']", "", w.lower())


def _layer_text(L) -> str:
    p = L.params
    parts = [p.get(k) for k in ("text", "kicker", "sub", "name", "role", "premise", "title", "tagline", "left", "right")]
    parts += list(p.get("lines") or [])
    return " ".join(str(x) for x in parts if x)


def changes(spec: Spec, tl: Timeline) -> list[float]:
    """Output times where something visibly new happens: cuts, graphics in/out, a moving shot start."""
    ts = {0.0}
    ts.update(spec.cut_points())
    for L in tl.layers:
        if L.comp in CHANGE_COMPS:
            ts.add(round(L.start, 3))
            if L.end - L.start > 0.6:
                ts.add(round(L.end, 3))
    return sorted(t for t in ts if t < tl.total)


def moving(shot) -> bool:
    return (isinstance(shot.zoom, list) or shot.punch_in > 0 or shot.shake is not None
            or isinstance(shot.frame_x, list) or shot.follow is not None or shot.speed != 1.0)


def repeats(words: list[dict], window: float = 20.0, n: int = 3) -> list[tuple[float, float, str]]:
    """Word ranges that repeat an n-gram spoken within the last `window` seconds."""
    toks = [(_norm(w["text"]), w) for w in words]
    toks = [(t, w) for t, w in toks if t]
    seen: dict[tuple, float] = {}
    out = []
    i = 0
    while i <= len(toks) - n:
        gram = tuple(t for t, _ in toks[i:i + n])
        start = toks[i][1]["start"]
        if all(g in STOP for g in gram):      # "you want to" twice is English, not a restart
            i += 1
            continue
        if gram in seen and start - seen[gram] <= window:
            out.append((start, toks[i + n - 1][1]["end"], " ".join(gram)))
            i += n
            continue
        seen[gram] = start
        i += 1
    return out


def _hook_factors(spec: Spec, tl: Timeline) -> dict:
    cfg = spec.style.get("hook", {})
    f = {}
    early = [L for L in tl.layers if L.comp in TEXT_COMPS and L.start <= cfg.get("text_by", 0.5)]
    shot0 = spec.shots[0]
    has_text = bool(early) or bool(shot0.text)
    f["hook_text"] = WEIGHTS["hook_text"][0 if has_text else 1]
    first_text = " ".join(_layer_text(L) for L in early) + " " + " ".join(shot0.text.lines if shot0.text else [])
    if "?" in first_text:
        f["hook_question"] = WEIGHTS["hook_question"]
    if spec.original_audio and tl.words:
        first = tl.words[0]["start"]
        f["hook_speech"] = WEIGHTS["hook_speech"][0 if first <= cfg.get("speech_by", 0.4) else 1]
    visual = moving(shot0) or any(0 < c <= 1.5 for c in spec.cut_points()) or any(
        L.comp in CHANGE_COMPS and L.start <= 1.0 for L in tl.layers)
    if not visual and spec.output.exists():
        try:
            from .hooks import hook_visual
            r, probs = hook_visual(spec, spec.output, {"window": 1.5, **cfg})
            visual = not probs
        except Exception:
            visual = True
    f["hook_visual"] = WEIGHTS["hook_visual"][0 if visual else 1]
    return f


def score(spec: Spec, tl: Timeline | None = None, step: float = 0.1) -> dict:
    tl = tl or build(spec)
    T = tl.total
    ts = np.arange(0, T, step) + step / 2
    words = tl.words
    talking = spec.original_audio and bool(words)
    change_ts = np.array(changes(spec, tl))
    caption_layers = [L for L in tl.layers if L.comp == "captions"]
    progress = any(L.comp == "progress" for L in tl.layers)
    hook = _hook_factors(spec, tl)
    reps = repeats(words) if talking else []
    starts = np.array(tl.starts)

    # per-sample factor table
    names = ["hook", "captions", "change", "silence", "filler", "repeat", "pace", "progress"]
    F = {n: np.ones(len(ts)) for n in names}
    hook_mult = float(np.prod(list(hook.values())))
    F["hook"][ts <= 3.0] = hook_mult
    if progress:
        F["progress"][:] = WEIGHTS["progress"]
    w_start = np.array([w["start"] for w in words]) if words else np.array([])
    w_end = np.array([w["end"] for w in words]) if words else np.array([])
    for j, t in enumerate(ts):
        # visual novelty: time since the last change, counted at half speed inside a moving shot
        last = change_ts[change_ts <= t].max() if (change_ts <= t).any() else 0.0
        i = int(np.searchsorted(starts, t, side="right")) - 1
        gap = (t - last) * (0.5 if moving(spec.shots[max(0, i)]) else 1.0)
        if gap > WEIGHTS["change_free_s"]:
            F["change"][j] = min(WEIGHTS["change_cap"], 1 + WEIGHTS["change_per_s"] * (gap - WEIGHTS["change_free_s"]))
        if not talking:
            continue
        speaking = bool(((w_start <= t) & (w_end + 0.25 > t)).any())
        if speaking and any(L.start <= t < L.end for L in caption_layers):
            F["captions"][j] = WEIGHTS["captions"]
        if not speaking:
            prev = w_end[w_end <= t].max() if (w_end <= t).any() else 0.0
            nxt = w_start[w_start > t].min() if (w_start > t).any() else T
            silence = nxt - prev
            if silence >= WEIGHTS["silence_min_s"]:
                F["silence"][j] = min(WEIGHTS["silence_cap"], 1 + WEIGHTS["silence_per_s"] * (silence - WEIGHTS["silence_min_s"]))
        k = int(np.searchsorted(w_start, t, side="right")) - 1
        if k >= 0 and w_start[k] <= t < w_end[k] and _norm(words[k]["text"]) in FILLERS:
            F["filler"][j] = WEIGHTS["filler"]
        if any(a <= t < b for a, b, _ in reps):
            F["repeat"][j] = WEIGHTS["repeat"]
        # pace over a 4s window
        lo, hi = t - 2.0, t + 2.0
        n = int(((w_start >= lo) & (w_start < hi)).sum())
        if speaking:
            wpm = n / 4.0 * 60
            F["pace"][j] = WEIGHTS["pace"]["slow" if wpm < 120 else "ok" if wpm < 160 else "good" if wpm < 200 else "fast"]

    mult = np.prod(np.stack([F[n] for n in names]), axis=0)
    k, lam = BASE["k"], BASE["lam"]
    h0 = (k / lam) * np.maximum(ts / lam, 1e-6) ** (k - 1)
    H = np.cumsum(h0 * mult) * step
    R = np.exp(-H)
    R0 = np.exp(-np.cumsum(h0) * step)          # the same cut with every multiplier neutral
    avp, avp0 = float(R.mean()), float(R0.mean())

    def at(arr, t):
        return float(arr[min(len(arr) - 1, int(t / step))]) if len(arr) else 1.0

    # Where the edit bleeds: contiguous stretches where the multiplier is above 1.15, ranked by lost viewers.
    peaks = []
    j = 0
    while j < len(ts):
        if mult[j] > 1.15:
            a = j
            while j < len(ts) and mult[j] > 1.15:
                j += 1
            seg = slice(a, j)
            reasons = {n: float(F[n][seg].max()) for n in names if F[n][seg].max() > 1.0}
            lost = float((h0[seg] * (mult[seg] - 1) * R[seg]).sum() * step)   # extra share of viewers lost here
            peaks.append({"from": round(float(ts[a] - step / 2), 2), "to": round(float(ts[j - 1] + step / 2), 2),
                          "mult": round(float(mult[seg].max()), 2), "lost_pct": round(100 * lost, 2),
                          "why": {n: round(v, 2) for n, v in sorted(reasons.items(), key=lambda x: -x[1])}})
        else:
            j += 1
    peaks.sort(key=lambda p: -p["lost_pct"])

    # Missed lifts: speech without captions, graphics-free stretches, no question hook.
    tips = []
    if talking and caption_layers == []:
        tips.append("no word-level captions: add a `captions` overlay (strongest single lift, ~12% watch time)")
    if "hook_question" not in hook:
        tips.append("open on a question in the text hook (curiosity gap) to lower the 3s cliff")
    for p in peaks[:5]:
        why = ", ".join(f"{n} x{v}" for n, v in p["why"].items())
        fix = {"change": "add a cut, push-in or graphic", "silence": "cut the pause", "filler": "cut the filler",
               "repeat": "cut the repeat", "pace": "tighten the pauses between words or cut to a faster line",
               "hook": "fix the opening (text by 0.5s, speech by 0.4s, motion in frame one)"}
        main = next(iter(p["why"]))
        tips.append(f"{p['from']:.1f}-{p['to']:.1f}s loses {p['lost_pct']:.1f}% of viewers ({why}): {fix.get(main, '')}")

    return {
        "score": round(100 * avp, 1),
        "avp": round(avp, 4),
        "neutral_avp": round(avp0, 4),
        "lift_pct": round(100 * (avp - avp0) / avp0, 1),
        "retention_3s": round(at(R, 3.0), 3),
        "retention_end": round(float(R[-1]), 3),
        "expected_watch_s": round(avp * T, 2),
        "duration_s": round(T, 2),
        "hook": {k_: round(v, 2) for k_, v in hook.items()},
        "captions_coverage": round(float((F["captions"] < 1).mean() / max(1e-6, (np.array([1.0 if ((w_start <= t) & (w_end + 0.25 > t)).any() else 0.0 for t in ts]).mean()))), 2) if talking else None,
        "changes_per_10s": round(10 * len(change_ts) / max(T, 1e-6), 1),
        "peaks": peaks[:8],
        "tips": tips,
        "curve": [[round(float(t), 2), round(float(r), 4)] for t, r in zip(ts[::max(1, int(0.5 / step))], R[::max(1, int(0.5 / step))])],
        "_mult": mult.tolist(), "_ts": ts.tolist(), "_R": R.tolist(), "_R0": R0.tolist(),
        "_F": {n: F[n].tolist() for n in names},
    }


def plot(report: dict, out: Path, tl: Timeline | None = None) -> Path:
    """Retention curve (predicted vs neutral) over a hazard strip, so the weak moments are visible."""
    from PIL import Image, ImageDraw
    W, H = 1600, 520
    img = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(img)
    ts, R, R0, mult = report["_ts"], report["_R"], report["_R0"], report["_mult"]
    T = report["duration_s"]
    x0, y0, x1, y1 = 60, 30, W - 30, 330
    sx = (x1 - x0) / max(T, 1e-6)
    for frac in (0.25, 0.5, 0.75, 1.0):
        y = y1 - (y1 - y0) * frac
        d.line([(x0, y), (x1, y)], fill=(40, 40, 46))
        d.text((8, y - 6), f"{int(frac * 100)}%", fill=(140, 140, 150))
    pts0 = [(x0 + t * sx, y1 - (y1 - y0) * r) for t, r in zip(ts, R0)]
    pts = [(x0 + t * sx, y1 - (y1 - y0) * r) for t, r in zip(ts, R)]
    d.line(pts0, fill=(110, 110, 120), width=2)
    d.line(pts, fill=(255, 204, 0), width=3)
    # hazard strip: red where the multiplier is above 1, green below
    for t, m in zip(ts, mult):
        x = x0 + t * sx
        if m > 1:
            c = (int(min(255, 120 + 90 * (m - 1))), 60, 60)
        else:
            c = (50, int(min(255, 120 + 400 * (1 - m))), 80)
        d.rectangle([x, 350, x + max(1, sx * 0.1), 390], fill=c)
    for p in report["peaks"][:6]:
        xa, xb = x0 + p["from"] * sx, x0 + p["to"] * sx
        d.rectangle([xa, 392, xb, 400], fill=(255, 90, 90))
        d.text((xa, 404), f"-{p['lost_pct']:.1f}% " + ",".join(p["why"]), fill=(255, 150, 150))
    for s in range(0, int(T) + 1, 5):
        x = x0 + s * sx
        d.line([(x, y1), (x, y1 + 6)], fill=(120, 120, 130))
        d.text((x - 8, y1 + 8), f"{s}s", fill=(140, 140, 150))
    if tl is not None:
        for c in tl.starts[1:]:
            x = x0 + c * sx
            d.line([(x, y0), (x, y1)], fill=(60, 60, 70))
    d.text((x0, 440), f"retention score {report['score']}  (neutral edit {round(100 * report['neutral_avp'], 1)}, "
                      f"lift {report['lift_pct']:+}%)   3s {int(report['retention_3s'] * 100)}%   "
                      f"end {int(report['retention_end'] * 100)}%   expected watch {report['expected_watch_s']}s of {T}s",
           fill=(230, 230, 235))
    d.text((x0, 470), "yellow: this edit   grey: same cut, neutral multipliers   strip: hazard above (red) / below (green) baseline",
           fill=(150, 150, 160))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def public(report: dict) -> dict:
    return {k: v for k, v in report.items() if not k.startswith("_")}


def write(report: dict, out: Path) -> None:
    out.write_text(json.dumps(public(report), indent=2))
