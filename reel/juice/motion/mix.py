"""The soundtrack of a motion video: voice-over, the music bed, hits on the cuts, ducking, loudness.

Voice is isolated first (juice/isolate.py: room noise, chatter and traffic removed), then keeps its
own sound (high-pass, presence EQ, gentle compression). The music is set by loudness, not a fader:
measured with the BS.1770 K-weighting, it plays as loud as the voice between lines (`level` dB from
it), dips `duck` dB under every line, and is gently leveled so a loud drop doesn't jump out. The
whole mix gets one gain to -14 LUFS and a limiter at -1.5 dB, and the render reports the
balance it measured.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
from scipy import signal

from .. import beds, isolate
from ..beds import SR, put

VOICE_FX = ("highpass=f=85,equalizer=f=220:t=q:w=1:g=-2.5,equalizer=f=3200:t=q:w=1.2:g=2.5,"
            "equalizer=f=9000:t=h:w=1:g=1.5,acompressor=threshold=-22dB:ratio=3:attack=6:release=90:makeup=3")


def _decode(path: str, af: str | None = None, channels=1) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-ac", str(channels), "-ar", str(SR)]
    if af:
        cmd += ["-af", af]
    p = subprocess.run(cmd + ["-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"no audio in {path}: {p.stderr.decode()[-300:]}")
    x = np.frombuffer(p.stdout, np.float32)
    return x.reshape(-1, channels) if channels > 1 else x


def hit_sound(h, B=0.5) -> list[tuple[np.ndarray, float, float]]:
    """(samples, start time, gain) for one hit."""
    a = h.amount
    if h.kind == "boom":
        return [(beds.boom(), h.at, 0.9 * a), (beds.braam(2.6), h.at, 0.35 * a)]
    if h.kind == "slam":
        return [(beds.boom(1.4, 110, 45), h.at, 0.45 * a), (beds.snare(0.3, 160), h.at, 0.25 * a)]
    if h.kind == "land":
        return [(beds.boom(2.2, 90, 40), h.at, 0.75 * a)]
    if h.kind == "braam":
        return [(beds.braam(3.8), h.at, 0.6 * a)]
    if h.kind == "whoosh":
        return [(beds.whoosh(0.6), h.at, 0.4 * a)]
    if h.kind == "pop":
        return [(beds.pop(), h.at, 0.35 * a)]
    if h.kind == "click":
        return [(beds.click(), h.at, 0.4 * a)]
    if h.kind == "glitch":
        return [(beds.glitch(0.18), h.at - 0.05, 0.3 * a)]
    if h.kind == "riser":
        d = max(0.3, h.at - (h.until if h.until is not None else h.at - 1.2))
        return [(beds.riser(d), h.at - d, 0.4 * a)]
    if h.kind == "sub":
        return [(beds.sub(), h.at, 0.55 * a)]
    if h.kind == "tick":
        return [(beds.tick(0.05), h.at, 0.22 * a)]
    if h.kind == "air":
        return [(beds.air(0.5), h.at, 0.3 * a)]
    if h.kind == "swell":
        d = max(0.4, h.at - (h.until if h.until is not None else h.at - 1.6))
        return [(beds.swell(d), h.at - d, 0.35 * a)]
    if h.kind == "type" and h.until:
        n = max(3, int((h.until - h.at) * 14))
        return [(beds.key(), h.at + (h.until - h.at) * k / n, 0.35 * a) for k in range(n)]
    return []


# --- loudness (ITU-R BS.1770: K-weighting, 400 ms blocks, gating) ---------------------------------

_K = [([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]),
      ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])]   # 48 kHz coefficients
HOP = 0.1


def _power(x: np.ndarray) -> np.ndarray:
    """K-weighted mean-square power per 100 ms hop, summed over channels."""
    x = x if x.ndim == 2 else x[:, None]
    for b, a in _K:
        x = signal.lfilter(b, a, x, axis=0)
    h = int(HOP * SR)
    n = len(x) // h
    return (x[:n * h] ** 2).reshape(n, h, -1).mean(1).sum(1)


def loudness(x: np.ndarray, region: np.ndarray | None = None) -> float:
    """Integrated loudness (LUFS) of x, optionally only over the samples where region is True."""
    pw = _power(x)
    blocks = np.convolve(pw, np.ones(4) / 4, "valid")      # 400 ms blocks, 75% overlap
    if region is not None:
        h = int(HOP * SR)
        keep = region[:len(pw) * h].reshape(len(pw), h).mean(1) > 0.5
        blocks = blocks[np.convolve(keep, np.ones(4), "valid") >= 4]
    blocks = blocks[blocks > 10 ** ((-70 + 0.691) / 10)]
    if not len(blocks):
        return -70.0
    blocks = blocks[blocks > blocks.mean() / 10]            # relative gate, -10 LU
    return -0.691 + 10 * np.log10(blocks.mean())


def short_term(x: np.ndarray, window=1.5) -> np.ndarray:
    """Loudness over a sliding window, per sample (LUFS)."""
    pw = _power(x)
    k = max(1, int(window / HOP))
    db = -0.691 + 10 * np.log10(np.maximum(np.convolve(pw, np.ones(k) / k, "same"), 1e-12))
    h = int(HOP * SR)
    return np.interp(np.arange(len(x)), np.arange(len(db)) * h + h / 2, db)


# --- the mix --------------------------------------------------------------------------------------

def _voices(video) -> dict[str, str]:
    """Each take's audio source: its isolated voice when asked for (and available)."""
    want = sorted({v.path for v in video.voices if v.isolate})
    if not want:
        return {}
    try:
        return {str(k): str(p) for k, p in isolate.voices([Path(w) for w in want]).items()}
    except (RuntimeError, OSError, subprocess.CalledProcessError) as e:
        warnings.warn(f"{e}; mixing the raw voice")
        return {}


def render(video, out: Path) -> Path:
    dur = video.duration
    N = int(dur * SR)
    vo = np.zeros(N, np.float32)
    iso = _voices(video)
    takes = {}
    for v in video.voices:
        src = iso.get(str(Path(v.path)), v.path) if v.isolate else v.path
        if src not in takes:
            takes[src] = _decode(src, VOICE_FX)
        seg = takes[src][int(v.start * SR):int(v.end * SR)].copy()
        f = int(0.012 * SR)
        if len(seg) > 2 * f:
            ramp = np.linspace(0, 1, f)
            seg[:f] *= ramp
            seg[-f:] *= ramp[::-1]
        i = int(v.at * SR)
        m = min(len(seg), N - i)
        if m > 0:
            vo[i:i + m] += seg[:m]
    if np.abs(vo).max() > 0:
        vo *= 0.7 / np.abs(vo).max()

    music = np.zeros((N, 2))
    bed = video.bed or {}
    if bed:
        if bed.get("file"):
            x = _decode(str(bed["file"]), channels=2)
        else:
            x = beds.read(beds.bed(bed["style"], dur, video.sections(), video._beat0(), bed.get("bpm")))
        m = min(len(x), N)
        music[:m] = x[:m]
    fx = np.zeros((N, 2))
    for h in video._all_hits():
        for snd, t, g in hit_sound(h):
            put(fx, snd, t, g)
    fx = beds.verb(fx, beds.reverb_ir(3.2, 2.2), 0.3)

    # Where the voice is: 80 ms in, 250 ms out.
    talk = np.zeros(N)
    for v in video.voices:
        talk[int(v.at * SR):int((v.at + v.end - v.start) * SR)] = 1.0
    sos = signal.butter(1, 4.0, fs=SR, output="sos")
    mask = np.clip(np.maximum(signal.sosfilt(sos, talk), signal.sosfilt(sos, talk[::-1])[::-1]) * 1.6, 0, 1)

    # Music by loudness: as loud as the voice between lines (+ level), duck dB under them, and leveled
    # over 2 s windows (within 9 dB) so every section sits at the same height as the voice.
    level, duck = bed.get("level", 0.0), bed.get("duck", 12.0)
    voice_l = loudness(np.stack([vo, vo], 1), talk > 0.5) if talk.any() else -23.0   # as mixed: both sides
    gaps = mask < 0.05
    curve = np.zeros(N)
    g0 = 1.0
    if np.abs(music).max() > 1e-6:
        music_l = loudness(music, gaps) if gaps.sum() > SR else loudness(music)
        g0 = 10 ** ((voice_l + level - music_l) / 20)
        st = short_term(music * g0, 2.0)
        corr = np.where(st > -50, np.clip(voice_l + level - st, -9, 9), 0.0)
        corr = signal.sosfiltfilt(signal.butter(1, 0.6, fs=SR, output="sos"), corr)
        curve = g0 * 10 ** (corr / 20)
        if gaps.sum() > SR:   # land the leveled music exactly on its target between the lines
            curve *= 10 ** ((voice_l + level - loudness(music * curve[:, None], gaps)) / 20)
        curve *= 1 - (1 - 10 ** (-duck / 20)) * mask
    mix = music * curve[:, None] + fx * g0 * 0.9 * (1 - 0.6 * mask)[:, None] + np.stack([vo, vo], 1)
    tail = int(0.6 * SR)
    mix[-tail:] *= (np.linspace(1, 0, tail) ** 2)[:, None]
    # One gain to -14 LUFS (a dynamic normalizer would undo the balance above), then a limiter at -1.5 dB.
    mix *= 10 ** ((-14.0 - loudness(mix)) / 20)
    raw = out.with_name(out.stem + "_raw.wav")
    head = max(1.0, float(np.abs(mix).max()))   # written with headroom, the gain restored before the limiter
    beds.write(raw, mix / head)
    r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af",
                        f"volume={head:.4f},alimiter=limit=0.84:attack=1:release=60:level=false", "-ar", str(SR), str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])
    raw.unlink()
    final = beds.read(out)
    m = min(len(final), N)
    body = np.arange(m) < m - int(1.2 * SR)   # leave out the fade
    music_only = gaps[:m] & body
    video.audio_report = {
        "isolated": sorted(Path(p).name for p in iso),
        "lufs": round(loudness(final[:m]), 1),
        "voice_lufs": round(loudness(final[:m], (talk[:m] > 0.5) & body), 1),
        "music_lufs": round(loudness(final[:m], music_only), 1) if music_only.sum() > SR else None,
    }
    return out
