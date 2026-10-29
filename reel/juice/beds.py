"""Music beds and hit sounds, synthesized: original, so there's nothing to license or credit.

A bed is a style (tempo, key, chords, instruments) arranged over a video's sections: each section
has an energy from 0 (drone + ticks) to 3 (full groove + arpeggio), bars line up from `beat0` (put
the first big cut on it), and risers lead into every lift. Rendered beds are cached by their
arrangement, so a re-render with the same cut costs nothing.

    juice beds            writes a 30 s preview of every style to ~/.cache/juice/beds/library
"""
from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
from scipy import signal

SR = 48000
CACHE = Path.home() / ".cache" / "juice" / "beds"

# Chords as MIDI notes; roots one octave down for the bass.
STYLES = {
    "hype": {"bpm": 120, "about": "launch trailers, hype edits: punchy drums, 16th bass, bright arps",
             "chords": [[50, 53, 57, 62], [46, 50, 53, 58], [45, 48, 53, 57], [48, 52, 55, 60]],
             "roots": [38, 46, 41, 48], "kick": 1.0, "hats": 1.0, "clap": 1.0, "arp": 1.0, "bright": 1.0},
    "tech": {"bpm": 110, "about": "product demos, explainers: steady pulse, plucky arps, lighter drums",
             "chords": [[57, 60, 64, 69], [53, 57, 60, 65], [48, 52, 55, 60], [55, 59, 62, 67]],
             "roots": [45, 41, 48, 43], "kick": 0.7, "hats": 0.8, "clap": 0.5, "arp": 1.3, "bright": 0.8},
    "calm": {"bpm": 84, "about": "stories, documentary, long form: warm pads, soft pulse, no hard drums",
             "chords": [[53, 57, 60, 65], [48, 52, 55, 60], [50, 53, 57, 62], [46, 50, 53, 58]],
             "roots": [41, 48, 50, 46], "kick": 0.35, "hats": 0.3, "clap": 0.0, "arp": 0.7, "bright": 0.5},
    "keynote": {"bpm": 120, "about": "launches, enterprise trailers: soft-key ostinato, warm sub, pumping pads, tight drums",
                "chords": [[50, 54, 57, 64], [47, 54, 57, 62], [55, 59, 62, 66], [52, 57, 61, 64]],
                "roots": [38, 35, 43, 45], "kick": 0.6, "hats": 0.4, "clap": 0.2, "arp": 1.1, "bright": 0.55,
                "keys": True, "pump": 0.55, "arp_step": 0.5},
    "future": {"bpm": 120, "about": "aesthetic, futuristic: wide detuned pads, glass bells in a ping-pong echo, deep "
                                    "sub, half-time drums, air",
               "chords": [[53, 57, 60, 64, 67], [55, 59, 62, 64, 69], [52, 55, 59, 62, 66], [57, 60, 64, 67, 71]],
               "roots": [41, 43, 40, 45], "engine": "future", "kick": 0.8, "hats": 0.5, "clap": 0.5, "arp": 1.0,
               "bright": 0.7},
}


def _rng(*key):
    # Seeded from the key's text (not hash(), which changes per process) so beds are reproducible.
    return np.random.default_rng(int(hashlib.sha1(repr(key).encode()).hexdigest()[:8], 16))


# --- synth ----------------------------------------------------------------------------------------

def env(n, a, d, s, r):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    sus = max(n - a - d - r, 0)
    e = np.concatenate([np.linspace(0, 1, max(a, 1)), np.linspace(1, s, max(d, 1)), np.full(sus, s),
                        np.linspace(s, 0, max(r, 1))])
    return np.pad(e, (0, max(0, n - len(e))))[:n]


def saw(f, n, phase=0.0):
    return 2 * ((phase + np.cumsum(np.broadcast_to(f, (n,)) / SR)) % 1.0) - 1


def sine(f, n, phase=0.0):
    return np.sin(phase + 2 * np.pi * np.cumsum(np.broadcast_to(f, (n,)) / SR))


def lp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, min(fc, SR * 0.45), "low", fs=SR, output="sos"), x)


def hp(x, fc, order=2):
    return signal.sosfilt(signal.butter(order, fc, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi, order=2):
    return signal.sosfilt(signal.butter(order, [lo, hi], "band", fs=SR, output="sos"), x)


def sweep_lp(x, f0, f1, block=256):
    """Lowpass whose cutoff sweeps exponentially from f0 to f1 (filter state carried across blocks)."""
    out, zi = np.zeros_like(x), np.zeros(2)
    n = len(x)
    for i in range(0, n, block):
        b, a = signal.butter(2, min(f0 * (f1 / f0) ** (i / max(n - 1, 1)), SR * 0.45), "low", fs=SR)
        out[i:i + block], zi = signal.lfilter(b, a, x[i:i + block], zi=zi)
    return out


def mtof(m):
    return 440 * 2 ** ((m - 69) / 12)


def noise(n, *key):
    return _rng("noise", n, *key).standard_normal(n)


def kick(dur=0.45, f0=150, f1=42):
    n = int(dur * SR)
    t = np.arange(n) / SR
    body = sine(f1 + (f0 - f1) * np.exp(-t * 28), n) * np.exp(-t * 7.5)
    click = hp(noise(n, "kick"), 2500) * np.exp(-t * 300) * 0.25
    return np.tanh(1.6 * (body + click)) * 0.9


def boom(dur=3.5, f0=70, f1=28):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sub = sine(f1 + (f0 - f1) * np.exp(-t * 6), n) * np.exp(-t * 1.25)
    crack = lp(noise(n, "crack"), 2400) * np.exp(-t * 14) * 0.6
    thud = lp(noise(n, "thud"), 180) * np.exp(-t * 5) * 1.2
    return np.tanh(1.3 * (sub + crack + thud))


def braam(dur=2.6, root=38):
    n = int(dur * SR)
    t = np.arange(n) / SR
    r = _rng("braam", root)
    x = sum(g * saw(mtof(m + det), n, r.random()) for m, g in ((root, 1.0), (root + 12, 0.7), (root + 19, 0.35),
                                                                 (root + 24, 0.25)) for det in (-0.12, 0.0, 0.11))
    cut = 180 + 2600 * np.exp(-t * 2.2) * (1 - np.exp(-t * 40))
    y, zi = np.zeros(n), np.zeros(2)
    for i in range(0, n, 256):
        b, a = signal.butter(2, cut[i], "low", fs=SR)
        y[i:i + 256], zi = signal.lfilter(b, a, x[i:i + 256], zi=zi)
    return np.tanh(y * 0.35) * env(n, 0.01, 0.4, 0.55, 1.6)


def snare(dur=0.25, tone=190):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return bp(noise(n, "snare"), 1200, 9000) * np.exp(-t * 22) * 0.7 + sine(tone, n) * np.exp(-t * 30) * 0.5


def clap(dur=0.3):
    n = int(dur * SR)
    t = np.arange(n) / SR
    e = np.zeros(n)
    for k, off in enumerate((0, 0.009, 0.018)):
        i = int(off * SR)
        e[i:] += np.exp(-t[:n - i] * (60 if k < 2 else 16))
    return bp(noise(n, "clap"), 900, 6000) * e * 0.6


def hat(dur=0.06):
    n = int(dur * SR)
    return hp(noise(n, "hat"), 7500) * np.exp(-np.arange(n) / SR * 70) * 0.35


def tick(dur=0.03):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (sine(3200, n) * 0.6 + hp(noise(n, "tick"), 5000) * 0.4) * np.exp(-t * 180)


def pluck(m, dur=0.22, bright=3500):
    n = int(dur * SR)
    r = _rng("pluck", m)
    x = saw(mtof(m), n, r.random()) * 0.6 + saw(mtof(m + 0.08), n, r.random()) * 0.4
    return sweep_lp(x, bright, 400) * np.exp(-np.arange(n) / SR * 11)


def keys(m, dur=0.5):
    """A soft electric-piano note: sine partials, a quick bloom and a long decay (no saw buzz)."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = mtof(m)
    x = sine(f, n) + 0.32 * sine(2 * f, n) * np.exp(-t * 6) + 0.12 * sine(3 * f + 1.5, n) * np.exp(-t * 12)
    return x * np.exp(-t * 4.5) * (1 - np.exp(-t * 400)) * 0.5


def bass(m, dur, cutoff=900):
    n = int(dur * SR)
    x = saw(mtof(m), n) * 0.7 + sine(mtof(m - 12), n) * 0.6
    return np.tanh(1.4 * sweep_lp(x, cutoff, 160)) * env(n, 0.004, 0.08, 0.6, 0.03)


def pad(ms, dur, cut=1400, att=0.5, rel=0.8):
    n = int(dur * SR)
    r = _rng("pad", tuple(ms))
    x = sum(saw(mtof(m + det), n, r.random()) for m in ms for det in (-0.09, 0.0, 0.1))
    return lp(x, cut) * env(n, att, 0.2, 0.9, rel) / (len(ms) * 3)


def riser(dur, f0=300, f1=9000, tone=True):
    n = int(dur * SR)
    p = np.arange(n) / n
    src, y = noise(n, "riser", dur), np.zeros(n)
    zi = None
    for i in range(0, n, 256):
        fc = f0 * (f1 / f0) ** (i / n)
        sos = signal.butter(2, [max(fc * 0.7, 30), min(fc * 1.3, SR * 0.45)], "band", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        y[i:i + 256], zi = signal.sosfilt(sos, src[i:i + 256], zi=zi)
    out = y * p ** 2.2 * 2.2
    if tone:
        f = mtof(50) * 2 ** (p * 2.0)
        out += (saw(f, n) * 0.18 + saw(f * 1.5, n) * 0.08) * p ** 2.5
    return out


def sub(dur=1.4):
    """A soft low thump: felt more than heard, for a lockup landing or a hard cut in the clean look."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    return sine(40 + 30 * np.exp(-t * 9), n) * np.exp(-t * 3.2) * (1 - np.exp(-t * 300)) * 0.9 + \
        lp(noise(n, "sub"), 120) * np.exp(-t * 8) * 0.5


def air(dur=0.5):
    """A short, dark whoosh: a wipe passing."""
    n = int(dur * SR)
    p = np.arange(n) / n
    return sweep_lp(noise(n, "air", dur), 300, 2600) * np.sin(np.pi * p) ** 3 * 0.7


def swell(dur=1.6):
    """Air breathing in and stopping dead on the moment (a reverse cymbal without the cymbal)."""
    n = int(dur * SR)
    p = np.arange(n) / n
    return sweep_lp(noise(n, "swell", dur), 500, 5000) * p ** 3 * 0.9


def whoosh(dur=0.7):
    n = int(dur * SR)
    p = np.arange(n) / n
    return sweep_lp(noise(n, "whoosh", dur), 400, 6000) * np.sin(np.pi * p) ** 2 * 0.8


def click():
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    return sine(1800, n) * np.exp(-t * 90) * 0.5 + hp(noise(n, "click"), 3000) * np.exp(-t * 400) * 0.3


def pop():
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    return sine(500 + 900 * np.exp(-t * 40), n) * np.exp(-t * 30) * 0.6


def key():
    n = int(0.04 * SR)
    t = np.arange(n) / SR
    return bp(noise(n, "key"), 2000, 7000) * np.exp(-t * 160) * 0.5 + sine(2400, n) * np.exp(-t * 200) * 0.2


def glitch(dur=0.18):
    n = int(dur * SR)
    r = _rng("glitch")
    x = np.zeros(n)
    for _ in range(6):
        i, L = int(r.random() * n * 0.8), int(SR * 0.015)
        x[i:i + L] += saw(r.choice([220, 440, 880, 1760]), min(L, n - i)) * 0.5
    return hp(x + noise(n, "glitch") * 0.15, 300) * np.exp(-np.arange(n) / SR * 10)


def reverb_ir(seconds=2.4, decay=3.2, seed=1):
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = np.stack([r.standard_normal(n), r.standard_normal(n)], 1) * np.exp(-decay * t)[:, None]
    ir = np.stack([lp(ir[:, 0], 7000), lp(ir[:, 1], 7000)], 1)
    ir[:int(0.012 * SR)] = 0
    return ir / np.sqrt((ir ** 2).sum(0))


def verb(x2, ir, mix):
    y = np.stack([signal.fftconvolve(x2[:, c], ir[:, c])[:len(x2)] for c in range(2)], 1)
    return x2 * (1 - mix) + y * mix * 2.0


def put(buf, x, t, gain=1.0, pan=0.0):
    """Mix mono or stereo x into buf at t seconds (pan -1..1 for mono)."""
    x2 = np.stack([x * (1 - max(pan, 0)), x * (1 + min(pan, 0))], 1) if x.ndim == 1 else x
    i = int(round(t * SR))
    if i < 0:
        x2, i = x2[-i:], 0
    m = min(len(x2), len(buf) - i)
    if m > 0:
        buf[i:i + m] += gain * x2[:m]


# --- arrangement ----------------------------------------------------------------------------------

def _beats(beat0, B, t0, t1, step=1.0) -> list[float]:
    """Times of every step-th beat in [t0, t1) on the grid that starts at beat0."""
    k = np.ceil((t0 - beat0) / B / step - 1e-6) * step
    out = []
    while beat0 + k * B < t1 - 1e-6:
        if beat0 + k * B >= 0:
            out.append(beat0 + k * B)
        k += step
    return out


def _chord(st, beat0, B, t):
    """(chord, root) of the bar t falls in: one chord per bar, four-bar loop."""
    i = int(np.floor((t - beat0) / (4 * B))) % 4 if t >= beat0 else 0
    return st["chords"][i], st["roots"][i]


def arrange(style: str, duration: float, sections: list[tuple[float, int]], beat0: float = 0.0,
            bpm: float | None = None) -> np.ndarray:
    """A stereo bed (N x 2 float) for duration seconds. sections = [(start, energy 0-3)], sorted."""
    st = STYLES[style]
    bpm = bpm or st["bpm"]
    B = 60 / bpm
    N = int(duration * SR) + SR // 10
    mus, drums, pads = np.zeros((N, 2)), np.zeros((N, 2)), np.zeros((N, 2))
    kicks = []
    sections = sorted(sections) or [(0.0, 2)]
    bounds = [(s, sections[i + 1][0] if i + 1 < len(sections) else duration, e) for i, (s, e) in enumerate(sections)]

    if st.get("engine") == "future":
        return _future(st, duration, bounds, beat0, B)

    def beats(t0, t1, step=1.0):
        return _beats(beat0, B, t0, t1, step)

    def chord(t):
        return _chord(st, beat0, B, t)

    for t0, t1, e in bounds:
        if e <= 0:  # drone + ticks
            put(mus, pad([st["roots"][0], st["roots"][0] + 7, st["roots"][0] + 12], t1 - t0 + 0.6, cut=500,
                         att=0.3, rel=0.5), t0, 0.5)
            for t in beats(t0, t1, 0.5):
                put(drums, tick(), t, 0.15 * st["bright"], pan=0.3 if int(t * 4) % 2 else -0.3)
            continue
        for t in beats(t0, t1, 4.0 if e == 1 else 1.0):
            if st["kick"]:
                put(drums, kick(), t, 0.8 * st["kick"])
                kicks.append(t)
        if e >= 2 and st["hats"]:
            for t in beats(t0, t1, 0.5):
                off = abs(((t - beat0) / B) % 1 - 0.5) < 1e-3
                put(drums, hat(), t, (0.5 if off else 0.25) * st["hats"], pan=0.25 if off else -0.2)
        if e >= 2 and st["clap"]:
            for t in beats(t0, t1, 2.0):
                put(drums, clap(), t + B, 0.45 * st["clap"])
        step = 0.25 if e >= 3 else 0.5
        cut = (600, 900, 1300, 1600)[e] * st["bright"]
        for t in beats(t0, t1, step):
            _, r = chord(t)
            k = int(round((t - beat0) / B / step))
            put(pads if st.get("pump") else mus, bass(r + (12 if e >= 3 and k % 4 == 2 else 0), step * B * 0.92, cut),
                t, 0.55)
        if (e >= 2 or (e >= 1 and st.get("keys"))) and st["arp"]:
            for t in beats(t0, t1, st.get("arp_step") or (0.25 if e >= 3 else 0.5)):
                c, _ = chord(t)
                k = int(round((t - beat0) / (B / 4)))
                seq = [c[0] + 12, c[2] + 12, c[1] + 12, c[3] + 12, c[2] + 24, c[1] + 12]
                note = keys(seq[k % len(seq)]) if st.get("keys") else pluck(seq[k % len(seq)], bright=3500 * st["bright"])
                put(mus, note, t, 0.16 * st["arp"] * (0.6 if e == 1 else 1.0), pan=0.35 if k % 2 else -0.35)
        for t in [t0] + beats(t0 + 0.01, t1, 4.0):
            c, _ = chord(t)
            put(pads, pad(c, min(4 * B, t1 - t) + 0.6, cut=900 + 400 * e * st["bright"]), t, 0.5)
    # Lead every lift in energy with a riser that peaks on the cut; ring out the last chord.
    for (a, _, e0), (b, _, e1) in zip(bounds, bounds[1:]):
        if e1 > e0 and b - a > 0.8:
            d = min(1.4, b - a)
            put(mus, riser(d, 300, 7000 + 2000 * st["bright"], tone=e1 >= 3), b - d, 0.35)
    if st.get("pump") and kicks:   # pads and bass breathe around the kick
        g = np.ones(N)
        L = int(0.32 * B / 0.5 * SR)
        shape = 1 - st["pump"] * np.exp(-np.arange(L) / (0.09 * SR))
        for t in kicks:
            i = int(t * SR)
            m = min(L, N - i)
            if m > 0:
                g[i:i + m] = np.minimum(g[i:i + m], shape[:m])
        pads *= g[:, None]
    mus += pads
    c, _ = chord(duration)
    put(mus, pad(c + [c[0] + 19], 4.0, cut=2200, att=0.02, rel=2.2), bounds[-1][0], 0.35)
    out = verb(mus, reverb_ir(3.2, 2.2), 0.28) + verb(drums, reverb_ir(1.2, 5.0, 2), 0.12)
    return np.tanh(out * 0.9)[:int(duration * SR)] / 0.9


# --- the future engine ----------------------------------------------------------------------------

def wide_pad(ms, dur, cut=1800, att=0.8, rel=1.6, key=""):
    """A wide supersaw pad: seven detuned saws per note, a different spread on each side (stereo
    width), a soft octave-up shimmer, lowpassed. Returns stereo."""
    n = int(dur * SR)
    r = _rng("wide", tuple(ms), key)
    out = np.zeros((n, 2))
    for ch in range(2):
        x = np.zeros(n)
        for m in ms:
            for det in np.linspace(-0.16, 0.16, 7) + r.normal(0, 0.02, 7):
                x += saw(mtof(m + det), n, r.random())
            x += 0.25 * sine(mtof(m + 12), n, r.random())
        out[:, ch] = lp(x, cut) / (len(ms) * 7)
    return out * env(n, att, 0.3, 0.9, rel)[:, None]


def bell(m, dur=1.6, bright=1.0):
    """A glass bell: FM with a 3.5 ratio, the brightness falling away fast, a long sine tail."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = mtof(m)
    index = (2.4 * bright) * np.exp(-t * 7) + 0.25
    x = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * f * 3.5 * t))
    return (x * np.exp(-t * 2.6) + 0.3 * sine(2 * f, n) * np.exp(-t * 5)) * (1 - np.exp(-t * 600)) * 0.4


def sub_bass(m, dur):
    """A round sine sub with a little second harmonic so it reads on small speakers."""
    n = int(dur * SR)
    f = mtof(m)
    x = sine(f, n) + 0.18 * sine(2 * f, n)
    return np.tanh(1.2 * x) * env(n, 0.04, 0.2, 0.85, 0.12) * 0.8


def pingpong(x2, d, fb=0.42, n=6, damp=3500):
    """Echoes every d seconds bouncing left/right, each a little darker."""
    m = lp(x2.mean(1), damp)
    out = x2.copy()
    k = int(d * SR)
    for i in range(1, n + 1):
        if k * i >= len(m):
            break
        out[k * i:, i % 2] += m[:len(m) - k * i] * fb ** i
    return out


def air_bed(dur):
    """A slow-moving band of air (filtered noise breathing over a few seconds): the futuristic hiss."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    for ch in range(2):
        x = bp(noise(n, "airbed", ch), 3000, 9000)
        out[:, ch] = x * (0.55 + 0.45 * np.sin(2 * np.pi * t / 5.3 + ch * 1.7))
    return out * 0.05


def _future(st, duration, bounds, beat0, B) -> np.ndarray:
    N = int(duration * SR) + SR // 10
    pads, bells, low, drums, fx = (np.zeros((N, 2)) for _ in range(5))
    kicks = []

    def beats(t0, t1, step=1.0):
        return _beats(beat0, B, t0, t1, step)

    def chord(t):
        return _chord(st, beat0, B, t)

    put(fx, air_bed(duration + 0.1), 0.0, 1.0)
    for t0, t1, e in bounds:
        bars = [t0] + beats(t0 + 0.01, t1, 4.0)
        for t in bars:  # pads: one chord a bar, brighter with energy
            c, r = chord(t)
            d = min(4 * B, t1 - t) + 1.2
            put(pads, wide_pad(c if e > 0 else c[:3], d, cut=(900, 1600, 2400, 3400)[e] * st["bright"] / 0.7,
                               att=0.6 if e else 1.2, rel=1.4), t, (0.5, 0.55, 0.55, 0.6)[e])
            if e >= 1:
                put(low, sub_bass(r - 12 if r >= 40 else r, d - 1.0), t, 0.14)
        if e >= 1:  # bells: quarter notes, then 8ths, then 8ths with a sparkle above
            step = 1.0 if e == 1 else 0.5
            for t in beats(t0, t1, step):
                c, _ = chord(t)
                k = int(round((t - beat0) / B / step))
                seq = [c[0] + 12, c[2] + 12, c[4] if len(c) > 4 else c[3] + 12, c[1] + 12, c[3] + 12, c[2] + 24]
                put(bells, bell(seq[k % len(seq)], bright=st["bright"]), t, 0.22 * st["arp"] * (0.7 if e == 1 else 1.0),
                    pan=0.4 if k % 2 else -0.4)
                if e >= 3 and k % 4 == 3:
                    put(bells, bell(c[0] + 36, 0.8, 0.6), t + B * 0.25, 0.06, pan=0.6)
        if e >= 2:  # half-time drums: kick on 1 and the and-of-2, snare-clap on 3
            for t in beats(t0, t1, 4.0):
                for off in (0.0, 2.5):
                    if t + off * B < t1:
                        put(drums, kick(0.5, 120, 40), t + off * B, 0.8 * st["kick"])
                        kicks.append(t + off * B)
                if t + 2 * B < t1:
                    put(drums, clap(), t + 2 * B, 0.5 * st["clap"])
                    put(drums, snare(0.3, 180), t + 2 * B, 0.18 * st["clap"])
            for t in beats(t0, t1, 0.25 if e >= 3 else 0.5):
                k = int(round((t - beat0) / (B / 4)))
                vel = (0.55, 0.2, 0.38, 0.2)[k % 4]
                put(drums, hat(0.05), t, vel * 0.5 * st["hats"], pan=0.3 if k % 2 else -0.15)
        else:  # quiet sections: a soft pulse of ticks
            for t in beats(t0, t1, 1.0):
                put(drums, tick(), t, 0.08, pan=0.3 if int(round((t - beat0) / B)) % 2 else -0.3)
    for (a, _, e0), (b, _, e1) in zip(bounds, bounds[1:]):  # a swell into every lift
        if e1 > e0 and b - a > 0.8:
            d = min(2.0, b - a)
            put(fx, riser(d, 400, 6000, tone=False), b - d, 0.22)
    c, r = chord(duration)
    put(pads, wide_pad(c, 5.0, cut=2600, att=0.05, rel=3.0, key="end"), bounds[-1][0], 0.5)
    if kicks:  # pads and sub breathe around the kick
        g = np.ones(N)
        L = int(0.35 * SR)
        shape = 1 - 0.45 * np.exp(-np.arange(L) / (0.1 * SR))
        for t in kicks:
            i = int(t * SR)
            m = min(L, N - i)
            if m > 0:
                g[i:i + m] = np.minimum(g[i:i + m], shape[:m])
        pads *= g[:, None]
        low *= g[:, None]
    bells = pingpong(bells, 0.75 * B, fb=0.45, n=6)
    hall = reverb_ir(4.5, 1.5, 7)
    out = verb(pads, hall, 0.35) + verb(bells, hall, 0.4) + low + verb(drums, reverb_ir(1.6, 4.0, 2), 0.18) + \
        verb(fx, hall, 0.5)
    out = np.stack([hp(out[:, 0], 28), hp(out[:, 1], 28)], 1)
    return np.tanh(out * 0.9)[:int(duration * SR)] / 0.9


def bed(style: str, duration: float, sections, beat0=0.0, bpm=None) -> Path:
    """arrange() through the cache: returns a wav path."""
    key = json.dumps([style, round(duration, 3), [[round(a, 3), e] for a, e in sections], round(beat0, 3), bpm, 1])
    out = CACHE / f"{style}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.wav"
    if not out.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        write(out.with_suffix(".part.wav"), arrange(style, duration, sections, beat0, bpm))
        out.with_suffix(".part.wav").replace(out)
    return out


def write(path: Path, x: np.ndarray) -> None:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def read(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767
        return x.reshape(-1, w.getnchannels())


def library(out_dir: Path = CACHE / "library", seconds: float = 30.0) -> list[Path]:
    """A preview of every style: 4 s intro, a lift, the full groove, a breakdown, the drop."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, st in STYLES.items():
        B = 60 / st["bpm"]
        b0 = round(4 / B) * B
        secs = [(0, 0), (b0, 1), (b0 + 8 * B, 2), (b0 + 24 * B, 1), (b0 + 32 * B, 3)]
        p = out_dir / f"{name}_{st['bpm']}bpm.wav"
        write(p, arrange(name, seconds, [s for s in secs if s[0] < seconds], beat0=b0))
        paths.append(p)
    return paths
