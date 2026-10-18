"""Synthesized sound effects, so every hit has sound without a stock library or a license.

Sounds: whoosh (rising swept noise), impact (pitch-dropping thump + crack), boom (deep hit with a
long tail), pop (short bright tick for text landing). Each sound's transient is at t=0 except the
whoosh, whose peak sits at `ATTACK["whoosh"]` so it can be started early and peak on the frame.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48000
ATTACK = {"whoosh": 0.32, "impact": 0.0, "boom": 0.0, "pop": 0.0}


def _band(noise: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(noise)
    f = np.fft.rfftfreq(len(noise), 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(spec, len(noise))


def whoosh(rng) -> np.ndarray:
    d = 0.5
    t = np.arange(int(SR * d)) / SR
    out = np.zeros_like(t)
    # Sweep the band upward in slices for a "rush past" feel.
    for k in range(10):
        a, b = int(len(t) * k / 10), int(len(t) * (k + 1) / 10)
        lo = 250 * (1 + k * 0.8)
        out[a:b] = _band(rng.standard_normal(len(t)), lo, lo * 3)[a:b]
    env = np.exp(-((t - ATTACK["whoosh"]) ** 2) / (2 * 0.09 ** 2))
    return out * env


def impact(rng) -> np.ndarray:
    d = 0.9
    t = np.arange(int(SR * d)) / SR
    f = 95 * np.exp(-t * 6) + 42
    thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 5.5)
    crack = _band(rng.standard_normal(len(t)), 1200, 7000) * np.exp(-t * 38) * 0.5
    return thump + crack


def boom(rng) -> np.ndarray:
    d = 1.8
    t = np.arange(int(SR * d)) / SR
    f = 70 * np.exp(-t * 3) + 32
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    rumble = _band(rng.standard_normal(len(t)), 30, 220) * np.exp(-t * 1.8) * 0.6
    crack = _band(rng.standard_normal(len(t)), 900, 6000) * np.exp(-t * 30) * 0.45
    return body + rumble + crack


def pop(rng) -> np.ndarray:
    d = 0.18
    t = np.arange(int(SR * d)) / SR
    return (np.sin(2 * np.pi * 1600 * t * np.exp(-t * 8)) * np.exp(-t * 30) * 0.6
            + _band(rng.standard_normal(len(t)), 3000, 9000) * np.exp(-t * 60) * 0.3)


SOUNDS = {"whoosh": whoosh, "impact": impact, "boom": boom, "pop": pop}


def render(name: str, out: Path, seed: int = 7) -> Path:
    if out.exists():
        return out
    x = SOUNDS[name](np.random.default_rng(seed))
    x = x / (np.abs(x).max() + 1e-9) * 0.89
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return out
