"""HDR footage in an SDR edit: tone-map HLG and PQ sources to bt709 so they don't come out flat and grey.

Phones and gimbal cameras (iPhone, DJI Osmo, GoPro) record HLG or PQ in bt2020 by default. Read as if it
were bt709, that footage is washed out: no contrast, desaturated skin, pink walls gone white. juice
masters are SDR bt709, so every decode of an HDR source goes through one 3D LUT first:

  signal -> display light (HLG: inverse OETF + OOTF for a 1000-nit display; PQ: the EOTF)
         -> bt2020 to bt709 primaries -> HDR reference white (203 nits) near SDR white, highlights
            rolled off on max(R,G,B) so hues hold -> BT.1886 (2.4) signal

The LUT is built once with numpy and cached in ~/.cache/juice/luts. There's no zscale/libplacebo in
every ffmpeg build, but lut3d is everywhere.

    vf += hdr.filters(path)     # [] for SDR sources
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

LUTS = Path.home() / ".cache" / "juice" / "luts"
HDR = {"arib-std-b67": "hlg", "smpte2084": "pq"}
REF_WHITE = 203.0           # nits: HLG 75% and the PQ reference white (BT.2408)
KNEE = 0.6                  # SDR linear level where the highlight roll-off starts
SIZE = 65
VERSION = 1

BT2020_TO_709 = np.array([[1.6605, -0.5876, -0.0728],
                          [-0.1246, 1.1329, -0.0083],
                          [-0.0182, -0.1006, 1.1187]])


@lru_cache(maxsize=64)
def transfer(path: str) -> str | None:
    """'hlg', 'pq' or None (SDR, or no video stream)."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=color_transfer", "-of", "json", path], capture_output=True, text=True)
    try:
        return HDR.get(json.loads(r.stdout)["streams"][0].get("color_transfer"))
    except (ValueError, KeyError, IndexError):
        return None


def _display_nits(rgb: np.ndarray, kind: str) -> np.ndarray:
    if kind == "pq":
        m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
        p = np.power(rgb, 1 / m2)
        return 10000 * np.power(np.maximum(p - c1, 0) / (c2 - c3 * p), 1 / m1)
    a, b, c = 0.17883277, 0.28466892, 0.55991073
    e = np.where(rgb <= 0.5, rgb ** 2 / 3, (np.exp((rgb - c) / a) + b) / 12)
    ys = e @ np.array([0.2627, 0.6780, 0.0593])
    return 1000 * np.power(np.maximum(ys, 1e-6), 0.2)[..., None] * e     # OOTF, gamma 1.2 at 1000 nits


def _sdr(nits: np.ndarray) -> np.ndarray:
    lin = np.maximum((nits / REF_WHITE) @ BT2020_TO_709.T, 0)
    m = lin.max(axis=-1, keepdims=True)
    top = KNEE + (1 - KNEE) * (1 - np.exp(-(m - KNEE) / (1 - KNEE)))
    mapped = np.where(m > KNEE, top, m)
    lin = np.where(m > 0, lin * mapped / np.maximum(m, 1e-9), 0)
    return np.clip(lin, 0, 1) ** (1 / 2.4)


def lut(kind: str) -> Path:
    path = LUTS / f"{kind}_to_sdr709_v{VERSION}.cube"
    if path.exists():
        return path
    g = np.linspace(0, 1, SIZE)
    # .cube order: red changes fastest, then green, then blue.
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    out = _sdr(_display_nits(np.stack([r, gg, b], -1).reshape(-1, 3), kind))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(f"TITLE \"juice {kind} to SDR bt709\"\nLUT_3D_SIZE {SIZE}\n")
        np.savetxt(f, out, fmt="%.6f")
    tmp.replace(path)
    return path


def filters(path: str | Path) -> list[str]:
    """The filters that bring this source to SDR bt709 ([] when it already is). Put them after any
    scale/crop, so the LUT runs on output-sized frames."""
    kind = transfer(str(path))
    if not kind:
        return []
    return ["scale=in_color_matrix=bt2020:in_range=tv,format=gbrp16le",
            f"lut3d=file='{lut(kind)}':interp=tetrahedral",
            "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p10le",
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"]
