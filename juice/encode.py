"""The one video encode juice uses, for cuts and motion videos alike, and how sources are decoded.

Masters are 10-bit HEVC Main 10 (hvc1 so Apple players open them, tagged bt709), so gradients, grain and
dark footage don't band the way 8-bit H.264 does, and platforms transcode from a clean source. Previews
(480p, "look at it in seconds") are 8-bit H.264; they're for checking a cut, not for posting.

Where the Mac has a media engine (Apple silicon), it does the work:
- Encode: VideoToolbox at constant quality. `quality` 65 matched x265 CRF 16 on VMAF (phone footage
  99.27 vs 99.28 at the same bitrate; motion graphics 97.95 vs 97.83 at 26% less) while the encode itself
  took ~5 CPU-seconds instead of 140-190 for 10 s of 1080x1920. Constant quality, not a bitrate: at x265's
  bitrate a VBR hardware encode scored lower on graphics (97.47), because it can't move bits to where
  they're needed.
- Decode: `-hwaccel videotoolbox` for HEVC and ProRes (and with "hardware", 8-bit 4:2:0 H.264, which is
  otherwise faster on the CPU's cores than queued on one media engine), frames downloaded to system memory
  (every filter, the HDR LUT included, runs on them). Each file is checked once (cached): in its first half second
  the media engine must return as many frames as software does, with no hardware error, because a hardware
  decode that fails can fail quietly (Sony 4:2:2 10-bit H.264 drops frames with no error).
Everywhere else, or with JUICE_ENCODER=software / style "encoder": "software", it's libx265 / libx264 and
CPU decoding. A hardware encode that fails anyway is redone in software by the caller.

    enc = video_args(style)      # "-c:v ... -pix_fmt ..." for an encode
    cp = copy_args(style)        # "-c:v copy" (+ the hvc1 tag) when remuxing what video_args produced
    pix = pix_fmt(style)         # the pixel format to end a filter graph on
    dec = decode_args(path)      # [] or ["-hwaccel", "videotoolbox"], before "-i path"

style keys: bit_depth (10 | 8), encoder (auto | hardware | software), quality (hardware, 1-100),
crf + preset (software). Env: JUICE_ENCODER and JUICE_DECODER (auto | hardware | software) override.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

COLOR = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv"]
HW_QUALITY = {10: 65, 8: 50}       # VideoToolbox -q:v for masters and previews
CACHE = Path.home() / ".cache" / "juice" / "hwdecode.json"
# What the media engine decodes reliably. Not 4:2:2 or 10-bit H.264 (Sony XAVC: frames dropped silently).
HW_DECODE = {"hevc": {"yuv420p", "yuvj420p", "yuv420p10le"}, "prores": None, "h264": {"yuv420p", "yuvj420p"}}
# What it decodes by default ("auto"): the codecs that are expensive in software. H.264 decodes faster on the
# CPU's cores than on the one media engine every render worker queues on (the OpenBeam trailer: 12.0 s with
# software decode, 16.7 s with hardware, though 120 vs 50 CPU-seconds); "hardware" sends it there too.
AUTO_DECODE = {"hevc", "prores"}
_lock = threading.Lock()


# --- encode ----------------------------------------------------------------------------------

def bits(style: dict) -> int:
    return int(style.get("bit_depth", 10))


def pix_fmt(style: dict) -> str:
    """The format to end a filter graph on: what the chosen encoder takes in, so nothing converts twice."""
    if hardware(style):
        return "p010le" if bits(style) == 10 else "nv12"
    return "yuv420p10le" if bits(style) == 10 else "yuv420p"


def _mode(style: dict | None, env: str, key: str) -> str:
    m = os.environ.get(env) or (style or {}).get(key) or "auto"
    if m not in ("auto", "hardware", "software"):
        raise ValueError(f"{env}/{key} must be auto, hardware or software, not {m!r}")
    return m


@lru_cache(maxsize=None)
def hardware_encoder(depth: int) -> str | None:
    """The VideoToolbox encoder for this bit depth if it really encodes here (one tiny test encode), else None."""
    if sys.platform != "darwin":
        return None
    name = "hevc_videotoolbox" if depth == 10 else "h264_videotoolbox"
    test = _vt_args(depth, HW_QUALITY[depth])
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=0.3",
                        "-vf", "format=" + ("p010le" if depth == 10 else "nv12"), *test, "-f", "null", "-"],
                       capture_output=True, text=True)
    return name if r.returncode == 0 else None


def hardware(style: dict) -> bool:
    """Whether this style's video gets encoded on the media engine."""
    m = _mode(style, "JUICE_ENCODER", "encoder")
    if m == "software":
        return False
    ok = hardware_encoder(bits(style)) is not None
    if m == "hardware" and not ok:
        print("juice: no hardware encoder here, encoding in software", file=sys.stderr)
    return ok


def software(style: dict) -> dict:
    """The same style, decoded and encoded in software (the fallback when the media engine fails)."""
    return {**style, "encoder": "software", "decoder": "software"}


def _vt_args(depth: int, quality: int) -> list[str]:
    if depth == 10:
        return ["-c:v", "hevc_videotoolbox", "-profile:v", "main10", "-pix_fmt", "p010le", "-q:v", str(quality),
                "-allow_sw", "0", "-tag:v", "hvc1", *COLOR]
    return ["-c:v", "h264_videotoolbox", "-profile:v", "high", "-pix_fmt", "nv12", "-q:v", str(quality),
            "-allow_sw", "0", *COLOR]


def video_args(style: dict) -> list[str]:
    if hardware(style):
        return _vt_args(bits(style), int(style.get("quality", HW_QUALITY[bits(style)])))
    crf, preset = str(style.get("crf", 16)), style.get("preset", "medium")
    if bits(style) == 10:
        return ["-c:v", "libx265", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p10le",
                "-tag:v", "hvc1", *COLOR, "-x265-params", "log-level=error"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p", *COLOR]


def describe(style: dict) -> str:
    """The encoder video_args picks, for render reports."""
    return video_args(style)[1]


def copy_args(style: dict) -> list[str]:
    return ["-c:v", "copy"] + (["-tag:v", "hvc1"] if bits(style) == 10 else [])


# --- decode ----------------------------------------------------------------------------------

@lru_cache(maxsize=256)
def _stream(path: str) -> tuple[str, str]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,pix_fmt", "-of", "json", path], capture_output=True, text=True)
    try:
        s = json.loads(r.stdout)["streams"][0]
        return s.get("codec_name", ""), s.get("pix_fmt", "")
    except (ValueError, KeyError, IndexError):
        return "", ""


def _count_cmd(path: str, hw: bool, seconds: float) -> list[str]:
    """Frames decoded from the first `seconds`, by the media engine or (hw=False) by software."""
    return ["ffmpeg", "-v", "error", "-nostats", "-progress", "pipe:1",
            *(["-hwaccel", "videotoolbox"] if hw else []), "-t", str(seconds), "-i", path, "-map", "0:v:0",
            "-f", "null", "-"]


def _frames_seen(progress: str) -> int:
    n = [ln.split("=", 1)[1] for ln in progress.splitlines() if ln.startswith("frame=")]
    return int(n[-1]) if n and n[-1].strip().isdigit() else 0


def _key(path: str) -> str:
    st = os.stat(path)
    return hashlib.sha1(f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _allowed(path: str) -> bool:
    codec, pix = _stream(path)
    allowed = HW_DECODE.get(codec, set())
    return allowed is None or pix in allowed


def check_decode(paths, seconds: float = 0.5, batch: int = 8) -> dict[str, bool]:
    """Whether the media engine decodes each file fully: in its first half second it returns as many frames as a
    software decode does, and ffmpeg reports no hardware failure. Each file is checked once (cached by path, size and mtime).
    The checks run as parallel ffmpeg processes, not Python threads: a motion render forks right after,
    and macOS aborts forked children of a process that ever ran a second thread (see motion/video.py)."""
    paths = [str(p) for p in dict.fromkeys(str(p) for p in paths) if os.path.exists(str(p))]
    cache, out, todo = _load(), {}, []
    for p in paths:
        if not _allowed(p):
            out[p] = False
        elif _key(p) in cache:
            out[p] = cache[_key(p)]["ok"]
        else:
            todo.append(p)
    fresh = {}
    for i in range(0, len(todo), batch):
        procs = [(p, subprocess.Popen(_count_cmd(p, hw, seconds), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True), hw) for p in todo[i:i + batch] for hw in (True, False)]
        res = {}
        for p, proc, hw in procs:
            so, se = proc.communicate()
            res[(p, hw)] = (_frames_seen(so), se, proc.returncode)
        for p in todo[i:i + batch]:
            (hn, herr, hrc), (sn, _, _) = res[(p, True)], res[(p, False)]
            ok = hrc == 0 and sn > 0 and hn == sn and "fail" not in herr.lower()
            codec, pix = _stream(p)
            fresh[_key(p)] = {"ok": ok, "path": p, "codec": codec, "pix_fmt": pix, "frames": [hn, sn],
                              "error": herr.strip()[-300:] or None}
            out[p] = ok
    if fresh:
        with _lock:
            cache = {**_load(), **fresh}
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE.with_name(f"{CACHE.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(cache, indent=1))
            os.replace(tmp, CACHE)
    return out


def hardware_decode_ok(path: str | Path) -> bool:
    """Whether the media engine decodes this file the same as software does (checked once per file, cached)."""
    return check_decode([path]).get(str(path), False)


def _wanted(path: str, mode: str) -> bool:
    """Whether this mode would decode this file on the media engine (before the per-file check)."""
    if mode == "software" or sys.platform != "darwin" or not os.path.exists(path):
        return False
    return mode == "hardware" or _stream(path)[0] in AUTO_DECODE


def prepare(paths, style: dict | None = None) -> None:
    """Check every source's hardware decode up front, in parallel (so each render step doesn't pay for it)."""
    mode = _mode(style, "JUICE_DECODER", "decoder")
    check_decode([p for p in map(str, paths) if _wanted(p, mode)])


@lru_cache(maxsize=256)
def _decode(path: str, mode: str) -> tuple[str, ...]:
    return ("-hwaccel", "videotoolbox") if _wanted(path, mode) and hardware_decode_ok(path) else ()


def decode_args(path: str | Path, style: dict | None = None) -> list[str]:
    """Input options for decoding this source: hardware where it's known to decode correctly, else none.
    Frames come back in system memory, so filters (crop, scale, the HDR LUT, overlays) work unchanged."""
    return list(_decode(str(path), _mode(style, "JUICE_DECODER", "decoder")))
