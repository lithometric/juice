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
    tail = finish_filters(style) # what every encode's filter graph ends with: that format + BT.709 tags
    dec = decode_args(path)      # [] or ["-hwaccel", "videotoolbox"], before "-i path"
    aud = audio_args("192k")     # AAC: Apple's encoder (aac_at) where there is one, else ffmpeg's

Audio is AAC from Apple's AudioToolbox encoder when ffmpeg has it (macOS): against a lossless source it
reproduced an isolated voice take 4.9-8.3 dB more faithfully than ffmpeg's built-in encoder at the same
bitrate (192/256 kb/s), music as well or better (and its 14-20 kHz band 6 dB better at 256 kb/s), in half
the time. Elsewhere, ffmpeg's encoder.

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


TAGS = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
# Phone footage brings HDR context along as frame side data (iPhone: Dolby Vision RPUs, ambient viewing
# environment; others: mastering display, content light level). An SDR master must not carry any of it: Apple
# devices and platforms read it as HDR signalling. juice's frames need none of it after the tone map.
STRIP = "sidedata=mode=delete"


def finish_filters(style: dict, rgb_in: bool = False) -> list[str]:
    """The end of every encode's filter graph: the encoder's pixel format, no leftover HDR side data, then
    BT.709 tags on the frames.
    ffmpeg 7+ writes the frames' own colour tags and ignores -color_primaries / -color_trc when they say
    "unknown", so without this a master said only "bt709 matrix" and every player guessed the rest
    (QuickTime and Chrome guess differently). rgb_in: the frames are RGB (motion videos), converted here
    with the BT.709 matrix into limited range."""
    conv = ["scale=out_color_matrix=bt709:out_range=tv"] if rgb_in else []
    return [*conv, f"format={pix_fmt(style)}", STRIP, TAGS]


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


@lru_cache(maxsize=None)
def _has_encoder(name: str) -> bool:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return any(ln.split()[1:2] == [name] for ln in r.stdout.splitlines() if ln.strip())


def audio_args(bitrate: str) -> list[str]:
    """AAC at this bitrate: Apple's AudioToolbox encoder where ffmpeg has it, else ffmpeg's own."""
    return ["-c:a", "aac_at" if _has_encoder("aac_at") else "aac", "-b:a", bitrate]


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


@lru_cache(maxsize=256)
def _video(path: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:stream_side_data=rotation", "-of", "json", path], capture_output=True, text=True)
    try:
        return json.loads(r.stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        return {}


def _dims(path: str) -> tuple[int, int]:
    v = _video(path)
    return int(v["width"]), int(v["height"])


def _rotation(path: str) -> int:
    """The display rotation ffprobe reports (-90 for a portrait iPhone clip), 0 if none."""
    for sd in _video(path).get("side_data_list", []):
        if "rotation" in sd:
            return int(float(sd["rotation"]))
    return 0


def engine_scale(path: str | Path, w: int, h: int, style: dict | None = None) -> tuple[list[str], list[str]] | None:
    """Decode and scale this source to cover w x h on the media engine, for previews and stills:
    (input options, filters ending in software frames), or None where the CPU path applies.

    Scaling on the engine instead of the CPU takes the CPU cost of decoding and scaling 4K60 iPhone footage
    from 4.7 to 0.15 CPU-seconds per 4 s of source. Its scaler is softer than lanczos (VMAF 90.5 against
    it at 1080x1920), so masters keep the CPU scale. The engine doesn't apply the clip's rotation, so it's
    cleared on input (-display_rotation 0, or the output would carry it and players would turn it again)
    and done with transpose_vt. Filters that only drop or retime frames (fps, setpts) can come before these."""
    path = str(path)
    if not decode_args(path, style):
        return None
    rot = -_rotation(path) % 360            # what ffmpeg's autorotate would turn: 90 = clockwise
    turn = {90: "clock", 270: "cclock", 180: "reversal"}.get(rot)
    sw, sh = _dims(path)
    if rot in (90, 270):
        sw, sh = sh, sw
    # the size scale=w:h:force_original_aspect_ratio=increase picks, kept even for the 4:2:0 surfaces
    tw, th = max(w, round(h * sw / sh)), max(h, round(w * sh / sw))
    tw, th = tw + tw % 2, th + th % 2
    sw_fmt = "p010le" if "10" in _stream(path)[1] else "nv12"
    inputs = ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld", "-display_rotation", "0"]
    filters = ([f"transpose_vt=dir={turn}"] if turn else []) + [f"scale_vt=w={tw}:h={th}", "hwdownload", f"format={sw_fmt}"]
    return inputs, filters


def decode_args(path: str | Path, style: dict | None = None) -> list[str]:
    """Input options for decoding this source: hardware where it's known to decode correctly, else none.
    Frames come back in system memory, so filters (crop, scale, the HDR LUT, overlays) work unchanged."""
    return list(_decode(str(path), _mode(style, "JUICE_DECODER", "decoder")))


# --- what a finished file says about its colour ---------------------------------------------------------

CODES = {"primaries": {1: "bt709", 9: "bt2020", 2: "unspecified"}, "transfer": {1: "bt709", 2: "unspecified", 13: "srgb",
         14: "bt2020-10", 16: "pq", 18: "hlg"}, "matrix": {1: "bt709", 2: "unspecified", 6: "bt601", 9: "bt2020nc"}}
HDR_BOXES = {b"dvcC": "Dolby Vision", b"dvvC": "Dolby Vision", b"dvwC": "Dolby Vision", b"mdcv": "HDR mastering display",
             b"clli": "HDR content light level", b"SmDm": "HDR mastering display", b"CoLL": "HDR content light level",
             b"amve": "ambient viewing environment"}


def _boxes(path: str) -> list[tuple[bytes, bytes]]:
    """(box type, payload) for the colour-related boxes in an MP4/MOV's video sample entries."""
    import struct
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stsd"}
    samples = {b"hvc1", b"hev1", b"avc1", b"avc3", b"dvh1", b"dvhe", b"apch", b"apcn", b"ap4h", b"ap4x"}
    want = {b"colr", b"gama", *HDR_BOXES}
    data = Path(path).read_bytes()
    out = []

    def walk(off: int, end: int):
        while off + 8 <= end:
            size, typ = struct.unpack(">I4s", data[off:off + 8])
            hdr = 8
            if size == 1:
                size, hdr = struct.unpack(">Q", data[off + 8:off + 16])[0], 16
            elif size == 0:
                size = end - off
            if size < hdr:
                return
            body = off + hdr
            if typ in want:
                out.append((typ, data[body:off + size]))
            if typ in containers:
                walk(body + (8 if typ == b"stsd" else 0), off + size)
            elif typ in samples:
                walk(body + 78, off + size)          # the visual sample entry's fixed fields, then its boxes
            off += size
    walk(0, len(data))
    return out


def color_report(path: str | Path) -> dict:
    """What a finished file declares about its colour, in words, and anything wrong for a social upload.
    juice masters are Rec.709 SDR: BT.709 primaries, transfer and matrix (the 1-1-1 every platform converts
    to), limited range, in an nclx colr box, with no gamma box and no HDR or Dolby Vision metadata. Resolve's
    "Rec.709-A" and "gamma 2.4" exports are display tweaks for Apple players, not other codes; juice uses neither."""
    import struct
    path = str(path)
    problems, extra = [], []
    colr = None
    for typ, body in _boxes(path):
        if typ == b"colr" and body[:4] in (b"nclx", b"nclc") and len(body) >= 10:
            p, t, m = struct.unpack(">HHH", body[4:10])
            full = bool(body[10] >> 7) if body[:4] == b"nclx" and len(body) > 10 else False
            colr = (p, t, m, full)
        elif typ == b"gama":
            extra.append("gama box (an Apple gamma override)")
        elif typ in HDR_BOXES:
            extra.append(HDR_BOXES[typ])
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,pix_fmt:stream_side_data=side_data_type", "-of", "json", path],
                       capture_output=True, text=True)
    st = (json.loads(r.stdout or "{}").get("streams") or [{}])[0]
    names = {"dovi": "Dolby Vision", "mastering": "HDR mastering display", "content light": "HDR content light level",
             "ambient": "ambient viewing environment"}
    f = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+#1", "-show_entries",
                        "frame_side_data=side_data_type", "-of", "json", path], capture_output=True, text=True)
    first = ((json.loads(f.stdout or "{}").get("frames") or [{}])[0]).get("side_data_list", [])   # in-band (SEI)
    for sd in st.get("side_data_list", []) + first:
        kind = sd.get("side_data_type", "").lower()
        extra += [v for k, v in names.items() if k in kind]
    bits = "10-bit" if "10" in st.get("pix_fmt", "") else "8-bit"
    codec = {"hevc": "HEVC", "h264": "H.264", "prores": "ProRes"}.get(st.get("codec_name", ""), st.get("codec_name", "?"))
    if colr is None:
        problems.append("no colr box: the file doesn't say its colour space, so each player guesses (Chrome shifted brand colours)")
        label = "untagged"
    else:
        p, t, m, full = colr
        names = [CODES[k].get(v, str(v)) for k, v in zip(("primaries", "transfer", "matrix"), (p, t, m))]
        code = f"{p}-{t}-{m}"
        if (p, t, m) == (1, 1, 1):
            label = f"Rec.709 SDR ({code})"
        elif t in (16, 18):
            label = f"HDR {'PQ' if t == 16 else 'HLG'} ({code})"
        else:
            label = f"{'/'.join(names)} ({code})"
        if (p, t, m) != (1, 1, 1):
            problems.append(f"colour tags are {code} ({', '.join(names)}), not Rec.709 1-1-1")
        if full:
            problems.append("tagged full range; social platforms expect limited (video) range")
    extra = sorted(set(extra))
    if extra:
        problems.append("carries " + ", ".join(extra) + ": an SDR upload should have no HDR or Dolby Vision metadata")
    rng = "full range" if colr and colr[3] else "limited range"
    summary = f"{label}, {rng}, {bits} {codec}" + (f"; also {', '.join(extra)}" if extra else "")
    return {"summary": summary, "colr": f"{colr[0]}-{colr[1]}-{colr[2]}" if colr else None,
            "range": rng.split()[0], "extra": extra, "problems": problems}
