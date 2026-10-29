"""Rebuild H.264 MOV/MP4 files whose moov box never arrived (a download cut off mid-file).

The frames are all still there: mdat holds AVCC samples (4-byte length + NAL). What died with the
moov is the SPS/PPS, so we borrow them from a donor recorded by the same encoder, then find the
broken file's picture size by decoding its first IDR at candidate sizes until one decodes clean.
Audio can't come back this way: its sample tables lived in the moov too.
"""
from __future__ import annotations

import concurrent.futures as cf
import mmap
import os
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

START = b"\x00\x00\x00\x01"
# NAL types a plain H.264 elementary stream carries; anything else means we've walked off the video.
NAL_OK = {1, 5, 6, 7, 8, 9, 10, 11, 12}
MAX_NAL = 50_000_000
# Tried before any sweep: (w, h) of common screens and cameras, crop included.
COMMON = [(1920, 1080), (1280, 720), (3840, 2160), (2560, 1440), (1080, 1920), (720, 1280),
          (1440, 900), (1280, 800), (1680, 1050), (1920, 1200), (2560, 1600), (2880, 1800),
          (3024, 1964), (3456, 2234), (1512, 982), (1728, 1117), (2940, 1912), (3360, 2100),
          (5120, 2880), (4096, 2160), (1366, 768), (1600, 900), (3840, 2400)]
HIGH_PROFILES = {100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135}


# --- bitstream helpers -------------------------------------------------------------------------

def unescape(b: bytes) -> bytes:
    """NAL payload -> RBSP (drop emulation-prevention 0x03 after 00 00)."""
    out, z = bytearray(), 0
    for x in b:
        if z >= 2 and x == 3:
            z = 0
            continue
        out.append(x)
        z = z + 1 if x == 0 else 0
    return bytes(out)


def escape(b: bytes) -> bytes:
    out, z = bytearray(), 0
    for x in b:
        if z >= 2 and x <= 3:
            out.append(3)
            z = 0
        out.append(x)
        z = z + 1 if x == 0 else 0
    return bytes(out)


class Bits:
    def __init__(self, b: bytes):
        self.b, self.p = b, 0

    def u(self, n: int) -> int:
        v = 0
        for _ in range(n):
            if self.p >> 3 >= len(self.b):
                raise ValueError("SPS ends early")
            v = (v << 1) | ((self.b[self.p >> 3] >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v

    def ue(self) -> int:
        z = 0
        while self.u(1) == 0:
            z += 1
            if z > 31:
                raise ValueError("bad exp-Golomb code")
        return (1 << z) - 1 + self.u(z)

    def se(self) -> int:
        k = self.ue()
        return (k + 1) // 2 if k & 1 else -(k // 2)

    def bits(self, start: int, end: int) -> list[int]:
        return [(self.b[i >> 3] >> (7 - (i & 7))) & 1 for i in range(start, end)]


class Writer:
    def __init__(self, bits: list[int] | None = None):
        self.bits = list(bits or [])

    def u(self, n: int, v: int):
        self.bits += [(v >> (n - 1 - i)) & 1 for i in range(n)]

    def ue(self, v: int):
        v += 1
        self.u(v.bit_length() - 1, 0)
        self.u(v.bit_length(), v)

    def rbsp(self) -> bytes:
        b = self.bits + [1]  # stop bit, then zero-pad to a byte
        b += [0] * (-len(b) % 8)
        return bytes(int("".join(map(str, b[i:i + 8])), 2) for i in range(0, len(b), 8))


# --- SPS ---------------------------------------------------------------------------------------

def parse_sps(nal: bytes) -> dict:
    """The fields that decide picture size, plus the untouched bits before them (kept verbatim)."""
    r = Bits(unescape(nal[1:]))
    f = {"header": nal[0], "profile": r.u(8), "constraints": r.u(8), "level": r.u(8), "id": r.ue(),
         "chroma": 1}
    if f["profile"] in HIGH_PROFILES:
        f["chroma"] = r.ue()
        if f["chroma"] == 3:
            r.u(1)
        r.ue(), r.ue(), r.u(1)  # bit depths, qpprime bypass
        if r.u(1):  # scaling matrices: walk them, the bits are copied as-is
            for i in range(8 if f["chroma"] != 3 else 12):
                if r.u(1):
                    last = nxt = 8
                    for _ in range(16 if i < 6 else 64):
                        if nxt:
                            nxt = (last + r.se() + 256) % 256
                        last = nxt or last
    r.ue()  # log2_max_frame_num
    poc = r.ue()
    if poc == 0:
        r.ue()
    elif poc == 1:
        r.u(1), r.se(), r.se()
        for _ in range(r.ue()):
            r.se()
    r.ue(), r.u(1)  # max refs, gaps allowed
    f["head_bits"] = r.bits(0, r.p)
    f["wm"], f["hm_units"] = r.ue() + 1, r.ue() + 1
    f["frame_mbs_only"] = r.u(1)
    if not f["frame_mbs_only"]:
        r.u(1)
    f["direct8x8"] = r.u(1)
    f["crop"] = (r.ue(), r.ue(), r.ue(), r.ue()) if r.u(1) else (0, 0, 0, 0)
    f["vui"] = r.u(1)
    return f


def _crop_units(f: dict) -> tuple[int, int]:
    sub_w, sub_h = {0: (1, 1), 1: (2, 2), 2: (2, 1), 3: (1, 1)}[f["chroma"]]
    return sub_w, sub_h * (2 - f["frame_mbs_only"])


def sps_size(f: dict) -> tuple[int, int]:
    cx, cy = _crop_units(f)
    l, r, t, b = f["crop"]
    hm = f["hm_units"] * (2 - f["frame_mbs_only"])
    return f["wm"] * 16 - cx * (l + r), hm * 16 - cy * (t + b)


def build_sps(f: dict, w: int, h: int) -> bytes:
    """Donor SPS rewritten for a w x h picture: cropping for non-multiple-of-16 sizes, no VUI."""
    cx, cy = _crop_units(f)
    mbh = 16 * (2 - f["frame_mbs_only"])
    wm, hm = -(-w // 16), -(-h // mbh) * (2 - f["frame_mbs_only"])
    wr = Writer(f["head_bits"])
    wr.ue(wm - 1)
    wr.ue(hm // (2 - f["frame_mbs_only"]) - 1)
    wr.u(1, f["frame_mbs_only"])
    if not f["frame_mbs_only"]:
        wr.u(1, 0)
    wr.u(1, f["direct8x8"])
    cr, cb = (wm * 16 - w) // cx, (hm * 16 - h) // cy
    if cr or cb:
        wr.u(1, 1)
        for v in (0, cr, 0, cb):
            wr.ue(v)
    else:
        wr.u(1, 0)
    wr.u(1, 0)  # VUI dropped: its timing/aspect fields belong to the donor
    return bytes([f["header"]]) + escape(wr.rbsp())


# --- reading the streams -----------------------------------------------------------------------

def nal_units(annexb: bytes) -> list[bytes]:
    parts = annexb.replace(b"\x00\x00\x01", START).split(START)
    return [p.rstrip(b"\x00") if i < len(parts) - 1 else p for i, p in enumerate(parts) if p.strip(b"\x00")]


def parameter_sets(donor: Path) -> tuple[bytes, bytes]:
    """SPS and PPS of an intact H.264 file, via ffmpeg's AVCC -> Annex-B filter (one frame)."""
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(donor), "-map", "0:v:0", "-c", "copy",
                        "-bsf:v", "h264_mp4toannexb", "-frames:v", "1", "-f", "h264", "-"],
                       capture_output=True, timeout=60)
    nals = nal_units(p.stdout)
    sps = [n for n in nals if n and n[0] & 0x1F == 7]
    pps = [n for n in nals if n and n[0] & 0x1F == 8]
    if not sps or not pps:
        raise ValueError(f"no SPS/PPS in {donor.name}")
    return sps[0], pps[0]


def mdat_range(m) -> tuple[int, int]:
    """Payload start and end of the mdat box (end clipped to the file: it's usually cut short)."""
    pos, size = 0, len(m)
    while pos + 8 <= size:
        n, kind = struct.unpack(">I4s", m[pos:pos + 8])
        hdr = 8
        if n == 1 and pos + 16 <= size:
            n, hdr = struct.unpack(">Q", m[pos + 8:pos + 16])[0], 16
        elif n == 0:
            n = size - pos
        if kind == b"mdat":
            return pos + hdr, min(pos + n, size)
        if n < hdr:
            break
        pos += n
    raise ValueError("no mdat box")


def walk_avcc(m, start: int, end: int, limit: int | None = None):
    """Yield (offset, length) of each AVCC NAL until a length or header stops making sense."""
    pos, n = start, 0
    while pos + 5 <= end and (limit is None or n < limit):
        ln = struct.unpack(">I", m[pos:pos + 4])[0]
        if ln == 0 or ln > MAX_NAL or pos + 4 + ln > end:
            return
        hb = m[pos + 4]
        if hb & 0x80 or hb & 0x1F not in NAL_OK:
            return
        yield pos + 4, ln
        pos += 4 + ln
        n += 1


def avcc_start(m, start: int, end: int, slack: int = 64) -> int:
    """Where the NAL chain really begins: some muxers (Windows Media Foundation) pad the head of mdat."""
    for s in range(start, min(start + slack, end)):
        if len(list(walk_avcc(m, s, end, limit=3))) == 3:
            return s
    return start


def first_picture(m, start: int, end: int) -> tuple[list[bytes], list[bytes]]:
    """(in-band SPS/PPS if any, slices of the first IDR picture)."""
    params, idr = [], []
    for off, ln in walk_avcc(m, start, end, limit=400):
        t = m[off] & 0x1F
        if t in (7, 8):
            params.append(bytes(m[off:off + ln]))
        elif t == 5:
            idr.append(bytes(m[off:off + ln]))
        elif idr and t == 1:
            break
    return params, idr


# --- finding the picture size ------------------------------------------------------------------

def _decode(sps: bytes, pps: bytes, idr: list[bytes], gray: bool = False) -> tuple[str, bytes]:
    data = b"".join(START + n for n in [sps, pps, *idr])
    out = ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"] if gray else ["-f", "null", "-"]
    p = subprocess.run(["ffmpeg", "-v", "warning", "-f", "h264", "-i", "pipe:0", *out],
                       input=data, capture_output=True, timeout=30)
    return p.stderr.decode(errors="replace").strip(), p.stdout


def _clean(f, pps, idr, w, h) -> bool:
    return _decode(build_sps(f, w, h), pps, idr)[0] == ""


def _exact(f, pps, idr, w, h) -> bool:
    """Right size: decodes clean, and one more MB row doesn't (a too-short frame also decodes
    clean, it just ignores the rest of the slice)."""
    return _clean(f, pps, idr, w, h) and not _clean(f, pps, idr, w, (-(-h // 16) + 1) * 16)


def _height(f, pps, idr, wm: int, lo: int, hi: int = 400) -> int:
    """Tallest clean height in MBs at width wm (clean is monotone: every height up to the real one)."""
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if _clean(f, pps, idr, wm * 16, mid * 16) else (lo, mid - 1)
    return lo


def smoothness(f, pps, idr, w, h) -> float:
    """Mean abs difference between neighboring pixels: a right-sized picture is smooth, a wrong
    stride shears it into noise. Lower is better."""
    _, raw = _decode(build_sps(f, w, h), pps, idr, gray=True)
    a = np.frombuffer(raw, np.uint8)
    if a.size != w * h:
        return float("inf")
    a = a.reshape(h, w).astype(np.int16)
    return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())


def find_size(f: dict, pps: bytes, idr: list[bytes], guesses: list[tuple[int, int]], deadline: float,
              workers: int = 8) -> tuple[int, int] | None:
    """Guesses first; then every width from 60 to 400 MBs on a 16-row frame (a wrong width trips
    the decoder within a few rows), and the tallest clean height at each width that survives."""
    with cf.ThreadPoolExecutor(workers) as ex:
        clean = [wh for wh, ok in zip(guesses, ex.map(lambda wh: _exact(f, pps, idr, *wh), guesses)) if ok]
        if not clean and time.time() < deadline:
            wms = range(60, 401)
            ok = [wm for wm, c in zip(wms, ex.map(lambda wm: _clean(f, pps, idr, wm * 16, 256), wms)) if c]
            if time.time() < deadline:
                clean = [(wm * 16, hm * 16) for wm, hm in
                         zip(ok, ex.map(lambda wm: _height(f, pps, idr, wm, 16), ok[:6])) if hm > 16]
    if len(clean) <= 1:
        return clean[0] if clean else None
    # Several shapes decode: the smoothest picture wins (a wrong stride shears it into noise),
    # ties go to the earlier guess (common sizes carry their real crop).
    return min((smoothness(f, pps, idr, *wh), i, wh) for i, wh in enumerate(clean))[2]


# --- donor & repair ----------------------------------------------------------------------------

def brand(path: Path) -> bytes:
    with open(path, "rb") as fh:
        head = fh.read(12)
    return head[8:12] if head[4:8] == b"ftyp" else b""


def rank_donors(broken: Path, donors: list[dict]) -> list[dict]:
    """Best donors first: same container brand, same folder, same extension, then nearest size."""
    b = brand(broken)

    def key(d):
        p = Path(d["path"])
        return (brand(p) != b, p.parent != broken.parent, p.suffix.lower() != broken.suffix.lower(),
                abs(p.stat().st_size - broken.stat().st_size))
    return sorted(donors, key=key)


def fps_of(stream: dict | None, broken: Path) -> float:
    """Donor frame rate: avg_frame_rate when it's a real rate, r_frame_rate when the donor is VFR
    (screen recordings average out to odd numbers like 54.6), else 60 for screen recordings, 30."""
    def rate(s):
        try:
            n, d = (s or "0/0").split("/")
            return float(n) / float(d)
        except (ValueError, ZeroDivisionError):
            return 0.0
    std = (23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60, 90, 100, 119.88, 120)
    if stream:
        for v in (rate(stream.get("avg_frame_rate")), rate(stream.get("r_frame_rate"))):
            if any(abs(v - s) / s < 0.003 for s in std):
                return round(v, 3)
    return 60.0 if "screen" in broken.name.lower() or brand(broken) == b"qt  " else 30.0


def repair(src: Path, out: Path, donors: list[dict], budget: float = 60, workers: int = 8) -> dict:
    """Rebuild src into out (mp4, video only). donors: probed intact H.264 files ({"path", "stream"}).
    Returns {"ok", "width", "height", "fps", "frames", "donor", "notes", "reason"}; never raises."""
    deadline = time.time() + budget
    res = {"ok": False, "notes": []}
    try:
        with open(src, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            start, end = mdat_range(m)
            start = avcc_start(m, start, end)
            inband, idr = first_picture(m, start, end)
            if not idr:
                res["reason"] = ("no H.264 IDR at the start of mdat (not AVCC H.264: HEVC, or audio/other "
                                 "data interleaved first)")
                return res
            choice = None
            sps_in = [n for n in inband if n[0] & 0x1F == 7]
            pps_in = [n for n in inband if n[0] & 0x1F == 8]
            if sps_in and pps_in and _decode(sps_in[0], pps_in[0], idr)[0] == "":
                choice = (sps_in[0], pps_in[0], None, sps_size(parse_sps(sps_in[0])))
            tried = [] if choice else rank_donors(src, donors)[:3]
            for d in tried:
                if time.time() > deadline:
                    break
                try:
                    sps, pps = parameter_sets(Path(d["path"]))
                    f = parse_sps(sps)
                except (ValueError, subprocess.SubprocessError):
                    continue
                size = find_size(f, pps, idr, [sps_size(f)] + COMMON, deadline, workers)
                if size:
                    choice = (build_sps(f, *size), pps, d, size)
                    break
            if not choice:
                res["reason"] = (f"no donor SPS/PPS decodes its first frame cleanly (tried "
                                 f"{', '.join(Path(d['path']).name for d in tried)}; different encoder "
                                 "settings, e.g. an OBS or HEVC capture, or no intact H.264 file from the "
                                 "same recorder in this batch)") if donors else "no intact H.264 donor file in this batch"
                return res
            sps, pps, donor, size = choice
            fps = fps_of(donor and donor.get("stream"), src)
            tmp = out.with_name(out.name + ".part.mp4")
            ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-r", f"{fps:g}", "-f", "h264", "-i", "pipe:0",
                                   "-c", "copy", str(tmp)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            frames, last = 0, start
            try:
                ff.stdin.write(START + sps + START + pps)
                for off, ln in walk_avcc(m, start, end):
                    t = m[off] & 0x1F
                    last = off + ln
                    if t in (7, 8) and choice[2] is not None:
                        continue  # the borrowed SPS/PPS rule
                    frames += t in (1, 5)
                    ff.stdin.write(START)
                    ff.stdin.write(m[off:off + ln])
                ff.stdin.close()
            except BrokenPipeError:
                pass
            err = ff.stderr.read().decode(errors="replace")
            if ff.wait() != 0:
                res["reason"] = f"remux failed: {err.strip()[-300:]}"
                tmp.unlink(missing_ok=True)
                return res
            os.replace(tmp, out)
        res.update(ok=True, width=size[0], height=size[1], fps=fps,
                   frames=frames, donor=donor["path"] if donor else None,
                   used_mb=round((last - start) / 1e6, 1), mdat_mb=round((end - start) / 1e6, 1))
        res["notes"].append(f"moov rebuilt from {Path(donor['path']).name if donor else 'in-band SPS/PPS'}"
                            f", video only (audio isn't recoverable without the moov)")
        if end - last > max(4e6, 0.02 * (end - start)):
            res["stopped_early"] = True
            res["notes"].append(f"stream stops at {(last - start) / 1e6:.0f} of {(end - start) / 1e6:.0f} MB "
                                f"(audio or damaged data interleaved after that point)")
        return res
    except Exception as e:  # one bad file must never take the batch down
        res["reason"] = f"{type(e).__name__}: {e}"
        return res
