"""Hardware (VideoToolbox) and software encoding and decoding: the choice, the safety check on decode, the
fallback to software, and that hardware masters concatenate and keep brand colours exact."""
import json
import subprocess
import sys

import numpy as np
import pytest

from juice import encode
from juice.motion import Video, scenes as s

HW = encode.hardware_encoder(10) is not None
needs_hw = pytest.mark.skipif(not HW, reason="no media engine (VideoToolbox) on this machine")


def frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,pix_fmt,nb_read_frames", "-of", "json", str(path)], capture_output=True, text=True)
    st = json.loads(r.stdout)["streams"][0]
    return st["codec_name"], st["pix_fmt"], int(st["nb_read_frames"])


def decode_errors(path) -> str:
    return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], capture_output=True, text=True).stderr


def clip(path, *enc, d=2):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=640x360:r=30:d={d}", *enc, str(path)],
                   check=True)
    return path


@pytest.fixture(autouse=True)
def fresh(monkeypatch, tmp_path):
    for k in ("JUICE_ENCODER", "JUICE_DECODER"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(encode, "CACHE", tmp_path / "hwdecode.json")
    encode._decode.cache_clear()


def test_software_when_asked(monkeypatch):
    monkeypatch.setenv("JUICE_ENCODER", "software")
    assert encode.video_args({"bit_depth": 10})[1] == "libx265"
    assert encode.video_args({"bit_depth": 8})[1] == "libx264"
    assert encode.pix_fmt({"bit_depth": 10}) == "yuv420p10le"
    monkeypatch.delenv("JUICE_ENCODER")
    assert encode.video_args({"bit_depth": 10, "encoder": "software"})[1] == "libx265"
    with pytest.raises(ValueError):
        encode.video_args({"encoder": "gpu"})


@needs_hw
def test_hardware_by_default():
    a = encode.video_args({"bit_depth": 10})
    assert a[1] == "hevc_videotoolbox" and "main10" in a and "p010le" in a and "hvc1" in a and "bt709" in a
    assert encode.video_args({"bit_depth": 8})[1] == "h264_videotoolbox"
    assert encode.pix_fmt({"bit_depth": 10}) == "p010le"
    assert encode.software({"bit_depth": 10})["encoder"] == "software"


@needs_hw
def test_decode_only_where_hardware_matches_software(tmp_path):
    hevc = clip(tmp_path / "hevc.mp4", "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error")
    avc = clip(tmp_path / "avc.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p")
    sony = clip(tmp_path / "xavc.mp4", "-c:v", "libx264", "-pix_fmt", "yuv422p10le")   # Sony's 4:2:2 10-bit H.264
    assert encode.decode_args(hevc) == ["-hwaccel", "videotoolbox"]
    assert encode.decode_args(avc) == []           # faster on the CPU by default...
    assert encode.decode_args(avc, {"decoder": "hardware"}) == ["-hwaccel", "videotoolbox"]   # ...unless asked
    assert encode.decode_args(sony, {"decoder": "hardware"}) == []   # never tried: the engine drops its frames
    cache = json.loads(encode.CACHE.read_text())
    assert {v["codec"] for v in cache.values()} == {"hevc", "h264"}
    assert all(v["frames"][0] == v["frames"][1] for v in cache.values())
    assert encode.decode_args(hevc, {"decoder": "software"}) == []


@needs_hw
def test_a_file_hardware_decodes_short_stays_on_software(tmp_path, monkeypatch):
    """The Sony failure: the media engine quietly returns fewer frames than there are packets."""
    hevc = clip(tmp_path / "hevc.mp4", "-c:v", "libx265", "-x265-params", "log-level=error")
    real = encode._count_cmd
    monkeypatch.setattr(encode, "_count_cmd", lambda p, hw, s: real(p, hw, s)[:-3] + ["-frames:v", "5", "-f", "null", "-"]
                        if hw else real(p, hw, s))
    assert encode.decode_args(hevc) == []
    entry = json.loads(encode.CACHE.read_text()).popitem()[1]
    hw_frames, sw_frames = entry["frames"]
    assert not entry["ok"] and hw_frames == 5 and sw_frames > 5


def _video(src):
    v = Video("landscape", "short")
    v.scene(s.Footage(v.source(src, 0, 1.5)), 0.0, 1.5, energy=0)
    v.scene(s.EndCard(line="Sign up at"), 1.5, 3.0, transition="flash", energy=0)
    return v


@needs_hw
def test_hardware_chunks_concatenate_cleanly(tmp_path):
    src = clip(tmp_path / "take.mp4", "-c:v", "libx265", "-pix_fmt", "yuv420p10le", "-x265-params", "log-level=error", d=4)
    rep = _video(src).render(tmp_path / "hw.mp4", workers=6)
    assert rep["encoder"] == "hevc_videotoolbox" and rep["workers"] > 1
    assert frames(tmp_path / "hw.mp4") == ("hevc", "yuv420p10le", rep["frames"])
    assert decode_errors(tmp_path / "hw.mp4") == ""


def test_motion_renders_from_a_process_that_ran_threads(tmp_path):
    """An MCP server, an agent that rendered a cut first, a test run: the forked workers must still draw type."""
    import threading
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    src = clip(tmp_path / "take.mp4", "-c:v", "libx264", d=4)
    rep = _video(src).render(tmp_path / "threaded.mp4", workers=3, preview=True)
    assert frames(tmp_path / "threaded.mp4")[2] == rep["frames"]


@needs_hw
def test_failed_hardware_encode_falls_back_to_software(tmp_path, monkeypatch):
    src = clip(tmp_path / "take.mp4", "-c:v", "libx264", d=4)
    real = encode._vt_args
    monkeypatch.setattr(encode, "_vt_args", lambda depth, q: real(depth, q) + ["-bogus_option", "1"])
    rep = _video(src).render(tmp_path / "fallback.mp4", workers=3)
    assert rep["encoder"] == "libx265"
    assert frames(tmp_path / "fallback.mp4")[2] == rep["frames"]


@needs_hw
@pytest.mark.parametrize("encoder", ["hardware", "software"])
def test_brand_colour_survives_the_master(tmp_path, encoder):
    """Read Y'CbCr back and convert with the BT.709 equations: the file holds the brand colour itself."""
    green = np.array([0x00, 0xDD, 0x94], np.uint8)
    W, H = 320, 240
    out = tmp_path / f"{encoder}.mp4"
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30",
                          "-i", "-", *encode.video_args({"bit_depth": 10, "encoder": encoder}), str(out)], stdin=subprocess.PIPE)
    p.communicate(np.tile(green, (H, W, 1)).tobytes() * 5)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-frames:v", "1", "-vf", "format=yuv444p16le",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    y, cb, cr = np.frombuffer(raw, np.uint16).reshape(3, H, W)[:, H // 2, W // 2] / 64
    Y, Pb, Pr = (y - 64) / 876, (cb - 512) / 896, (cr - 512) / 896
    R, B = Y + 1.5748 * Pr, Y + 1.8556 * Pb
    G = (Y - 0.2126 * R - 0.0722 * B) / 0.7152
    got = np.clip(np.array([R, G, B]) * 255, 0, 255)
    assert np.abs(got - green).max() <= 1.5, got.round()


def test_no_media_engine_means_software(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    encode.hardware_encoder.cache_clear()
    try:
        assert encode.video_args({"bit_depth": 10})[1] == "libx265"
        assert encode.decode_args(__file__) == []
    finally:
        monkeypatch.undo()
        encode.hardware_encoder.cache_clear()
