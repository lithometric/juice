import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from juice.media import ingest, recover

# A real macOS screen-recording SPS (Main profile, 2940x1846, cropped, with VUI).
APPLE_SPS = bytes.fromhex("274d0034ab2805c01d3dcd3501010102")
FPS, SECONDS = 30, 2


def encode(out: Path, w: int, h: int):
    # Same encoder settings for every clip (like one recorder), moov at the end (no faststart).
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:r={FPS}:d={SECONDS}",
                    "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-g", "30", str(out)], check=True)


def frames(path: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return int(out.stdout.strip())


def truncate(src: Path, dst: Path, frac: float):
    data = src.read_bytes()
    dst.write_bytes(data[:int(len(data) * frac)])


def test_sps_round_trip():
    f = recover.parse_sps(APPLE_SPS)
    assert recover.sps_size(f) == (2940, 1846)
    assert f["profile"] == 77 and f["vui"] == 1
    for w, h in [(2940, 1846), (2784, 1728), (1920, 1080), (1040, 592), (722, 406)]:
        g = recover.parse_sps(recover.build_sps(f, w, h))
        assert recover.sps_size(g) == (w, h)
        assert g["vui"] == 0 and g["head_bits"] == f["head_bits"]


def test_emulation_prevention():
    raw = bytes([0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 7, 0, 0])
    esc = recover.escape(raw)
    assert b"\x00\x00\x00" not in esc and b"\x00\x00\x01" not in esc
    assert recover.unescape(esc) == raw


@pytest.fixture(scope="module")
def pile(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("pile")
    src = d / "src"
    src.mkdir()
    encode(src / "donor.mov", 640, 360)
    encode(src / "wide.mov", 1040, 592)  # not a common size: found by the width sweep
    inbox = d / "downloads"
    inbox.mkdir()
    (inbox / "donor.mov").write_bytes((src / "donor.mov").read_bytes())
    truncate(src / "wide.mov", inbox / "cut.mov", 0.8)  # moov (at the end) is gone
    # A Drive zip that stalled mid-download: Chrome's "Unconfirmed 123.crdownload", no central directory.
    z = d / "full.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        zf.write(src / "donor.mov", "SCREEN RECORDINGS/a_donor.mov")
        zf.write(src / "wide.mov", "SCREEN RECORDINGS/b_wide.mov")
    truncate(z, inbox / "Unconfirmed 123.crdownload", 0.85)
    (inbox / "notes.txt").write_text("not media")
    return d


@pytest.fixture(scope="module")
def run(pile) -> dict:
    out = pile / "footage"
    res = ingest.ingest([pile / "downloads"], out, workers=4, transcribe=False, log=lambda *a: None)
    return {"out": out, "res": res, "man": json.loads((out / "manifest.json").read_text())}


def clip(run, name):
    return next(c for c in run["man"]["clips"] if c["name"] == name)


def test_sniffs_by_magic_not_extension(pile):
    assert ingest.sniff(pile / "downloads" / "Unconfirmed 123.crdownload") == "archive"
    assert ingest.sniff(pile / "downloads" / "cut.mov") == "media"
    assert ingest.sniff(pile / "downloads" / "notes.txt") is None


def test_truncated_zip_streams_out_complete_member(run):
    (a,) = run["man"]["archives"]
    assert a["truncated"] and a["partial_member"].endswith("b_wide.mov")
    assert Path(a["dir"]).is_relative_to(run["out"] / "raw")
    donor = clip(run, "a_donor.mov")
    assert donor["status"] == "ok" and (donor["width"], donor["height"]) == (640, 360)


def test_repairs_moov_less_file(run, pile):
    c = clip(run, "cut.mov")
    assert c["status"] == "recovered", c["notes"]
    assert (c["width"], c["height"]) == (1040, 592)
    assert Path(c["path"]).is_relative_to(run["out"] / "recovered")
    n = frames(Path(c["path"]))
    assert 0.5 * FPS * SECONDS <= n <= FPS * SECONDS
    assert c["fps"] == FPS


def test_member_cut_by_truncated_archive_is_recovered_and_noted(run):
    # Cut mid-download means moov-less: "recovered", with a note that the recording ran longer.
    c = clip(run, "b_wide.mov")
    assert c["status"] == "recovered" and (c["width"], c["height"]) == (1040, 592)
    assert any("download stopped" in n for n in c["notes"])


def test_outputs(run):
    out, man = run["out"], run["man"]
    assert man["done"] is True
    assert {c["name"] for c in man["clips"]} == {"donor.mov", "cut.mov", "a_donor.mov", "b_wide.mov"}
    assert "notes.txt" in man["skipped"]
    for c in man["clips"]:
        assert {"path", "original", "status", "notes", "duration", "width", "height", "fps", "codec",
                "has_audio", "speech"} <= c.keys()
    assert (out / "contact.jpg").stat().st_size > 10_000
    md = (out / "INGEST.md").read_text()
    assert "truncated" in md and "cut.mov" in md
    # Nothing written outside the output folder.
    assert sorted(p.name for p in out.parent.iterdir()) == ["downloads", "footage", "full.zip", "src"]


def test_unrecoverable_without_donor(tmp_path):
    encode(tmp_path / "x.mov", 640, 360)
    truncate(tmp_path / "x.mov", tmp_path / "lonely.mov", 0.7)
    (tmp_path / "x.mov").unlink()
    res = ingest.ingest([tmp_path / "lonely.mov"], tmp_path / "out", workers=2, transcribe=False,
                        log=lambda *a: None)
    (c,) = res["clips"]
    assert c["status"] == "unrecoverable" and "donor" in " ".join(c["notes"])
