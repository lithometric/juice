"""Multicam: sync recordings by their sound, cut a group's angles on its clock, pick angles, hand off to Resolve."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from juice.media import sync
from juice.shortform import check, handoff, multicam, render, spec
from juice.shortform.spec import SpecError

SR = 48000


def speechlike(seconds: float, seed: int) -> np.ndarray:
    """Syllable-like bursts of band-limited noise: a pattern only one stretch of time matches."""
    rng = np.random.default_rng(seed)
    x, t = np.zeros(int(seconds * SR)), 0.0
    while t < seconds:
        on = rng.uniform(0.08, 0.3)
        a, b = int(t * SR), int(min(t + on, seconds) * SR)
        x[a:b] += rng.standard_normal(b - a) * np.hanning(b - a) * rng.uniform(0.2, 1.0)
        t += on + rng.uniform(0.03, 0.4)
    x = sosfiltfilt(butter(4, [200, 3000], btype="band", fs=SR, output="sos"), x)
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


def wav(x: np.ndarray, out: Path, channels: int = 1) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", str(channels), "-i", "-",
                    "-c:a", "pcm_s24le", str(out)], input=x.astype(np.float32).tobytes(), check=True)
    return out


def camera(out: Path, color: str, rec: Path, start: float, dur: float, af: str) -> Path:
    """A camera that started rolling at `start` on the recorder's clock: a flat colour, its own (worse) sound."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s=320x180:r=30",
                    "-ss", str(start), "-t", str(dur), "-i", str(rec), "-f", "lavfi",
                    "-i", "anoisesrc=c=pink:a=0.01:r=48000", "-filter_complex",
                    f"[1:a]{af}[s];[s][2:a]amix=inputs=2:duration=first:normalize=0[a]", "-map", "0:v", "-map", "[a]",
                    "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(out)], check=True)
    return out


@pytest.fixture(scope="module")
def shoot(tmp_path_factory):
    d = tmp_path_factory.mktemp("shoot")
    rec = wav(speechlike(100, 1), d / "rec.wav")
    a = camera(d / "camA.mp4", "red", rec, 5.0, 90, "lowpass=f=3000,aecho=0.8:0.6:40|70:0.4|0.25,volume=-8dB")
    # camB's clock runs fast against the recorder's: 48002 / 48000 = 41.7 ppm (sample rates are whole numbers)
    b = camera(d / "camB.mp4", "blue", rec, 12.25, 86, "highpass=f=300,volume=-3dB,asetrate=48002,aresample=48000")
    other = d / "other.wav"
    wav(speechlike(30, 99), other)            # sound from somewhere else entirely
    data = sync.align([a, b, other, rec])
    sync.write(data, d / "sync.json")
    return d, data


def by_name(data, name):
    return next(c for c in data["clips"] if c["name"] == name)


def test_sync_finds_each_cameras_offset_and_drift(shoot):
    _, data = shoot
    assert data["clock"] == "rec.wav"                 # the recorder with no picture is the clock
    a, b = by_name(data, "camA.mp4"), by_name(data, "camB.mp4")
    assert abs(a["offset"] - 5.0) < 0.002 and abs(a["drift_ppm"]) < 5
    assert abs(b["offset"] - 12.25) < 0.002 and abs(b["drift_ppm"] - 41.7) < 3
    assert data["unplaced"] == ["other.wav"]          # no shared sound: not forced onto the clock
    assert (a["angle"], b["angle"]) == ("camA", "camB")


def reel(d: Path, shots: list, **extra) -> Path:
    p = d / "reel.json"
    p.write_text(json.dumps({"orientation": "landscape", "length": "long", "multicam": {"pod": "sync.json"},
                             "original_audio": True, "style": {"hook": {"require_text": False}},
                             "shots": [{"source": "pod", **s} for s in shots], **extra}))
    return p


def test_shots_are_timed_on_the_group_clock(shoot):
    d, _ = shoot
    sp = spec.load(reel(d, [{"angle": "camA", "start": 10, "end": 12}, {"angle": "camB", "start": 40, "end": 42}]))
    a, b = sp.shots
    assert sp.video(a).name == "camA.mp4" and sp.video(b).name == "camB.mp4"
    assert sp.sources["pod"].name == "rec.wav"        # the sound, transcript and clock are the recorder's
    assert abs(sp.local(a, 10) - 5.0) < 0.002
    assert abs(sp.local(b, 40) - (40 - 12.25) / (48002 / 48000)) < 0.002
    with pytest.raises(SpecError, match="isn't rolling"):
        spec.load(reel(d, [{"angle": "camB", "start": 6, "end": 8}]))
    with pytest.raises(SpecError, match="no camera angle"):
        spec.load(reel(d, [{"angle": "rec", "start": 20, "end": 21}]))


def test_the_same_moment_from_another_angle_is_reused_footage(shoot):
    d, _ = shoot
    sp = spec.load(reel(d, [{"angle": "camA", "start": 20, "end": 22}, {"angle": "camB", "start": 21, "end": 23}]))
    probs = check.footage_problems(sp)
    assert any("reuse the same footage" in p for p in probs)
    sp = spec.load(reel(d, [{"angle": "camA", "start": 20, "end": 22}]))
    assert any("angles never used: camB" in p for p in check.footage_problems(sp))


def test_cut_takes_picture_from_the_angle_and_sound_from_the_clock(shoot):
    d, _ = shoot
    sp = spec.preview(spec.load(reel(d, [{"angle": "camA", "start": 30, "end": 31.5},
                                         {"angle": "camB", "start": 50, "end": 51.5}])))
    rep = render.render(sp)
    out = Path(rep["output"])
    assert [s["angle"] for s in rep["shots"]] == ["camA", "camB"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(out), "-vf", "select='eq(n\\,3)+eq(n\\,30)'", "-fps_mode", "passthrough",
                          "-s", "4x4", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    f1, f2 = np.frombuffer(raw[:48], np.uint8).reshape(16, 3).mean(0), np.frombuffer(raw[48:96], np.uint8).reshape(16, 3).mean(0)
    assert f1[0] > 150 > f1[2] and f2[2] > 150 > f2[0]      # red camera, then blue camera
    ref, got = sync._audio(d / "rec.wav"), sync._audio(out)
    for o, g in ((rep["shots"][0]["at"], 30.0), (rep["shots"][1]["at"], 50.0)):   # the recorder's sound at each clock time
        cw = got[int((o + 0.2) * sync.SR): int((o + 1.2) * sync.SR)]
        r0 = int((g - 0.3) * sync.SR)
        k, _ = sync._gcc(ref[r0: r0 + len(cw) + int(sync.SR)], cw)
        assert abs((r0 + k) / sync.SR - 0.2 - g) < 0.002


def test_export_hands_resolve_the_cut_and_a_stack_of_synced_angles(shoot, tmp_path):
    d, _ = shoot
    sp = spec.load(reel(d, [{"angle": "camA", "start": 30, "end": 31.5, "note": "A talks"},
                            {"angle": "camB", "start": 50, "end": 51.5, "speed": 2}]))
    rep = handoff.export(sp, tmp_path)
    cut = json.loads(Path(rep["files"]["cut"]).read_text())
    v1, a1 = cut["tracks"]["children"][:2]
    clips = [c for c in v1["children"] if c["OTIO_SCHEMA"] == "Clip.2"]
    assert [c["media_references"]["DEFAULT_MEDIA"]["target_url"].rsplit("/", 1)[1] for c in clips] == ["camA.mp4", "camB.mp4"]
    assert clips[0]["source_range"]["start_time"]["value"] == round(sp.local(sp.shots[0], 30) * 30)
    assert clips[1]["effects"][0]["time_scalar"] == 2
    assert a1["kind"] == "Audio" and a1["children"][0]["source_range"]["start_time"]["value"] == 30 * 30
    assert cut["tracks"]["markers"][0]["comment"] == "A talks"
    ang = json.loads(Path(rep["files"]["angles_pod"]).read_text())
    tracks = {t["name"]: t for t in ang["tracks"]["children"]}
    assert tracks["camA"]["kind"] == tracks["camB"]["kind"] == "Video"
    gap = tracks["camB"]["children"][0]
    assert gap["OTIO_SCHEMA"] == "Gap.1" and gap["source_range"]["duration"]["value"] == round((12.25 - 0) * 30)
    assert "Convert Compound" in (tmp_path / "README.md").read_text()


# --- picking angles: the editing rules, with faces stood in -----------------------------------------------

WORDS = ([("A", 0.5 + 0.4 * i) for i in range(9)] +        # A: 0.5-4.1
         [("B", 4.5 + 0.4 * i) for i in range(12)] +       # B: 4.5-9.3
         [("A", 9.6), ("A", 10.0)] +                       # A cuts in for under a second
         [("B", 10.6 + 0.4 * i) for i in range(8)] +       # B carries on: 10.6-13.8
         [("AB", 14.2 + 0.4 * i) for i in range(6)] +      # both at once: 14.2-16.6
         [("A", 17.0 + 0.4 * i) for i in range(6)])        # A: 17.0-19.4


@pytest.fixture
def talk(tmp_path, monkeypatch):
    n = int(20 * SR)
    ch = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(3)
    for who, t in WORDS:
        a, b = int(t * SR), int((t + 0.3) * SR)
        burst = rng.standard_normal(b - a) * np.hanning(b - a) * 0.4
        for c, me in ((0, "A"), (1, "B")):
            ch[a:b, c] += burst * (1.0 if me in who else 0.1)     # each lav hears the other 20 dB down
    wav(ch.reshape(-1), tmp_path / "lavs.wav", channels=2)
    wav(ch.mean(1), tmp_path / "mono.wav")
    clips = [{"name": n, "path": str(tmp_path / n), "duration": 20.0, "video": v, "audio": True, "offset": 0.0,
              "rate": 1.0, "angle": Path(n).stem, "camera": n} for n, v in
             (("closeA.mp4", True), ("closeB.mp4", True), ("wide.mp4", True), ("lavs.wav", False), ("mono.wav", False))]
    (tmp_path / "sync.json").write_text(json.dumps({"clock": "lavs.wav", "clips": clips}))
    grid_talk = {"A": [t for w, t in WORDS if "A" in w], "B": [t for w, t in WORDS if "B" in w]}

    def fake_look(group, name, grid):     # each close-up's mouth opens while its person talks
        def subject(who, x, size):
            z = np.full(len(grid), -0.5)
            for t in grid_talk.get(who, []):
                z[(grid >= t) & (grid < t + 0.3)] = 2.0
            return multicam.Subject(angle=name, x=x, sizes=[size] * len(grid), mouth=z, z=z)
        if name == "wide":
            return [subject(None, 0.3, 0.12), subject(None, 0.7, 0.12)]
        return [subject(name[-1], 0.5, 0.3)]
    monkeypatch.setattr(multicam, "look", fake_look)
    words = [{"text": "w", "start": t, "end": t + 0.3, "type": "word", "speaker_id": "speaker_0"} for _, t in WORDS]
    return tmp_path, words


def cuts(res):
    return [(s["angle"], round(s["start"], 2)) for s in res["shots"]]


def test_picks_the_talker_from_a_mic_per_person(talk):
    d, words = talk
    res = multicam.plan(spec.load_group(d, "sync.json"), words, max_shot=30)
    assert res["evidence"] == "mics" and res["cast"] == {"mic 1": "closeA", "mic 2": "closeB"}
    # B's turn cuts 0.08 s before B's first word; A's interjection is held on B; two at once is the wide
    assert cuts(res) == [("closeA", 0.42), ("closeB", 4.42), ("wide", 14.12), ("closeA", 16.92)]


def test_picks_the_talker_from_mouths_with_one_mic(talk):
    d, words = talk
    res = multicam.plan(spec.load_group(d, {"sync": "sync.json", "audio": "mono.wav"}), words, max_shot=30)
    assert res["evidence"] == "mouths"
    assert cuts(res)[:2] == [("closeA", 0.42), ("closeB", 4.42)]


def test_a_long_turn_cuts_away_at_a_pause(talk):
    d, words = talk
    res = multicam.plan(spec.load_group(d, "sync.json"), words, max_shot=6)
    away = [s for s in res["shots"] if s["note"].startswith("long turn")]
    assert away and all(s["end"] - s["start"] >= 1.5 for s in res["shots"])
    assert all(s["end"] - s["start"] <= 6 + 4 for s in res["shots"])
