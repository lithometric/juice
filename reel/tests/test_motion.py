import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from juice import beds, render, spec
from juice.motion import Video, scenes as s


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                        "stream=codec_type,width,height,nb_read_frames:format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def footage(tmp_path_factory):
    d = tmp_path_factory.mktemp("footage")
    clip = d / "take.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=4",
                    "-f", "lavfi", "-i", "sine=f=220:d=4", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", str(clip)], check=True)
    (d / "juice.json").write_text(json.dumps({"juice": "1", "name": "Acme", "url": "https://acme.test",
                                              "colors": {"primary": "#F2C21B", "secondary": "#3EBE7A"}}))
    return d


def build(footage, orientation):
    take = footage / "take.mp4"
    v = Video(orientation, "short", brand=footage / "juice.json")
    vo = v.voice(take, 0.0, 1.0, at=0.0, isolate=False)   # isolation has its own test
    v.scene(s.Footage(v.source(take, 0, 1)), 0.0, 1.0, energy=0)
    v.overlay(s.Headline("Now in beta.", highlight={"beta.": "accent"}), 0.1, 1.0)
    v.scene(s.LogoReveal(badge="beta", tagline="A tagline"), 1.0, 2.0, transition="flash")
    v.scene(s.Panels([v.source(take, 1, 2), v.source(take, 2, 3)], callouts=[(1, 0.3, 0.4, "A callout")], chip="LIVE"),
            2.0, 2.6, transition="logo")
    v.scene(s.Linked(v.source(take, 0, 1), v.source(take, 1, 2), sub="Sub line"), 2.6, 3.0, transition="whip")
    v.overlay(s.Slam("BIG WORD"), 2.6, 3.0)
    v.scene(s.FeatureGrid("A grid.", [s.Card("One", v.source(take, 0, 1), rec=True), s.Card("Two", url="https://x.test",
                                                                                            button="Copy")]), 3.0, 3.4)
    v.scene(s.Notification(v.source(take, 1, 2), "Ready.", body="Body", button_label="Go"), 3.4, 3.8)
    v.overlay(s.UrlBar("acme.test", (3.4, 3.6)), 3.4, 3.8)
    v.scene(s.EndCard(line="Sign up at", badge="beta"), 3.8, 4.4, transition="iris", energy=0)
    v.hit(vo.out(0.5), "slam")
    v.music("tech")
    return v


@pytest.mark.parametrize("orientation,size", [("landscape", (1920, 1080)), ("portrait", (1080, 1920))])
def test_every_piece_renders_in_both_orientations(footage, tmp_path, orientation, size):
    v = build(footage, orientation)
    sheet = v.stills([0.5, 1.6, 2.3, 2.8, 3.2, 3.6, 4.2], tmp_path / "sheet.jpg", scale=0.25)
    assert sheet.exists() and sheet.stat().st_size > 10_000
    assert (v.W, v.H) == size


def test_preview_and_full_render(footage, tmp_path):
    v = build(footage, "landscape")
    pre = v.render(tmp_path / "pre.mp4", preview=True, workers=3)
    full = v.render(tmp_path / "full.mp4", workers=3)
    p, f = probe(pre["output"]), probe(full["output"])
    vp = next(x for x in p["streams"] if x["codec_type"] == "video")
    vf = next(x for x in f["streams"] if x["codec_type"] == "video")
    assert (int(vp["height"]), int(vf["height"])) == (480, 1080)
    # Chunks concatenate without dropping or duplicating frames (the audio sets the end).
    assert abs(int(vf["nb_read_frames"]) - 4.4 * 30) <= 1 and abs(int(vp["nb_read_frames"]) - 4.4 * 15) <= 1
    assert any(x["codec_type"] == "audio" for x in f["streams"])
    assert abs(float(f["format"]["duration"]) - 4.4) < 0.05
    assert pre["render_s"] < full["render_s"]


def test_beds_are_reproducible_and_follow_the_sections():
    a = beds.arrange("hype", 6.0, [(0, 0), (2.0, 3)], beat0=2.0)
    b = beds.arrange("hype", 6.0, [(0, 0), (2.0, 3)], beat0=2.0)
    assert a.shape == (6 * beds.SR, 2) and np.array_equal(a, b)
    quiet, full = np.abs(a[:int(1.5 * beds.SR)]).mean(), np.abs(a[int(2.5 * beds.SR):]).mean()
    assert full > 2 * quiet  # the drone section sits well under the full groove
    p = beds.bed("calm", 3.0, [(0, 1)])
    assert p.exists() and beds.bed("calm", 3.0, [(0, 1)]) == p  # cached


def test_cut_preview_render(footage, tmp_path):
    (tmp_path / "take.mp4").symlink_to(footage / "take.mp4")
    (tmp_path / "reel.json").write_text(json.dumps({
        "sources": {"a": "take.mp4"}, "original_audio": True, "orientation": "landscape",
        "shots": [{"source": "a", "start": 0, "end": 1.5, "fx": [{"type": "chip", "text": "HELLO"}]},
                  {"source": "a", "start": 2, "end": 3.5}]}))
    sp = spec.preview(spec.load(tmp_path / "reel.json"))
    report = render.render(sp)
    v = next(x for x in probe(report["output"])["streams"] if x["codec_type"] == "video")
    assert Path(report["output"]).name == "final_preview.mp4"
    assert (int(v["width"]), int(v["height"])) == (854, 480) and abs(int(v["nb_read_frames"]) - 3.0 * 15) <= 1


def build_clean(footage, orientation):
    from juice.motion import clean as c
    take = footage / "take.mp4"
    v = Video(orientation, "short", brand=footage / "juice.json")
    vo = v.voice(take, 0.0, 1.0, at=0.0, isolate=False)   # isolation has its own test
    v.scene(c.Footage(v.source(take, 0, 1)), 0.0, 1.0, energy=0)
    v.overlay(c.Title("Now in beta.", kicker="Acme", highlight={"beta.": "accent"}, scrim=True), 0.1, 1.0)
    v.scene(c.Lockup(line="A tagline."), 1.0, 1.6, energy=1)
    v.scene(c.Frame(v.source(take, 1, 2), kicker="Product", title=["A long title", "on two lines."],
                    body=["Body one.", "Body two."], url="acme.test"), 1.6, 2.2, transition="wipe")
    v.scene(c.Split(v.source(take, 0, 1), v.source(take, 1, 2), left_label="Agent"), 2.2, 2.8, transition="dissolve")
    v.overlay(c.Terms([("If", 2.2), ("you", 2.3)], [("One", 2.4), ("Two", 2.6)]), 2.2, 2.8)
    v.scene(c.Bento("A grid.", [c.Card("One", v.source(take, 0, 1), rec=True), c.Card("Two", v.source(take, 1, 2)),
                                c.Card("Three", v.source(take, 2, 3)),
                                c.Card("Four", body="Body", url="https://x.test", button="Copy")], every=0.1),
            2.8, 3.4, energy=3)
    v.scene(c.Footage(v.source(take, 2, 3)), 3.4, 3.8)
    v.overlay(c.Caption("A caption", index="01"), 3.4, 3.8)
    v.overlay(c.Notice("Ready.", body="Body"), 3.5, 3.8)
    v.scene(c.Lockup(line="Sign up at", url=True, sub="One · Two"), 3.8, 4.4, transition="fade", energy=0)
    for k in ("sub", "tick", "air"):
        v.hit(vo.out(0.5), k)
    v.hit(1.6, "swell", until=1.0)
    v.music("future")
    return v


@pytest.mark.parametrize("orientation", ["landscape", "portrait"])
def test_clean_look_renders_in_both_orientations(footage, tmp_path, orientation):
    v = build_clean(footage, orientation)
    sheet = v.stills([0.6, 1.3, 1.75, 2.3, 2.7, 3.3, 3.7, 4.2], tmp_path / "sheet.jpg", scale=0.25)
    assert sheet.exists() and sheet.stat().st_size > 10_000
    import skia
    ctx = v._ctx(0.25)
    surf = skia.Surface(round(v.W * 0.25), round(v.H * 0.25))
    v._configure(0.25, v.fps)
    hits = v._all_hits()
    assert v.draw(surf, ctx, 3.8, hits).mean() < 3            # the fade dips to black on the cut
    assert v.draw(surf, ctx, 3.5, hits).mean() > 20
    for src in v.sources():
        src.close()
    assert any(h.kind == "air" and h.at < 1.6 for h in hits)  # the wipe brings its own air


def test_keynote_bed_breathes_with_the_kick():
    a = beds.arrange("keynote", 6.0, [(0, 0), (2.0, 3)], beat0=2.0)
    assert np.isfinite(a).all() and np.abs(a).max() <= 1.0
    quiet, full = np.abs(a[:int(1.5 * beds.SR)]).mean(), np.abs(a[int(2.5 * beds.SR):]).mean()
    assert full > 2 * quiet


def test_music_sits_at_the_voice_level(footage, tmp_path):
    v = build_clean(footage, "landscape")
    v.render_audio(tmp_path / "mix.wav")
    rep = v.audio_report
    assert abs(rep["lufs"] + 14) < 0.5
    assert abs(rep["voice_lufs"] - rep["music_lufs"]) < 1.5   # music between the lines = the voice
    v.bed["level"] = -6
    v.render_audio(tmp_path / "mix.wav")
    assert v.audio_report["voice_lufs"] - v.audio_report["music_lufs"] > 4


def test_future_bed_is_balanced():
    from scipy import signal
    x = beds.arrange("future", 8.0, [(0, 0), (2.0, 3)], beat0=2.0)
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0
    f, P = signal.welch(x[:, 0], beds.SR, nperseg=8192)
    band = lambda lo, hi: 10 * np.log10(P[(f >= lo) & (f < hi)].sum())
    assert band(20, 60) - band(500, 2000) < 6        # the sub doesn't swamp the mids


@pytest.mark.skipif(shutil.which("uvx") is None, reason="needs uvx for DeepFilterNet")
def test_isolation_removes_the_room(tmp_path):
    from juice import isolate
    from juice.motion import mix
    src = tmp_path / "noisy.wav"
    # 1 s of room noise, then a voiced buzz with formants over the same noise.
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "anoisesrc=d=2:c=pink:a=0.05:r=48000",
                    "-f", "lavfi", "-i", "sine=f=140:d=1:r=48000", "-filter_complex",
                    "[1]aformat=channel_layouts=mono,volume=0.4,adelay=1000,apad[v];[0][v]amix=inputs=2:duration=first",
                    str(src)], check=True)
    out = isolate.voices([src])[src]
    x, y = mix._decode(str(src)), mix._decode(str(out))
    noise = lambda a: 20 * np.log10(np.sqrt((a[int(0.2 * 48000):int(0.9 * 48000)] ** 2).mean()) + 1e-9)
    assert noise(y) < noise(x) - 10 and abs(len(x) - len(y)) < 4800   # ~34 dB on real takes
    assert isolate.voices([src])[src] == out   # cached
