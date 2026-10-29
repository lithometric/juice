import json
from pathlib import Path

from juice import retention, spec, timeline


def make(tmp_path: Path, shots, words=None, **over) -> spec.Spec:
    (tmp_path / "clip.mov").write_bytes(b"")
    body = {"sources": {"main": "clip.mov"}, "shots": shots, "original_audio": True, "theme": "clean"}
    if words is not None:
        tp = tmp_path / "t.json"
        tp.write_text(json.dumps({"words": [{"text": w, "start": a, "end": b, "type": "word"} for w, a, b in words]}))
        body["transcripts"] = {"main": "t.json"}
    body.update(over)
    p = tmp_path / "reel.json"
    p.write_text(json.dumps(body))
    return spec.load(p)


def talk(start, end, wps=3.0):
    """Evenly spaced words over a source range."""
    n = int((end - start) * wps)
    step = (end - start) / n
    return [(f"w{i}", round(start + i * step, 3), round(start + i * step + step * 0.8, 3)) for i in range(n)]


def test_retention_falls_with_length_and_is_front_loaded(tmp_path):
    short = retention.score(make(tmp_path, [{"source": "main", "start": 0, "end": 15}], talk(0, 15)))
    long_ = retention.score(make(tmp_path, [{"source": "main", "start": 0, "end": 60}], talk(0, 60)))
    assert short["avp"] > long_["avp"]
    assert short["retention_3s"] > short["retention_end"]
    # the first 3s lose more than the next 3s (k < 1: hazard decreases)
    c = dict(short["curve"])
    assert (1 - c[3.0 + 0.05] if 3.05 in c else 1 - short["retention_3s"]) > (short["retention_3s"] - c.get(6.05, short["retention_end"]))


def test_captions_and_question_hook_lift_the_score(tmp_path):
    base = make(tmp_path, [{"source": "main", "start": 0, "end": 20}], talk(0, 20))
    plain = retention.score(base)
    better = make(tmp_path, [{"source": "main", "start": 0, "end": 20,
                              "fx": [{"type": "title", "text": "really?", "at": 0.1, "dur": 2}]}], talk(0, 20),
                  overlays=[{"type": "captions"}, {"type": "progress"}])
    good = retention.score(better)
    assert good["avp"] > plain["avp"]
    assert "hook_question" in good["hook"] and "hook_question" not in plain["hook"]
    assert good["captions_coverage"] > 0.9
    assert any("captions" in t for t in plain["tips"]) and not any("captions" in t for t in good["tips"])


def test_dead_air_repeats_and_no_change_show_up_as_peaks(tmp_path):
    words = talk(0, 4) + [("insane", 7.0, 7.4)] + talk(8, 30, 2.5)        # 3s of silence from 4 to 7
    words += [("the", 30.0, 30.2), ("more", 30.2, 30.4), ("favors", 30.4, 30.7),
              ("the", 31.0, 31.2), ("more", 31.2, 31.4), ("favors", 31.4, 31.7)]      # a restart
    s = make(tmp_path, [{"source": "main", "start": 0, "end": 32}], words)
    rep = retention.score(s)
    whys = [set(p["why"]) for p in rep["peaks"]]
    assert any("silence" in w for w in whys)
    assert any("repeat" in w for w in whys)
    assert any("change" in w for w in whys)           # one 32s shot with no graphics
    assert rep["peaks"][0]["lost_pct"] >= rep["peaks"][-1]["lost_pct"]
    # cutting the pause out removes the silence peak
    cut = make(tmp_path, [{"source": "main", "start": 0, "end": 4.1}, {"source": "main", "start": 6.9, "end": 32}], words)
    rep2 = retention.score(cut)
    assert not any("silence" in set(p["why"]) for p in rep2["peaks"])
    assert rep2["avp"] > rep["avp"]


def test_push_ins_count_as_visual_change(tmp_path):
    still = retention.score(make(tmp_path, [{"source": "main", "start": 0, "end": 12}], talk(0, 12)))
    moving = retention.score(make(tmp_path, [{"source": "main", "start": 0, "end": 12, "zoom": [1.0, 1.15]}], talk(0, 12)))
    assert moving["avp"] > still["avp"]


def test_plot_writes_png(tmp_path):
    s = make(tmp_path, [{"source": "main", "start": 0, "end": 10}], talk(0, 10))
    tl = timeline.build(s)
    out = retention.plot(retention.score(s, tl), tmp_path / "r.png", tl)
    assert out.exists() and out.stat().st_size > 1000
