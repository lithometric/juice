import json
from pathlib import Path

import pytest

from juice.shortform import spec, theme, timeline


def make(tmp_path: Path, **over) -> spec.Spec:
    subprocess_free_src = tmp_path / "clip.mov"
    subprocess_free_src.write_bytes(b"")
    body = {"sources": {"main": "clip.mov"},
            "shots": [{"source": "main", "start": 30.0, "end": 31.0, "replay": True},   # hook
                      {"source": "main", "start": 10.0, "end": 12.0},
                      {"source": "main", "start": 30.0, "end": 31.0, "speed": 0.5}]}
    body.update(over)
    p = tmp_path / "reel.json"
    p.write_text(json.dumps(body))
    return spec.load(p)


def test_theme_inherits_from_base():
    t = theme.load({"base": "clean", "colors": {"accent": [1, 2, 3]}})
    assert t["colors"]["accent"] == [1, 2, 3]
    assert t["fonts"]["display"]["family"] == "Inter"          # from clean
    assert "chip" in theme.load({"colors": {"accent": [0, 0, 0]}})  # partial theme filled from bold


def test_events_fire_where_the_source_moment_lands_but_not_in_replays(tmp_path):
    s = make(tmp_path, events=[{"t": 30.5, "fx": [{"type": "flash"}], "sfx": "impact"}])
    tl = timeline.build(s)
    # shot 0 (replay) is skipped; shot 2 starts at 3.0s and runs at half speed -> 3.0 + 0.5/0.5 = 4.0
    assert [round(a, 3) for a, _ in tl.sfx] == [4.0]
    flash = [L for L in tl.layers if L.comp == "flash"]
    assert len(flash) == 1 and flash[0].start == pytest.approx(4.0)


def test_slow_motion_gets_letterbox_and_badge(tmp_path):
    tl = timeline.build(make(tmp_path))
    kinds = {L.comp for L in tl.layers}
    assert {"letterbox", "chip"} <= kinds


def test_meter_steps_and_set(tmp_path):
    s = make(tmp_path, meters={"hp": 100},
             events=[{"t": 10.5, "meter": {"id": "hp", "delta": -30}}, {"t": 11.5, "meter": {"id": "hp", "set": 0}}])
    tl = timeline.build(s)
    ctx = timeline.Ctx(s, s.theme, tl)
    ctx.t_out = 0.2
    assert ctx.meter("hp") == pytest.approx(100)
    ctx.t_out = 1.5 + 0.5      # well after the first hit (lands at 1.5s)
    assert ctx.meter("hp") == pytest.approx(70)
    ctx.t_out = 2.5 + 0.5      # after "set 0"
    assert ctx.meter("hp") == pytest.approx(0)


def test_big_type_draws_over_pinned_labels(tmp_path):
    s = make(tmp_path, overlays=[{"type": "watermark", "text": "X"}],
             shots=[{"source": "main", "start": 0, "end": 1,
                     "fx": [{"type": "slam", "text": "K.O."}, {"type": "flash"}]}])
    order = [L.comp for L in timeline.build(s).layers]
    assert order.index("watermark") < order.index("slam") < order.index("flash")


def test_unknown_component_is_rejected(tmp_path):
    with pytest.raises(spec.SpecError, match="unknown fx"):
        make(tmp_path, shots=[{"source": "main", "start": 0, "end": 1, "fx": [{"type": "sparkles"}]}])


def test_hook_text_wants_one_short_line(tmp_path):
    from juice.shortform import hooks
    cfg = dict(hooks.HOOK_DEFAULTS)
    two = make(tmp_path, overlays=[
        {"type": "chip", "text": "they heard a food question. what we really asked:", "from": 0, "to": 3},
        {"type": "chip", "text": "how far would you drive for a hookup?", "from": 0, "to": 3}])
    info, probs = hooks.hook_text(two, timeline.build(two), cfg)
    assert info["words"] == 17 and any("one glance" in p for p in probs)
    one = make(tmp_path, overlays=[{"type": "chip", "text": "how far would you drive for a hookup?", "from": 0, "to": 3},
                                   {"type": "chip", "text": "they heard a food question.", "from": 2.0, "to": 3}])
    info, probs = hooks.hook_text(one, timeline.build(one), cfg)
    assert info["words"] == 8 and probs == []
    none = make(tmp_path)
    assert any("no on-screen text" in p for p in hooks.hook_text(none, timeline.build(none), cfg)[1])


def test_hook_spoken_flags_late_start_and_fillers(tmp_path):
    from juice.shortform import hooks
    t = tmp_path / "t.json"
    t.write_text(json.dumps({"words": [{"text": "Okay,", "start": 30.6, "end": 30.75, "type": "word"},
                                       {"text": "two", "start": 30.8, "end": 30.95, "type": "word"}]}))
    s = make(tmp_path, original_audio=True, transcripts={"main": "t.json"})
    s.shots[0].replay = False
    info, probs = hooks.hook_spoken(s, dict(hooks.HOOK_DEFAULTS))
    assert info["first_word_at"] == pytest.approx(0.6)
    assert any("start by" in p for p in probs) and any("filler" in p for p in probs)


def test_hook_score_prefers_a_standalone_answer_with_a_laugh():
    from juice.shortform.hooks import score
    base = dict(dur=1.8, complete=True, filler_start=False, interviewer_question=False, laugh_after=False,
                reaction_after=False, energy_db=0.0, face=0.05, motion=2.0)
    answer = score({**base, "laugh_after": True, "reaction_after": True})[0]
    question = score({**base, "interviewer_question": True})[0]
    rambling = score({**base, "dur": 9.0, "filler_start": True, "complete": False})[0]
    interviewer = score({**base, "interviewer": True, "laugh_after": True})[0]
    assert answer > question and answer > rambling and answer > interviewer


def test_caption_chunks_break_at_cuts_pauses_and_punctuation():
    from juice.shortform.components import _caption_chunks
    words = [{"text": "so", "start": 0.0, "end": 0.2}, {"text": "many", "start": 0.2, "end": 0.4},
             {"text": "people.", "start": 0.4, "end": 0.7}, {"text": "and", "start": 0.75, "end": 0.9},
             {"text": "one", "start": 1.2, "end": 1.4}, {"text": "thing", "start": 1.4, "end": 1.6}]
    ch = _caption_chunks(words, 3, cuts=(1.0,))
    assert ch[0] == ch[2] == (0, 3)        # "so many people." closes on the period
    assert ch[3] == (3, 4)                 # "and" is alone: the cut at 1.0 sits between it and "one"
    assert ch[4] == ch[5] == (4, 6)


def test_caption_words_need_most_of_the_word_inside_the_shot(tmp_path):
    tp = tmp_path / "t.json"
    tp.write_text(json.dumps({"words": [{"text": "are", "start": 10.0, "end": 10.4, "type": "word"},
                                        {"text": "necessarily", "start": 10.4, "end": 11.2, "type": "word"},
                                        {"text": "so", "start": 11.9, "end": 12.1, "type": "word"}]}))
    s = make(tmp_path, shots=[{"source": "main", "start": 10.0, "end": 10.45},     # 50ms of "necessarily"
                              {"source": "main", "start": 11.0, "end": 12.0}],     # most of "so", tail of "necessarily"
             transcripts={"main": "t.json"})
    tl = timeline.build(s)
    assert [w["text"] for w in tl.words] == ["are", "so"]
    assert tl.words[1]["end"] <= tl.total + 1e-6          # clipped to the shot, not running past the cut
