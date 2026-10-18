import json
from pathlib import Path

import pytest

from juice_reel import spec, theme, timeline


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
