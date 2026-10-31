import json
from pathlib import Path

import pytest

from juice.shortform import spec, text


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "clip.mov").write_bytes(b"")
    return tmp_path


def write(project: Path, body: dict) -> Path:
    p = project / "reel.json"
    p.write_text(json.dumps(body))
    return p


def base(**over):
    body = {"sources": {"main": "clip.mov"},
            "shots": [{"source": "main", "start": 23.12, "end": 26.00},
                      {"source": "main", "start": 55.64, "end": 58.60},
                      {"source": "main", "start": 63.80, "end": 66.10, "punch_in": 0.18}]}
    body.update(over)
    return body


def test_cut_points_are_frame_exact(project):
    s = spec.load(write(project, base()))
    # 2.88s -> 86 frames, 2.96s -> 89 frames at 30fps
    assert [s.shot_frames(x) for x in s.shots] == [86, 89, 69]
    assert s.cut_points() == [round(86 / 30, 3), round(175 / 30, 3)]
    assert s.duration == pytest.approx(244 / 30)


def test_freeze_counts_toward_length(project):
    body = base()
    body["shots"][-1]["freeze_s"] = 1.0
    s = spec.load(write(project, body))
    assert s.shot_frames(s.shots[-1]) == 99


def test_rejects_backwards_shot(project):
    body = base()
    body["shots"][0]["end"] = 20.0
    with pytest.raises(spec.SpecError, match="end"):
        spec.load(write(project, body))


def test_rejects_unknown_source(project):
    body = base()
    body["shots"][0]["source"] = "nope"
    with pytest.raises(spec.SpecError, match="unknown source"):
        spec.load(write(project, body))


def test_project_style_overrides_defaults(project):
    (project / "style.json").write_text(json.dumps({"font": {"size": 60}}))
    s = spec.load(write(project, base(style={"fps": 24})))
    assert s.style["font"]["size"] == 60
    assert s.style["font"]["index"] == 1      # deep merge keeps the rest of the font block
    assert s.style["fps"] == 24


def test_right_aligned_text_ends_at_x(project):
    s = spec.load(write(project, base()))
    t = spec.Text(lines=["can we pull her", "SEO analytics?"], x=0.95, y=0.15, align="right")
    left, top, right, bottom = text.bounds(t, s.style)
    assert right == pytest.approx(0.95)
    assert left < right and top < 0.15 < bottom


def test_two_kinds_of_video_each_short_or_long(project):
    s = spec.load(write(project, base()))
    assert (s.orientation, s.length) == ("portrait", "short")
    assert (s.style["width"], s.style["height"]) == (1080, 1920)
    assert s.safe_zone["buttons"]  # portrait keeps clear of the apps' action buttons
    s = spec.load(write(project, base(orientation="landscape", length="long")))
    assert (s.style["width"], s.style["height"]) == (1920, 1080)
    assert s.safe_zone["buttons"] is None
    assert s.style["hook"]["window"] > 1.5  # long form gets a longer hook window


def test_formats_are_orientations(project):
    body = base(formats={"portrait": {}, "horizontal": {}})  # old name still loads
    assert spec.formats(write(project, body)) == ["portrait", "landscape"]
    s = spec.load(write(project, body), "landscape")
    assert s.orientation == "landscape" and s.output.name == "final_landscape.mp4"
    with pytest.raises(spec.SpecError, match="portrait or landscape"):
        spec.load(write(project, base(formats={"square": {}})))
    with pytest.raises(spec.SpecError, match="short or long"):
        spec.load(write(project, base(length="medium")))


def test_brand_kit_becomes_the_theme(project):
    (project / "juice.json").write_text(json.dumps({
        "juice": "1", "name": "Acme", "colors": {"primary": "#F2C21B", "background": "#06080A"},
        "fonts": {"display": {"family": "JetBrains Mono", "weight": 900}}, "logo": {"mark": "mark.svg"}}))
    s = spec.load(write(project, base()))
    assert s.brand["name"] == "Acme" and s.brand["logo"]["mark"] == str(project / "mark.svg")
    assert s.theme["colors"]["accent"] == [242, 194, 27] and s.theme["colors"]["ink"] == [6, 8, 10]
    from juice.shortform import theme
    assert theme.font(s.theme["fonts"]["display"]["family"], 40).getname()[0] != "JetBrains Mono"


def test_piecewise_linear_expression_hits_its_points():
    from juice.shortform.render import _pw_linear
    expr = _pw_linear("T", [0.0, 1.0, 2.0], [10.0, 20.0, 0.0]).replace("if(", "iff(")
    funcs = {"lt": lambda a, b: a < b, "iff": lambda c, a, b: a if c else b}
    for t, want in [(0.0, 10.0), (0.5, 15.0), (1.0, 20.0), (1.5, 10.0), (3.0, 0.0)]:
        assert eval(expr.replace("T", repr(t)), funcs) == pytest.approx(want)


def test_check_flags_duplicate_footage_and_unused_clips(project):
    from juice.shortform.check import footage_problems
    (project / "other.mov").write_bytes(b"")
    s = spec.load(write(project, {"sources": {"main": "clip.mov", "spare": "other.mov"},
                                  "shots": [{"source": "main", "start": 1.0, "end": 3.0},
                                            {"source": "main", "start": 2.5, "end": 4.0},
                                            {"source": "main", "start": 1.0, "end": 2.0, "replay": True}]}))
    probs = footage_problems(s)
    assert any("reuse the same footage" in p and "shots 0 and 1" in p for p in probs)
    assert not any("shots 0 and 2" in p or "shots 1 and 2" in p for p in probs)   # replay is allowed
    assert any("not used: spare" in p for p in probs)


def test_monospace_fonts_are_never_used():
    from juice.shortform import theme
    for name in theme.names():
        for role, f in theme.load(name)["fonts"].items():
            assert not theme.is_mono(f["family"]), (name, role, f["family"])
    assert theme.is_mono("JetBrains Mono") and theme.is_mono("~/fonts/FiraCode-Bold.ttf")
    assert theme.font("JetBrains Mono", 40).getname() == theme.font("Inter", 40).getname()
