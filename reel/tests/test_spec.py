import json
from pathlib import Path

import pytest

from juice_reel import spec, text


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


def test_platform_safe_zones(project):
    s = spec.load(write(project, base()))
    assert s.style["platform"] == "all"
    strictest = s.safe_zone
    for name in ("tiktok", "reels", "shorts"):
        z = s.style["safe_zones"][name]
        assert strictest["top"] >= z["top"] and strictest["bottom"] >= z["bottom"]
    assert spec.load(write(project, base(style={"platform": "shorts"}))).safe_zone["bottom"] == 0.18


def test_rejects_unknown_platform(project):
    with pytest.raises(spec.SpecError, match="platform"):
        spec.load(write(project, base(style={"platform": "myspace"})))


def test_piecewise_linear_expression_hits_its_points():
    from juice_reel.render import _pw_linear
    expr = _pw_linear("T", [0.0, 1.0, 2.0], [10.0, 20.0, 0.0]).replace("if(", "iff(")
    funcs = {"lt": lambda a, b: a < b, "iff": lambda c, a, b: a if c else b}
    for t, want in [(0.0, 10.0), (0.5, 15.0), (1.0, 20.0), (1.5, 10.0), (3.0, 0.0)]:
        assert eval(expr.replace("T", repr(t)), funcs) == pytest.approx(want)
