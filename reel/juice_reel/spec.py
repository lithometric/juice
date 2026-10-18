"""Load and validate a reel spec (reel.json).

A spec is the whole edit of one short-form video (TikTok, Reels, Shorts, ...) as data: which ranges of which sources, in what order,
with what text, over what music. Every edit is a change to this file followed by
a re-render, so it is the thing an AI (or a person) actually edits.

Paths inside a spec are relative to the spec file.

One spec can render several formats (e.g. vertical 9:16 and horizontal 16:9) from the same
cuts: `"formats": {"vertical": {"style": {...}}, "horizontal": {"style": {...}}}`. A shot can
override fields for one format with `"per_format": {"horizontal": {...}}`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_STYLE = Path(__file__).with_name("style.default.json")


class SpecError(ValueError):
    pass


@dataclass
class Text:
    lines: list[str]
    x: float = 0.5          # center x as a fraction of width (right edge when align="right")
    y: float = 0.58         # center y of the block as a fraction of height
    align: str = "center"   # "center" | "right" | "left" (x is the left edge)


@dataclass
class Shot:
    source: str
    start: float
    end: float
    text: Text | None = None
    punch_in: float = 0.0   # final zoom amount, e.g. 0.18 = 18% tighter by the end of the shot
    freeze_s: float = 0.0   # hold the last frame this long
    speed: float = 1.0      # 0.5 = half-speed slow motion (smooth from 60fps sources), 2 = double
    frame_x: float | list = 0.5  # crop center when the output is narrower than the source; [from, to] pans
    shake: dict | None = None    # {"at": s, "dur": 0.5, "amp": 0.025}: decaying camera shake
    zoom: float | list = 1.0     # 1.6 = static close-up; [1.0, 1.3] = push in, [1.4, 1.0] = pull out (eased)
    follow: str | None = None    # keep a tracked target (see "tracks") centered through crop and zoom
    follow_y: float = 2.0        # follow point = head center + this many head-heights down (upper body)
    replay: bool = False         # a flash-forward/hook: shown, but source events don't fire here
    blur: float = 0.0            # background blur (sigma at 1080p) under title cards / end cards
    fx: list = field(default_factory=list)  # overlays / sound effects, see components.py and sfx.py

    @property
    def duration(self) -> float:
        return (self.end - self.start) / self.speed + self.freeze_s


@dataclass
class Music:
    file: Path
    offset: float = 0.0     # where in the song the reel starts, in seconds


@dataclass
class Spec:
    path: Path
    sources: dict[str, Path]
    shots: list[Shot]
    output: Path
    music: Music | None = None
    original_audio: bool = False
    style: dict = field(default_factory=dict)
    format: str | None = None
    tracks: dict | None = None   # output of `juice-reel track`; shots can "follow" its targets
    theme: dict = field(default_factory=dict)       # resolved theme (see theme.py)
    overlays: list = field(default_factory=list)    # layers on the output timeline (see timeline.py)
    events: list = field(default_factory=list)      # moments in the source that drive fx/sfx/meters
    meters: dict = field(default_factory=dict)      # named values for HUD meters: {"gpt": 100}
    transcripts: dict = field(default_factory=dict) # source key -> word-level transcript path (captions)

    @property
    def edit_dir(self) -> Path:
        return self.output.parent

    @property
    def safe_zone(self) -> dict:
        """UI-free area for the target platform ("all" = strictest of TikTok, Reels and Shorts)."""
        return self.style["safe_zones"][self.style["platform"]]

    @property
    def fps(self) -> int:
        return self.style["fps"]

    def shot_frames(self, shot: Shot) -> int:
        """Every shot is rounded to whole frames; audio is cut to match."""
        return round(shot.duration * self.fps)

    @property
    def duration(self) -> float:
        return sum(self.shot_frames(s) for s in self.shots) / self.fps

    def cut_points(self) -> list[float]:
        """Output-timeline times where one shot hands off to the next (frame-exact)."""
        n, out = 0, []
        for s in self.shots[:-1]:
            n += self.shot_frames(s)
            out.append(round(n / self.fps, 3))
        return out


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_style(spec_dir: Path, overrides: dict | None) -> dict:
    style = json.loads(DEFAULT_STYLE.read_text())
    # A project-level style.json next to the spec carries the creator's taste across reels.
    project_style = spec_dir / "style.json"
    if project_style.exists():
        style = _deep_merge(style, json.loads(project_style.read_text()))
    return _deep_merge(style, overrides or {})


def formats(path: str | Path) -> list[str | None]:
    """Format names a spec renders to ([None] for a single-format spec)."""
    raw = json.loads(Path(path).read_text())
    return list(raw.get("formats", {})) or [None]


def load(path: str | Path, fmt: str | None = None) -> Spec:
    from . import components, sfx  # local: keeps spec importable without numpy for simple use
    from . import theme as theme_mod

    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    base = path.parent
    fmts = raw.get("formats", {})
    if fmt is None and fmts:
        fmt = next(iter(fmts))
    if fmt is not None and fmt not in fmts:
        raise SpecError(f"unknown format {fmt!r} (spec has: {', '.join(fmts) or 'none'})")

    sources = {k: (base / v).resolve() for k, v in raw.get("sources", {}).items()}
    if not sources:
        raise SpecError("spec has no sources")
    for k, p in sources.items():
        if not p.exists():
            raise SpecError(f"source {k!r} not found: {p}")

    shots = []
    for i, s in enumerate(raw.get("shots", [])):
        s = {**s, **s.get("per_format", {}).get(fmt, {})} if fmt else s
        if s["source"] not in sources:
            raise SpecError(f"shot {i}: unknown source {s['source']!r}")
        if s["end"] <= s["start"]:
            raise SpecError(f"shot {i}: end ({s['end']}) must be after start ({s['start']})")
        text = None
        if s.get("text"):
            t = s["text"]
            text = Text(lines=t["lines"], x=t.get("x", 0.5), y=t.get("y", 0.58), align=t.get("align", "center"))
            if text.align not in ("center", "right", "left"):
                raise SpecError(f"shot {i}: align must be center, right or left")
        speed = float(s.get("speed", 1))
        if speed <= 0:
            raise SpecError(f"shot {i}: speed must be positive")
        for f in s.get("fx", []):
            kind = f.get("type")
            if kind not in components.COMPONENTS and kind != "sfx":
                raise SpecError(f"shot {i}: unknown fx type {kind!r}")
            sound = f.get("sound") if kind == "sfx" else f.get("sfx")
            if sound and sound not in sfx.SOUNDS:
                raise SpecError(f"shot {i}: unknown sound {sound!r} (have: {', '.join(sfx.SOUNDS)})")
        shots.append(Shot(source=s["source"], start=float(s["start"]), end=float(s["end"]), text=text,
                          punch_in=float(s.get("punch_in", 0)), freeze_s=float(s.get("freeze_s", 0)),
                          speed=speed, frame_x=s.get("frame_x", 0.5), shake=s.get("shake"),
                          fx=list(s.get("fx", [])), zoom=s.get("zoom", 1.0), follow=s.get("follow"),
                          follow_y=float(s.get("follow_y", 2.0)), replay=bool(s.get("replay", False)),
                          blur=float(s.get("blur", 0))))
    if not shots:
        raise SpecError("spec has no shots")

    music = None
    if raw.get("music"):
        m = raw["music"]
        music = Music(file=(base / m["file"]).resolve(), offset=float(m.get("offset", 0)))
        if not music.file.exists():
            raise SpecError(f"music file not found: {music.file}")

    style = load_style(base, raw.get("style"))
    if fmt:
        style = _deep_merge(style, fmts[fmt].get("style", {}))
    platforms = [k for k in style["safe_zones"] if not k.startswith("_")]
    if style["platform"] not in platforms:
        raise SpecError(f"platform must be one of {', '.join(platforms)}")

    tracks = None
    if raw.get("tracks"):
        tracks = json.loads((base / raw["tracks"]).read_text())
    for i, sh in enumerate(shots):
        if sh.follow and (not tracks or sh.follow not in tracks["targets"]):
            raise SpecError(f"shot {i}: follow target {sh.follow!r} not in tracks")

    output = (base / raw.get("output", "edit/final.mp4")).resolve()
    if fmt:
        output = output.with_name(f"{output.stem}_{fmt}{output.suffix}")

    return Spec(
        path=path,
        sources=sources,
        shots=shots,
        output=output,
        format=fmt,
        tracks=tracks,
        theme=theme_mod.load(raw.get("theme"), base),
        overlays=list(raw.get("overlays", [])),
        events=list(raw.get("events", [])),
        meters=dict(raw.get("meters", {})),
        transcripts={k: str((base / v).resolve()) for k, v in raw.get("transcripts", {}).items()},
        music=music,
        original_audio=bool(raw.get("original_audio", False)),
        style=style,
    )
