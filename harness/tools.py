"""The tool table: every juice command as a function the model can call.

Each tool is (name, description, JSON schema, fn). fn(ws, **args) -> Result(text, images). The
descriptions are the model's manual, so they say when to use a tool, not just what it does. The
same table serves the OpenRouter loop (agent.py) and the MCP server (mcp_server.py).
"""
from __future__ import annotations

import inspect as pyinspect
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from juice import brand as brand_mod
from juice.media import inspect as look
from juice.media import music, transcribe
from juice.shortform import spec as spec_mod

from .workspace import Result, Workspace

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., Result]


TOOLS: list[Tool] = []


def _schema(props: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def tool(name: str, description: str, props: dict | None = None, required: list[str] | None = None):
    def deco(fn):
        TOOLS.append(Tool(name, description.strip(), _schema(props or {}, required or []), fn))
        return fn
    return deco


def by_name(name: str) -> Tool:
    for t in TOOLS:
        if t.name == name:
            return t
    raise KeyError(name)


def _s(desc: str, **kw) -> dict:
    return {"type": "string", "description": desc, **kw}


def _n(desc: str) -> dict:
    return {"type": "number", "description": desc}


def _b(desc: str) -> dict:
    return {"type": "boolean", "description": desc}


def _j(x) -> str:
    return json.dumps(x, indent=1, default=str)


def _transcript(ws: Workspace, media: str) -> Path:
    m = ws.media(media)
    return transcribe.transcribe(m, m.parent / "transcripts")


# --- footage in -------------------------------------------------------------------------------

@tool("ingest", """
Turn a pile of downloads into usable footage: unpacks archives (zips, stalled Chrome *.crdownload), repairs
recordings with a missing moov, transcribes speech (word-level, cached) and writes <out>/manifest.json,
<out>/contact.jpg and <out>/INGEST.md. Call this first on whatever the person gave you, then `footage`.
Inputs can be anywhere on disk (e.g. ~/Downloads/shoot); the output folder is inside the project.""",
      {"inputs": {"type": "array", "items": _s("file or folder"), "description": "files or folders to ingest"},
       "out": _s("output folder inside the project", default="footage"),
       "transcribe": _b("transcribe speech (default true)")},
      ["inputs"])
def t_ingest(ws: Workspace, inputs: list[str], out: str = "footage", transcribe: bool = True) -> Result:
    from juice.media import ingest
    paths = [ws.media(p) for p in inputs]
    res = ingest.ingest(paths, ws.inside(out), transcribe=transcribe, log=lambda *a: print(*a, file=sys.stderr))
    sheet = ws.inside(out) / "contact.jpg"
    return Result(ingest.summary(res), [sheet] if sheet.exists() else [])


@tool("footage", """
What footage the project has: the ingest manifest (name, path, duration, size, first words, status) if
`ingest` ran, else every media file in the folder with its duration. Start here to know what you can cut.""",
      {"dir": _s("folder to list", default="footage")})
def t_footage(ws: Workspace, dir: str = "footage") -> Result:
    d = ws.path(dir)
    man = d / "manifest.json"
    if man.exists():
        m = json.loads(man.read_text())
        clips = [{k: v for k, v in c.items() if k not in ("thumbs",) and not k.startswith("_")} for c in m.get("clips", [])]
        head = {k: v for k, v in m.items() if k != "clips"}
        imgs = [d / "contact.jpg"] if (d / "contact.jpg").exists() else []
        return Result(_j({**head, "clips": clips}), imgs)
    if not d.exists():
        return Result(f"{dir} does not exist (project: {ws.root}). Run `ingest` on the footage first, or `list_files`.")
    rows = []
    for p in sorted(d.rglob("*")):
        if p.suffix.lower() in (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".mp3", ".wav", ".m4a", ".aac"):
            try:
                rows.append(f"{ws.rel(p):<60} {look.probe_duration(p):7.2f}s")
            except Exception as e:  # unreadable file
                rows.append(f"{ws.rel(p):<60} unreadable ({e})")
    return Result("\n".join(rows) or f"no media files under {dir}")


@tool("probe", "Stream facts about one media file (duration, width, height, fps, audio), via ffprobe.",
      {"media": _s("path to a video or audio file")}, ["media"])
def t_probe(ws: Workspace, media: str) -> Result:
    import subprocess
    m = ws.media(media)
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(m)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    streams = [{k: s.get(k) for k in ("codec_type", "codec_name", "width", "height", "r_frame_rate", "sample_rate",
                                       "channels", "duration")} for s in d.get("streams", [])]
    return Result(_j({"file": ws.rel(m), "duration": d.get("format", {}).get("duration"), "streams": streams}))


# --- what is said ------------------------------------------------------------------------------

@tool("transcribe", "Transcribe a file (word-level, cached next to it in transcripts/). `phrases` and `words` call it for you.",
      {"media": _s("video or audio file")}, ["media"])
def t_transcribe(ws: Workspace, media: str) -> Result:
    p = _transcript(ws, media)
    return Result(f"{ws.rel(p)}  ({len(transcribe.words(p))} words)")


@tool("phrases", """
Phrase-level read of everything said in a file: [start-end] speaker text. The fastest way to know the story
in a clip and roughly where each line is; then `words` for exact cut points.""",
      {"media": _s("video or audio file")}, ["media"])
def t_phrases(ws: Workspace, media: str) -> Result:
    rows = [f"[{s:7.2f}-{e:7.2f}] {spk:10} {txt}" for s, e, spk, txt in transcribe.phrases(_transcript(ws, media))]
    return Result("\n".join(rows) or "(no speech)")


@tool("words", """
Word timings in a range: start end speaker word. Use it to cut exactly: a talking shot starts at
first_word.start - 0.08 and ends right after the last word.""",
      {"media": _s("video or audio file"), "start": _n("from (s)"), "end": _n("to (s)")}, ["media"])
def t_words(ws: Workspace, media: str, start: float | None = None, end: float | None = None) -> Result:
    rows = [f"{w['start']:8.2f} {w['end']:8.2f}  {w.get('speaker_id', ''):10} {w['text']}"
            for w in transcribe.words(_transcript(ws, media), start, end)]
    return Result("\n".join(rows) or "(no words in that range)")


# --- what is seen ------------------------------------------------------------------------------

@tool("filmstrip", """
Look at a time range: N frames evenly across it plus the audio waveform, as one image. Use it to see who is
where, when the camera moves, where the energy is. Four windows across a clip is a good first pass.""",
      {"media": _s("video file"), "start": _n("from (s)"), "end": _n("to (s)"),
       "n": {"type": "integer", "description": "frames (default 10)"}}, ["media", "start", "end"])
def t_filmstrip(ws: Workspace, media: str, start: float, end: float, n: int = 10) -> Result:
    m = ws.media(media)
    out = ws.look / f"filmstrip_{m.stem}_{start:g}-{end:g}.png"
    look.filmstrip(m, start, end, out, n=n)
    return Result(f"filmstrip of {ws.rel(m)} {start:g}-{end:g}s -> {ws.rel(out)} (image attached)", [out])


@tool("frames", """
Frames at exact times on one labeled contact sheet. Use it to find the exact frame something happens (a hit,
a reaction, a turn) by narrowing the times each call, and to confirm what a cut lands on.""",
      {"media": _s("video file"), "times": {"type": "array", "items": _n("time (s)"), "description": "times in seconds"},
       "width": {"type": "integer", "description": "frame width px (default 480)"}}, ["media", "times"])
def t_frames(ws: Workspace, media: str, times: list[float], width: int = 480) -> Result:
    m = ws.media(media)
    out = ws.look / f"frames_{m.stem}_{int(time.time())}.jpg"
    look.contact_sheet(m, [float(t) for t in times], out, width=width)
    return Result(f"frames of {ws.rel(m)} at {', '.join(f'{t:g}' for t in times)} -> {ws.rel(out)} (image attached)", [out])


@tool("hooks", """
Rank hook candidates across clips (lines that stand alone, are short, get a laugh or reaction right after,
have energy and a face on screen; interviewer lines pushed down) with a contact sheet. A shortlist: pick by
reading the lines, since the most wrong-sounding wording is a judgment call the score can't make.""",
      {"clips": {"type": "array", "items": _s("video file"), "description": "the clips"},
       "n": {"type": "integer", "description": "how many (default 10)"}}, ["clips"])
def t_hooks(ws: Workspace, clips: list[str], n: int = 10) -> Result:
    from juice.shortform import hooks
    paths = [ws.media(c) for c in clips]
    cands = hooks.candidates(paths, top=n)
    rows = [f"#{i + 1:<2} {c['score']:>5}  {c['clip']:<14} {c['start']:7.2f}-{c['end']:7.2f}  {c['text']}\n      {', '.join(c['why'])}"
            for i, c in enumerate(cands)]
    out = ws.look / "hooks.jpg"
    hooks.sheet({p.name: p for p in paths}, cands, out)
    return Result("\n".join(rows) or "(no candidates)", [out] if out.exists() else [])


# --- brand, look -------------------------------------------------------------------------------

@tool("brand", """
The brand kit (juice.json): colors, fonts, logo, line, link. With no argument, looks for juice.json in the
project. A domain fetches https://<domain>/.well-known/juice.json. Cuts and motion scenes read the kit
automatically when it sits next to the spec; this tool is for knowing what's in it.""",
      {"value": _s("path, URL or domain (optional)")})
def t_brand(ws: Workspace, value: str | None = None) -> Result:
    kit = brand_mod.find(value, ws.root)
    if kit is None:
        return Result("no juice.json in the project. Ask for one, or build it with `npx juice-press <site> <repo>`.")
    s = _j(kit)
    return Result(s if len(s) < 6000 else s[:6000] + "\n…")


@tool("gallery", """
Every component of a theme on one sheet (title, chip, captions, slam, meters, end card, ...), optionally over a
real frame. Look at it before choosing a look; themes: bold, clean, loud, or a theme JSON path.""",
      {"theme": _s("theme name or file", default="bold"), "bg": _s("a frame to draw over (optional)")})
def t_gallery(ws: Workspace, theme: str = "bold", bg: str | None = None) -> Result:
    from juice.shortform import gallery
    out = ws.look / f"gallery_{Path(theme).stem}.png"
    gallery.render(theme, out, ws.media(bg) if bg else None)
    return Result(f"gallery of {theme} -> {ws.rel(out)} (image attached)", [out])


# --- music -------------------------------------------------------------------------------------

@tool("music_search", "Find a track to download (title, duration, views, url).", {"query": _s("search text")}, ["query"])
def t_music_search(ws: Workspace, query: str) -> Result:
    rows = [f"{r['title']}  |  {r['duration']}s  |  {r['views']} views  |  {r['url']}" for r in music.search(query)]
    return Result("\n".join(rows) or "nothing found")


@tool("music_get", "Download a track into music/ (yt-dlp). If an upload 403s, try another upload of the same song.",
      {"url": _s("track url"), "name": _s("file name without extension")}, ["url", "name"])
def t_music_get(ws: Workspace, url: str, name: str) -> Result:
    return Result(ws.rel(music.fetch(url, ws.inside("music"), name)))


@tool("music_lyrics", "Where each lyric line lands in a track, to find the clean, recognizable part and the vocal entrance.",
      {"track": _s("audio file")}, ["track"])
def t_music_lyrics(ws: Workspace, track: str) -> Result:
    rows = [f"[{s:7.2f}-{e:7.2f}] {txt}" for s, e, _, txt in music.lyrics(ws.media(track))]
    return Result("\n".join(rows) or "(no lyrics found; align by the waveform with filmstrip)")


@tool("music_offset", "The music.offset that lands a lyric (at lyric_s in the track) on a cut (at cut_s in the edit).",
      {"lyric_s": _n("time in the track"), "cut_s": _n("time in the edit")}, ["lyric_s", "cut_s"])
def t_music_offset(ws: Workspace, lyric_s: float, cut_s: float) -> Result:
    return Result(f"{music.offset_for(float(lyric_s), float(cut_s)):.2f}")


@tool("beds", "The generated music bed styles for motion videos (name, bpm, feel): nothing to license.")
def t_beds(ws: Workspace) -> Result:
    from juice.media import beds
    return Result("\n".join(f"{name:8s} {st['bpm']} bpm  {st['about']}" for name, st in beds.STYLES.items()))


# --- files -------------------------------------------------------------------------------------

@tool("read_file", "Read a text file in the project (reel.json, a motion script, edit/project.md, footage/INGEST.md, a transcript).",
      {"path": _s("path")}, ["path"])
def t_read(ws: Workspace, path: str) -> Result:
    p = ws.media(path)
    s = p.read_text()
    return Result(s if len(s) < 60000 else s[:60000] + "\n…(truncated)")


@tool("write_file", """
Write a text file in the project: reel.json (the cut), a juice.motion script (.py), style.json, or edit/project.md
(your decisions, one short section per round). Overwrites. Every edit to a video is a change to its file
followed by `render` (or `motion`).""",
      {"path": _s("path inside the project"), "content": _s("the whole file")}, ["path", "content"])
def t_write(ws: Workspace, path: str, content: str) -> Result:
    p = ws.inside(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == ".json":
        json.loads(content)  # fail loudly before writing a broken spec
    p.write_text(content)
    return Result(f"wrote {ws.rel(p)} ({len(content)} chars)")


@tool("list_files", "List files in a project folder (recursive, media sizes shown).", {"dir": _s("folder", default=".")})
def t_list(ws: Workspace, dir: str = ".") -> Result:
    d = ws.path(dir)
    rows = []
    for p in sorted(d.rglob("*")):
        if any(part.startswith(".") or part in ("__pycache__", "agent") for part in p.relative_to(d).parts):
            continue
        if p.is_file():
            rows.append(f"{ws.rel(p):<70} {p.stat().st_size / 1e6:8.1f} MB")
    return Result("\n".join(rows[:400]) or "(empty)")


# --- cuts: render, check -----------------------------------------------------------------------

def _spec(ws: Workspace, path: str, fmt: str | None):
    return spec_mod.load(ws.media(path), fmt)


@tool("render", """
Render a cut from its reel.json. preview=true (default) renders at 480p, half frame rate, in seconds, to
edit/final_preview.mp4 and attaches a filmstrip of the result; use it while iterating. preview=false renders full
quality to the spec's output (edit/final.mp4) for every format in the spec; follow it with `check`.""",
      {"spec": _s("the spec", default="reel.json"), "preview": _b("fast preview (default true)"),
       "format": _s("one format name when the spec has several (optional)")})
def t_render(ws: Workspace, spec: str = "reel.json", preview: bool = True, format: str | None = None) -> Result:
    from juice.shortform import render
    reports, images = [], []
    for fmt in ([format] if format else spec_mod.formats(ws.media(spec))):
        sp = _spec(ws, spec, fmt)
        if preview:
            sp = spec_mod.preview(sp)
        rep = render.render(sp)
        reports.append(rep)
        if preview:
            out = Path(rep.get("output", sp.output))
            img = ws.look / f"render_{out.stem}.png"
            dur = look.probe_duration(out)
            look.filmstrip(out, 0, max(dur - 0.05, 0.1), img, n=min(16, max(4, int(dur * 1.5))))
            images.append(img)
    return Result(_j(reports if len(reports) > 1 else reports[0]) + ("\n(filmstrip of the preview attached)" if images else ""), images)


@tool("check", """
Verify a full render against its spec: lengths, loudness, text inside the safe zones, footage reused or unused,
the triple hook, the retention score. Attaches cuts.jpg (the frame either side of every cut), overview.png and
retention.png. `problems` must be empty before you show the person anything.""",
      {"spec": _s("the spec", default="reel.json"), "format": _s("one format (optional)")})
def t_check(ws: Workspace, spec: str = "reel.json", format: str | None = None) -> Result:
    from juice.shortform import check
    reports, images = [], []
    for fmt in ([format] if format else spec_mod.formats(ws.media(spec))):
        rep = check.check(_spec(ws, spec, fmt))
        reports.append(rep)
        images += [Path(p) for p in rep.get("look_at", []) if Path(p).exists()]
    return Result(_j(reports if len(reports) > 1 else reports[0]), images)


@tool("retention", """
Predicted retention of a cut without rendering: a 0-100 score, 3 s / end retention, and the moments that lose
the most viewers and why. Score before rendering, fix the top peaks, re-score.""",
      {"spec": _s("the spec", default="reel.json"), "format": _s("one format (optional)")})
def t_retention(ws: Workspace, spec: str = "reel.json", format: str | None = None) -> Result:
    from juice.shortform import retention, timeline as tlmod
    outs, images = [], []
    for fmt in ([format] if format else spec_mod.formats(ws.media(spec))):
        sp = _spec(ws, spec, fmt)
        tl = tlmod.build(sp)
        rep = retention.score(sp, tl)
        verify = sp.edit_dir / (f"verify_{fmt}" if fmt else "verify")
        png = retention.plot(rep, verify / "retention.png", tl)
        retention.write(rep, verify / "retention.json")
        pub = retention.public(rep)
        pub.pop("curve", None)
        outs.append(pub)
        images.append(png)
    return Result(_j(outs if len(outs) > 1 else outs[0]), images)


# --- tracking ----------------------------------------------------------------------------------

@tool("track", """
Follow heads/objects through a range (keyframed CSRT) and write tracks.json plus a debug sheet. targets:
{"name": {"color": [r,g,b], "keys": {"<t>": [x, y, w, h], ...}, "hide": [[a, b], ...]}} with boxes as fractions
of the frame. Add keys every ~1 s through fast motion; inspect the sheet; add keys where it drifts.""",
      {"media": _s("video file"), "targets": {"type": "object", "description": "targets", "additionalProperties": True},
       "start": _n("from (s)"), "end": _n("to (s)"), "out": _s("tracks file", default="tracks.json")},
      ["media", "targets", "start", "end"])
def t_track(ws: Workspace, media: str, targets: dict, start: float, end: float, out: str = "tracks.json") -> Result:
    from juice.shortform import track
    m = ws.media(media)
    tr = track.track(m, targets, start, end)
    o = ws.inside(out)
    o.write_text(json.dumps(tr))
    times = [start + (end - start) * i / 11 for i in range(12)]
    sheet = track.debug_sheet(m, tr, times, o.with_suffix(".jpg"))
    return Result(f"wrote {ws.rel(o)}; debug sheet attached", [sheet])


@tool("tag", "Burn name tags that follow the tracks into a copy of the video (keeps timestamps), to use as a shot source.",
      {"media": _s("video file"), "tracks": _s("tracks.json"), "out": _s("output video", default="tagged.mp4"),
       "size": {"type": "integer", "description": "tag size px (default 46)"}}, ["media", "tracks"])
def t_tag(ws: Workspace, media: str, tracks: str, out: str = "tagged.mp4", size: int = 46) -> Result:
    from juice.shortform import track
    p = track.tag(ws.media(media), json.loads(ws.media(tracks).read_text()), ws.inside(out), size=size)
    return Result(ws.rel(p))


# --- motion videos -----------------------------------------------------------------------------

@tool("motion", """
Render a juice.motion script (a .py that builds a Video). mode: "stills" (one frame per scene on a contact
sheet, seconds: look every round), "preview" (480p, every other frame) or "render" (full quality, split across
cores). Output goes to out/<script>_<orientation>[_preview].mp4 next to the script. Set orientation to render
the same script portrait or landscape.""",
      {"script": _s("the script"), "mode": _s("stills | preview | render", enum=["stills", "preview", "render"]),
       "orientation": _s("portrait | landscape (optional)", enum=["portrait", "landscape"]),
       "times": {"type": "array", "items": _n("time (s)"), "description": "stills at these output times (optional)"}},
      ["script"])
def t_motion(ws: Workspace, script: str, mode: str = "stills", orientation: str | None = None,
             times: list[float] | None = None) -> Result:
    import runpy
    from juice.motion import Video
    path = ws.media(script)
    old = os.environ.get("JUICE_ORIENTATION")
    if orientation:
        os.environ["JUICE_ORIENTATION"] = orientation
    try:
        g = runpy.run_path(str(path), run_name="__juice__")
    finally:
        if orientation:
            os.environ.pop("JUICE_ORIENTATION", None)
            if old:
                os.environ["JUICE_ORIENTATION"] = old
    videos = [x for x in g.values() if isinstance(x, Video)]
    if not videos:
        return Result(f"{path.name} makes no Video")
    v = videos[-1]
    out = path.parent / "out" / f"{path.stem}_{v.orientation}{'_preview' if mode == 'preview' else ''}.mp4"
    if mode == "stills":
        ts = [float(t) for t in times] if times else \
            sorted({round(min(p.start + 0.6, (p.start + p.end) / 2), 2) for p in v.scenes})
        sheet = v.stills(ts, out.with_name(out.stem + "_stills.jpg"))
        return Result(f"stills of {path.name} ({v.orientation}) at {', '.join(f'{t:g}' for t in ts)} -> {ws.rel(sheet)}", [Path(sheet)])
    rep = v.render(out, preview=(mode == "preview"))
    return Result(_j(rep))


# --- docs --------------------------------------------------------------------------------------

def _section(md: str, title: str) -> str:
    lines, keep, out = md.splitlines(), False, []
    for ln in lines:
        if ln.startswith("## "):
            keep = ln[3:].strip().lower().startswith(title.lower())
        if keep:
            out.append(ln)
    return "\n".join(out)


def _pieces(mod) -> str:
    from juice.motion.scenes import Piece
    rows = [mod.__doc__ or ""]
    for name, cls in vars(mod).items():
        if pyinspect.isclass(cls) and issubclass(cls, Piece) and cls is not Piece and cls.__module__ == mod.__name__:
            try:
                sig = str(pyinspect.signature(cls))
            except (TypeError, ValueError):
                sig = "(...)"
            rows.append(f"{name}{sig}\n    {(cls.__doc__ or '').strip()}")
    return "\n\n".join(rows)


@tool("docs", """
Reference text. topics: "guide" (how to edit with juice: hooks, retention, cutting, text and music rules, genre
recipes), "spec" (every reel.json field), "components" (themes and the overlay components with their params),
"motion" (the Video API, every scene and overlay of the loud look and the clean look, with signatures).""",
      {"topic": _s("guide | spec | components | motion", enum=["guide", "spec", "components", "motion"])}, ["topic"])
def t_docs(ws: Workspace, topic: str) -> Result:
    readme = (REPO / "README.md").read_text() if (REPO / "README.md").exists() else ""
    if topic == "guide":
        return Result((REPO / "CLAUDE.md").read_text() if (REPO / "CLAUDE.md").exists() else "(guide missing)")
    if topic == "spec":
        return Result(_section(readme, "A reel"))
    if topic == "components":
        from juice.shortform.components import COMPONENTS
        return Result(_section(readme, "Themes and components") + "\n\ncomponents: " + ", ".join(sorted(COMPONENTS)))
    if topic == "motion":
        from juice.motion import clean, scenes, video
        return Result("\n\n".join([_section(readme, "Motion videos"), video.__doc__ or "",
                                   "# scenes (loud look)\n" + _pieces(scenes), "# clean look\n" + _pieces(clean)]))
    return Result(f"unknown topic {topic}")


# --- the OpenAI-style function list ----------------------------------------------------------------

def openai_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in TOOLS]


def run(ws: Workspace, name: str, args: dict) -> Result:
    """Run one tool; errors come back as text so the model can recover."""
    try:
        t = by_name(name)
    except KeyError:
        return Result(f"error: no tool named {name}")
    try:
        return t.fn(ws, **args)
    except TypeError as e:
        return Result(f"error: bad arguments for {name}: {e}")
    except SystemExit as e:
        return Result(f"error: {e}")
    except Exception as e:  # the model sees the failure and tries something else
        return Result(f"error: {type(e).__name__}: {e}")
