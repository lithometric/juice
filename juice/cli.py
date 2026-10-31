"""juice command line.

  juice transcribe <video>                    word-level transcript (cached)
  juice words <video> [--from S --to E]       word timings, for exact cut points
  juice phrases <video>                       phrase-level read of the whole transcript
  juice filmstrip <media> <start> <end>       frames + waveform PNG for a time range
  juice frames <media> <t> [<t> ...]          labeled contact sheet of frames at exact times
  juice track <video> <targets.json> --from S --to E   follow heads/objects (keyframed CSRT)
  juice tag <video> <tracks.json> -o tagged.mp4         burn name tags that follow the tracks
  juice hooks <clip> [<clip> ...] [-n 10]     rank hook candidates across clips (+ contact sheet)
  juice gallery --theme T [--bg frame.jpg]    every component in a theme on one sheet
  juice music search "<query>"                find a track
  juice music get <url> --name <name>         download it into ./music
  juice music lyrics <track>                  where each lyric line lands
  juice music offset <lyric_s> <cut_s>        offset that lands a lyric on a cut
  juice ingest <files/folders...> [-o footage]  unpack, repair and transcribe a pile of downloads (parallel)
  juice render <reel.json> [--format F] [--preview]   build a cut (every format in the spec by default)
  juice motion <script.py> [--preview] [--stills [t,...]] [--orientation O]
                                              render a motion-graphics video (juice.motion script)
  juice beds [-o dir]                         the music bed styles, with a 30 s preview of each
  juice check <reel.json> [--format F]        verify the render (lengths, loudness, cuts, text zones,
                                                   reused/unused footage, triple hook)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .media import inspect, music, transcribe
from .shortform import check as checkmod, render, spec


def _transcript_for(media: Path) -> Path:
    return transcribe.transcribe(media, media.parent / "transcripts")


def cmd_transcribe(a):
    p = _transcript_for(Path(a.media))
    print(f"{p}  ({len(transcribe.words(p))} words)")


def cmd_words(a):
    for w in transcribe.words(_transcript_for(Path(a.media)), a.start, a.end):
        print(f"{w['start']:8.2f} {w['end']:8.2f}  {w.get('speaker_id', ''):10} {w['text']}")


def cmd_phrases(a):
    for s, e, spk, txt in transcribe.phrases(_transcript_for(Path(a.media))):
        print(f"[{s:7.2f}-{e:7.2f}] {spk:10} {txt}")


def cmd_filmstrip(a):
    out = Path(a.out or f"filmstrip_{a.start:g}-{a.end:g}.png")
    print(inspect.filmstrip(Path(a.media), a.start, a.end, out, n=a.n))


def cmd_frames(a):
    print(inspect.contact_sheet(Path(a.media), a.times, Path(a.out), cols=a.cols))


def cmd_track(a):
    from .shortform import track
    tr = track.track(Path(a.media), json.loads(Path(a.targets).read_text()), a.start, a.end)
    out = Path(a.out)
    out.write_text(json.dumps(tr))
    times = a.check or [a.start + (a.end - a.start) * i / 11 for i in range(12)]
    print(out, track.debug_sheet(Path(a.media), tr, times, out.with_suffix(".jpg")))


def cmd_tag(a):
    from .shortform import track
    print(track.tag(Path(a.media), json.loads(Path(a.tracks).read_text()), Path(a.out), size=a.size))


def cmd_hooks(a):
    from .shortform import hooks
    clips = [Path(c) for c in a.clips]
    cands = hooks.candidates(clips, top=a.n)
    for i, c in enumerate(cands):
        print(f"#{i + 1:<2} {c['score']:>5}  {c['clip']:<14} {c['start']:7.2f}-{c['end']:7.2f}  {c['text']}")
        print(f"      {', '.join(c['why'])}")
    print(hooks.sheet({c.name: c for c in clips}, cands, Path(a.out)))


def cmd_gallery(a):
    from .shortform import gallery
    print(gallery.render(a.theme, Path(a.out), Path(a.bg) if a.bg else None))


def cmd_music(a):
    if a.action == "search":
        for r in music.search(a.arg):
            print(f"{r['title']}  |  {r['duration']}s  |  {r['views']} views  |  {r['url']}")
    elif a.action == "get":
        print(music.fetch(a.arg, Path(a.dir), a.name or "track"))
    elif a.action == "lyrics":
        for s, e, _, txt in music.lyrics(Path(a.arg)):
            print(f"[{s:7.2f}-{e:7.2f}] {txt}")
    elif a.action == "offset":
        print(music.offset_for(float(a.arg), float(a.cut)))


def _formats(a) -> list:
    return [a.format] if a.format else spec.formats(a.spec)


def cmd_render(a):
    for fmt in _formats(a):
        sp = spec.load(a.spec, fmt)
        report = render.render(spec.preview(sp) if a.preview else sp, keep_clips=a.keep_clips)
        print(json.dumps(report, indent=2))


def cmd_retention(a):
    from .shortform import retention, timeline as tlmod
    for fmt in _formats(a):
        sp = spec.load(a.spec, fmt)
        tl = tlmod.build(sp)
        rep = retention.score(sp, tl)
        verify = sp.edit_dir / (f"verify_{fmt}" if fmt else "verify")
        png = retention.plot(rep, verify / "retention.png", tl)
        retention.write(rep, verify / "retention.json")
        pub = retention.public(rep)
        pub.pop("curve", None)
        print(json.dumps({**pub, "plot": str(png)}, indent=2))


def cmd_check(a):
    failed = False
    for fmt in _formats(a):
        report = checkmod.check(spec.load(a.spec, fmt))
        print(json.dumps(report, indent=2))
        failed |= bool(report["problems"])
    if failed:
        sys.exit(1)


def cmd_motion(a):
    import os
    import runpy
    from .motion import Video
    if a.orientation:
        os.environ["JUICE_ORIENTATION"] = a.orientation
    script = Path(a.script).resolve()
    g = runpy.run_path(str(script), run_name="__juice__")
    videos = [x for x in g.values() if isinstance(x, Video)]
    if not videos:
        sys.exit(f"{script.name} makes no Video")
    v = videos[-1]
    tag = "_preview" if a.preview else ""
    out = Path(a.out) if a.out else script.parent / "out" / f"{script.stem}_{v.orientation}{tag}.mp4"
    if a.stills is not None:
        times = [float(x) for x in a.stills.split(",")] if a.stills else \
            sorted({round(min(p.start + 0.6, (p.start + p.end) / 2), 2) for p in v.scenes})
        print(v.stills(times, out.with_name(out.stem + "_stills.jpg")))
        return
    print(json.dumps(v.render(out, preview=a.preview), indent=2))


def cmd_beds(a):
    from .media import beds
    paths = beds.library(Path(a.out).expanduser()) if a.out else beds.library()
    for (name, st), p in zip(beds.STYLES.items(), paths):
        print(f"{name:5s} {st['bpm']} bpm  {st['about']}\n      {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="juice", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("transcribe"); p.add_argument("media"); p.set_defaults(fn=cmd_transcribe)
    p = sub.add_parser("words"); p.add_argument("media")
    p.add_argument("--from", dest="start", type=float); p.add_argument("--to", dest="end", type=float)
    p.set_defaults(fn=cmd_words)
    p = sub.add_parser("phrases"); p.add_argument("media"); p.set_defaults(fn=cmd_phrases)
    p = sub.add_parser("filmstrip"); p.add_argument("media")
    p.add_argument("start", type=float); p.add_argument("end", type=float)
    p.add_argument("-n", type=int, default=10); p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_filmstrip)
    p = sub.add_parser("frames"); p.add_argument("media"); p.add_argument("times", type=float, nargs="+")
    p.add_argument("-o", "--out", default="frames.jpg"); p.add_argument("--cols", type=int, default=4)
    p.set_defaults(fn=cmd_frames)
    p = sub.add_parser("track"); p.add_argument("media"); p.add_argument("targets")
    p.add_argument("--from", dest="start", type=float, required=True)
    p.add_argument("--to", dest="end", type=float, required=True)
    p.add_argument("-o", "--out", default="tracks.json")
    p.add_argument("--check", type=float, nargs="*", help="times for the debug sheet")
    p.set_defaults(fn=cmd_track)
    p = sub.add_parser("tag"); p.add_argument("media"); p.add_argument("tracks")
    p.add_argument("-o", "--out", default="tagged.mp4"); p.add_argument("--size", type=int, default=46)
    p.set_defaults(fn=cmd_tag)
    p = sub.add_parser("hooks"); p.add_argument("clips", nargs="+"); p.add_argument("-n", type=int, default=10)
    p.add_argument("-o", "--out", default="hooks.jpg"); p.set_defaults(fn=cmd_hooks)
    p = sub.add_parser("gallery"); p.add_argument("--theme", default="bold")
    p.add_argument("--bg"); p.add_argument("-o", "--out", default="gallery.png"); p.set_defaults(fn=cmd_gallery)
    p = sub.add_parser("music"); p.add_argument("action", choices=["search", "get", "lyrics", "offset"])
    p.add_argument("arg"); p.add_argument("cut", nargs="?", help="offset: reel time of the cut")
    p.add_argument("--name"); p.add_argument("--dir", default="music")
    p.set_defaults(fn=cmd_music)
    p = sub.add_parser("render"); p.add_argument("spec"); p.add_argument("--keep-clips", action="store_true")
    p.add_argument("--format", help="render one format (default: all formats in the spec)")
    p.add_argument("--preview", action="store_true", help="480p, half frame rate, fast encode (to *_preview.mp4)")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("motion", help="render a juice.motion script")
    p.add_argument("script"); p.add_argument("--preview", action="store_true")
    p.add_argument("--stills", nargs="?", const="", help="contact sheet at these times (default: every scene)")
    p.add_argument("--orientation", choices=["portrait", "landscape"]); p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_motion)
    p = sub.add_parser("beds", help="music bed styles + previews"); p.add_argument("-o", "--out")
    p.set_defaults(fn=cmd_beds)
    from .media import ingest
    ingest.register(sub)
    p = sub.add_parser("check"); p.add_argument("spec"); p.add_argument("--format")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("retention", help="predicted retention score + the moments that lose viewers (no render needed)")
    p.add_argument("spec"); p.add_argument("--format"); p.set_defaults(fn=cmd_retention)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except spec.SpecError as e:
        sys.exit(f"spec error: {e}")


if __name__ == "__main__":
    main()
