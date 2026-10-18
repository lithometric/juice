"""juice-reel command line.

  juice-reel transcribe <video>                    word-level transcript (cached)
  juice-reel words <video> [--from S --to E]       word timings, for exact cut points
  juice-reel phrases <video>                       phrase-level read of the whole transcript
  juice-reel filmstrip <media> <start> <end>       frames + waveform PNG for a time range
  juice-reel frames <media> <t> [<t> ...]          labeled contact sheet of frames at exact times
  juice-reel track <video> <targets.json> --from S --to E   follow heads/objects (keyframed CSRT)
  juice-reel tag <video> <tracks.json> -o tagged.mp4         burn name tags that follow the tracks
  juice-reel gallery --theme T [--bg frame.jpg]    every component in a theme on one sheet
  juice-reel music search "<query>"                find a track
  juice-reel music get <url> --name <name>         download it into ./music
  juice-reel music lyrics <track>                  where each lyric line lands
  juice-reel music offset <lyric_s> <cut_s>        offset that lands a lyric on a cut
  juice-reel render <reel.json> [--format F]       build the video (every format in the spec by default)
  juice-reel check <reel.json> [--format F]        verify the render (lengths, loudness, cuts, text zones)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import inspect, music, render, spec, transcribe


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
    from . import track
    tr = track.track(Path(a.media), json.loads(Path(a.targets).read_text()), a.start, a.end)
    out = Path(a.out)
    out.write_text(json.dumps(tr))
    times = a.check or [a.start + (a.end - a.start) * i / 11 for i in range(12)]
    print(out, track.debug_sheet(Path(a.media), tr, times, out.with_suffix(".jpg")))


def cmd_tag(a):
    from . import track
    print(track.tag(Path(a.media), json.loads(Path(a.tracks).read_text()), Path(a.out), size=a.size))


def cmd_gallery(a):
    from . import gallery
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
        report = render.render(spec.load(a.spec, fmt), keep_clips=a.keep_clips)
        print(json.dumps(report, indent=2))


def cmd_check(a):
    failed = False
    for fmt in _formats(a):
        report = inspect.check(spec.load(a.spec, fmt))
        print(json.dumps(report, indent=2))
        failed |= bool(report["problems"])
    if failed:
        sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="juice-reel", description=__doc__,
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
    p = sub.add_parser("gallery"); p.add_argument("--theme", default="bold")
    p.add_argument("--bg"); p.add_argument("-o", "--out", default="gallery.png"); p.set_defaults(fn=cmd_gallery)
    p = sub.add_parser("music"); p.add_argument("action", choices=["search", "get", "lyrics", "offset"])
    p.add_argument("arg"); p.add_argument("cut", nargs="?", help="offset: reel time of the cut")
    p.add_argument("--name"); p.add_argument("--dir", default="music")
    p.set_defaults(fn=cmd_music)
    p = sub.add_parser("render"); p.add_argument("spec"); p.add_argument("--keep-clips", action="store_true")
    p.add_argument("--format", help="render one format (default: all formats in the spec)")
    p.set_defaults(fn=cmd_render)
    p = sub.add_parser("check"); p.add_argument("spec"); p.add_argument("--format")
    p.set_defaults(fn=cmd_check)

    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except spec.SpecError as e:
        sys.exit(f"spec error: {e}")


if __name__ == "__main__":
    main()
