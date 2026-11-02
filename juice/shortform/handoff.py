"""Hand an edit to DaVinci Resolve (or anything that reads OpenTimelineIO) to finish by hand.

    juice export reel.json        # edit/resolve/: cut.otio, angles_<group>.otio, captions.srt, sfx/, README.md

- cut.otio: juice's cut. V1 is every shot from its own file (a multicam shot from its angle's file, drift
  taken out at each cut), V2 the graphics on transparent (edit/overlays.mov, `juice overlays`), A1 the
  shots' sound, then the music and each sound effect on their own tracks. A timeline marker at every cut
  says which angle and why (a shot's "note").
- angles_<group>.otio, one per multicam group: each camera on its own video track, every file placed where
  `juice sync` put it, the group's sound on A1 and each camera's own sound under it. In Resolve's Media Pool,
  right-click that timeline > Convert Compound Clips (Timelines) to Multicam Clips: a real multicam clip,
  synced, for switching angles by hand or with SmartSwitch. A long take is split wherever its clock has
  drifted half a frame from the sound's, so the sync holds to the end (multicam clips sync once).
- captions.srt: the caption chunks juice shows, for File > Import > Subtitle.

Framing (crops, zooms, follows) stays juice's: a clip carries it in its metadata, not as a Resolve transform.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import ClassVar

from ..media.sync import _probe
from . import sfx as sfxmod
from . import timeline as tlmod
from .spec import Spec

EXACT = {23.976: 24000 / 1001, 29.97: 30000 / 1001, 59.94: 60000 / 1001, 47.952: 48000 / 1001, 119.88: 120000 / 1001}


def _rate(fps: float) -> float:
    """NTSC rates exactly (59.94 is 60000/1001), so long timelines don't creep."""
    for k, v in EXACT.items():
        if abs(fps - k) < 0.01:
            return v
    return float(round(fps, 3))


def _rt(frames: float, rate: float) -> dict:
    return {"OTIO_SCHEMA": "RationalTime.1", "rate": rate, "value": float(frames)}


def _tr(start: float, dur: float, rate: float) -> dict:
    return {"OTIO_SCHEMA": "TimeRange.1", "start_time": _rt(start, rate), "duration": _rt(dur, rate)}


def _item(schema: str, name: str, frames: int, rate: float, **kw) -> dict:
    return {"OTIO_SCHEMA": schema, "metadata": kw.pop("metadata", {}), "name": name,
            "source_range": _tr(kw.pop("start", 0), frames, rate), "effects": kw.pop("effects", []),
            "markers": [], "enabled": True, **kw}


class Media:
    """A file as an editor sees it: its frame rate and the timecode its first frame carries."""
    _cache: ClassVar[dict] = {}

    def __init__(self, path: Path):
        self.path = Path(path)
        info = Media._cache.get(self.path) or Media._cache.setdefault(self.path, _probe(self.path))
        self.video, self.audio = info["video"], info["audio"]
        self.rate = _rate(info["fps"]) if self.video else None
        self.tc = info.get("tc_s", 0.0)
        self.duration = info["duration"]

    def ref(self, rate: float) -> dict:
        r = self.rate or rate
        return {"OTIO_SCHEMA": "ExternalReference.1", "metadata": {}, "name": self.path.name,
                "available_range": _tr(round(self.tc * r), int(self.duration * r), r),
                "available_image_bounds": None, "target_url": self.path.resolve().as_uri()}

    def clip(self, name: str, local_s: float, frames: int, rate: float, **kw) -> dict:
        """frames (on a `rate` timeline) of this file from local_s seconds into it."""
        r = self.rate or rate
        dur = round(frames * r / rate, 3)
        return _item("Clip.2", name, dur, r, start=round((self.tc + local_s) * r),
                     media_references={"DEFAULT_MEDIA": self.ref(rate)},
                     active_media_reference_key="DEFAULT_MEDIA", **kw)


def _gap(frames: int, rate: float) -> dict:
    return _item("Gap.1", "", frames, rate)


def _track(name: str, kind: str, placed: list[tuple[int, int, dict]], rate: float) -> dict:
    """A track from (first frame, frame count, item) placements, with gaps between them."""
    kids, at = [], 0
    for f0, n, item in sorted(placed, key=lambda x: x[0]):
        if f0 > at:
            kids.append(_gap(f0 - at, rate))
        kids.append(item)
        at = max(at, f0) + n
    return {"OTIO_SCHEMA": "Track.1", "metadata": {}, "name": name, "source_range": None, "effects": [],
            "markers": [], "enabled": True, "kind": kind, "children": kids}


def _pack(placed: list[tuple[int, int, dict]]) -> list[list]:
    """Spread placements over as few tracks as keep any two from overlapping."""
    tracks: list[list] = []
    for p in sorted(placed, key=lambda x: x[0]):
        for t in tracks:
            if t[-1][0] + t[-1][1] <= p[0]:
                t.append(p)
                break
        else:
            tracks.append([p])
    return tracks


def _timeline(name: str, rate: float, tracks: list[dict], markers: list[dict]) -> dict:
    return {"OTIO_SCHEMA": "Timeline.1", "metadata": {"juice": {"made_by": "juice export"}}, "name": name,
            "global_start_time": _rt(3600 * round(rate), rate),     # 01:00:00:00, as Resolve starts timelines
            "tracks": {"OTIO_SCHEMA": "Stack.1", "metadata": {}, "name": "tracks", "source_range": None,
                       "effects": [], "markers": markers, "enabled": True, "children": tracks}}


def _marker(name: str, f0: int, rate: float, comment: str = "", color: str = "BLUE") -> dict:
    return {"OTIO_SCHEMA": "Marker.2", "metadata": {}, "name": name, "color": color,
            "marked_range": _tr(f0, 1, rate), "comment": comment}


def _why(shot) -> str:
    parts = [shot.note] if shot.note else []
    if shot.speed != 1:
        parts.append(f"speed {shot.speed:g}x")
    if shot.zoom != 1.0 or shot.punch_in:
        parts.append(f"zoom {shot.zoom}" + (f", punch-in {shot.punch_in:g}" if shot.punch_in else ""))
    if shot.text:
        parts.append("text: " + " / ".join(shot.text.lines))
    return "; ".join(parts)


def cut(spec: Spec, out_dir: Path) -> tuple[dict, dict]:
    rate = float(spec.fps)
    tl = tlmod.build(spec)
    video, sound, markers = [], [], []
    f0 = 0
    for i, (shot, t0) in enumerate(zip(spec.shots, tl.starts)):
        n = spec.shot_frames(shot)
        live = min(n, round((shot.end - shot.start) / shot.speed * rate))
        pic = Media(spec.video(shot))
        meta = {"juice": {"shot": i, "source": shot.source, "angle": shot.angle, "start": shot.start,
                          "end": shot.end, "frame_x": shot.frame_x, "zoom": shot.zoom, "punch_in": shot.punch_in,
                          "follow": shot.follow, "speed": shot.speed}}
        warp = [{"OTIO_SCHEMA": "LinearTimeWarp.1", "metadata": {}, "name": "speed", "effect_name": "LinearTimeWarp",
                 "time_scalar": shot.speed}] if shot.speed != 1 else []
        name = f"{i:02d} {shot.source}" + (f" {shot.angle}" if shot.angle else "")
        video.append((f0, live, pic.clip(name, spec.local(shot, shot.start), live, rate, effects=warp, metadata=meta)))
        if n > live:     # a freeze: the last frame, held
            hold = [{"OTIO_SCHEMA": "FreezeFrame.1", "metadata": {}, "name": "freeze", "effect_name": "FreezeFrame",
                     "time_scalar": 0.0}]
            last = spec.local(shot, shot.end) - 1 / (pic.rate or rate)
            video.append((f0 + live, n - live, pic.clip(name + " freeze", last, n - live, rate, effects=hold)))
        if spec.original_audio:
            snd = Media(spec.sources[shot.source])
            sound.append((f0, live, snd.clip(name, shot.start, live, rate, effects=warp)))
        markers.append(_marker(name, f0, rate, _why(shot), "GREEN" if shot.angle else "BLUE"))
        f0 += n
    total = f0

    tracks = [_track("V1 cut", "Video", video, rate)]
    over = spec.edit_dir / f"overlays{'_' + spec.format if spec.format else ''}.mov"
    if over.exists():
        tracks.append(_track("V2 graphics", "Video", [(0, total, Media(over).clip("juice graphics", 0, total, rate))], rate))
    audio = [_track("A1 sound", "Audio", sound, rate)] if sound else []
    if spec.music:
        m = Media(spec.music.file)
        audio.append(_track("music", "Audio", [(0, total, m.clip(spec.music.file.stem, spec.music.offset, total, rate))],
                            rate))
    hits = []
    for at, sound_name in tl.sfx:
        wav = sfxmod.render(sound_name, out_dir / "sfx" / f"{sound_name}.wav")
        a = round(max(0.0, at - sfxmod.ATTACK[sound_name]) * rate)
        n = max(1, int(Media(wav).duration * rate))
        hits.append((a, min(n, total - a), Media(wav).clip(sound_name, 0, min(n, total - a), rate)))
    for k, t in enumerate(_pack([h for h in hits if h[1] > 0])):
        audio.append(_track(f"sfx {k + 1}", "Audio", t, rate))
    name = f"{spec.path.parent.name} cut" + (f" {spec.format}" if spec.format else "")
    return _timeline(name, rate, tracks + audio, markers), {"frames": total, "fps": rate, "shots": len(spec.shots),
                                                            "graphics": over.exists(), "sfx": len(hits)}


def angles(spec: Spec, key: str) -> tuple[dict, dict]:
    """Every camera of a multicam group on its own track, on the group's clock."""
    g = spec.groups[key]
    cams = {k: a for k, a in g.angles.items() if a.video}
    rate = _rate(Counter(round(Media(t.path).rate, 2) for a in cams.values() for t in a.takes if t.video)
                 .most_common(1)[0][0])
    t0 = min(t.offset for a in g.angles.values() for t in a.takes)
    tracks, audio, pieces, end = [], [], 0, 0
    for k, a in cams.items():
        pics, snds = [], []
        for tk in a.takes:
            m = Media(tk.path)
            # Split where the file's clock has drifted half a frame from the sound's; each piece is placed anew.
            step = tk.duration if abs(tk.rate - 1) < 1e-9 else min(tk.duration, 0.5 / rate / abs(tk.rate - 1))
            f_at, f_last = round((tk.covers[0] - t0) * rate), round((tk.covers[1] - t0) * rate)
            while f_at < f_last:
                n = min(f_at + max(1, round(step * rate)), f_last) - f_at
                local = max(0.0, tk.local(f_at / rate + t0))
                pics.append((f_at, n, m.clip(k, local, n, rate)))
                if m.audio and tk.path != g.sound.path:     # the group's own sound goes on A1
                    snds.append((f_at, n, m.clip(k, local, n, rate)))
                f_at += n
        tracks.append(_track(k, "Video", pics, rate))
        if snds:
            audio.append(_track(f"{k} sound", "Audio", snds, rate))
        pieces += len(pics)
        end = max(end, max(f + n for f, n, _ in pics))
    n = int(g.sound.duration * g.sound.rate * rate)
    sound = Media(g.sound.path).clip(g.sound.path.stem, 0, n, rate)
    audio.insert(0, _track(f"sound ({g.sound.path.name})", "Audio", [(round((g.sound.offset - t0) * rate), n, sound)], rate))
    return _timeline(f"{key} angles", rate, tracks + audio, []), {
        "angles": len(cams), "pieces": pieces, "fps": round(rate, 3), "length_s": round(end / rate, 2)}


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def srt(spec: Spec) -> str | None:
    """The caption chunks as subtitles (the theme's max words per chunk, broken at cuts and pauses)."""
    from .components import _caption_chunks
    tl = tlmod.build(spec)
    words = tl.words
    if not words:
        return None
    cap = next((L.params for L in tl.layers if L.comp == "captions"), {})
    chunks = _caption_chunks(words, cap.get("max_words", 3), tuple(tl.starts[1:]))
    out, seen = [], set()
    for a, b in chunks:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        text = " ".join(w["text"] for w in words[a:b])
        out.append(f"{len(out) + 1}\n{_srt_time(words[a]['start'])} --> {_srt_time(words[b - 1]['end'])}\n"
                   f"{text.upper() if cap.get('uppercase', True) else text}\n")
    return "\n".join(out)


README = """# Opening this edit in DaVinci Resolve

1. File > Import > Timeline, pick `cut.otio` (tick "Automatically import source clips into media pool").
   V1 is juice's cut, A1 its sound, then music and sound effects. {graphics}
   Each cut has a marker saying which angle and why.
{angles}
{captions}
Framing (crops, zooms, follows) is juice's: in Resolve the clips start un-cropped. Render the finished cut
with `juice render`, or reframe in Resolve.
"""


def export(spec: Spec, out_dir: Path | None = None) -> dict:
    out_dir = Path(out_dir) if out_dir else spec.edit_dir / ("resolve" + (f"_{spec.format}" if spec.format else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    tl, info = cut(spec, out_dir)
    files = {"cut": str(_write(tl, out_dir / "cut.otio"))}
    steps = []
    for key in spec.groups:
        at, ainfo = angles(spec, key)
        files[f"angles_{key}"] = str(_write(at, out_dir / f"angles_{key}.otio"))
        info[f"angles_{key}"] = ainfo
        steps.append(f"2. File > Import > Timeline, pick `angles_{key}.otio`: every camera of \"{key}\" on its own track,\n"
                     f"   synced by juice. In the Media Pool, right-click the \"{key} angles\" timeline > Convert Compound\n"
                     f"   Clips (Timelines) to Multicam Clips (it can't be undone: duplicate the timeline first to keep it).\n"
                     f"   Switch angles on it by hand, or with SmartSwitch (Studio).")
    sub = srt(spec)
    if sub:
        (out_dir / "captions.srt").write_text(sub)
        files["captions"] = str(out_dir / "captions.srt")
    graphics = ("V2 holds the graphics on transparent." if info["graphics"] else
                "No graphics yet:\n   `juice overlays reel.json`, then export again, puts them on V2.")
    (out_dir / "README.md").write_text(README.format(
        graphics=graphics, angles="\n".join(steps),
        captions="3. File > Import > Subtitle, pick `captions.srt`, and drag it onto the cut.\n" if sub else ""))
    return {"output": str(out_dir), "files": files, **info}


def _write(data: dict, path: Path) -> Path:
    path.write_text(json.dumps(data, indent=2))
    return path
