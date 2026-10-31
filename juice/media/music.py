"""Get a track and find where its lyrics land, so music can be aligned to cuts."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from . import transcribe


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_") or "track"


def search(query: str, n: int = 5) -> list[dict]:
    r = subprocess.run(["yt-dlp", "--no-playlist", "--print", "%(title)s\t%(duration)s\t%(view_count)s\t%(webpage_url)s",
                        f"ytsearch{n}:{query}"], capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            out.append({"title": parts[0], "duration": parts[1], "views": parts[2], "url": parts[3]})
    return out


def fetch(url: str, music_dir: Path, name: str) -> Path:
    music_dir.mkdir(parents=True, exist_ok=True)
    out = music_dir / f"{slug(name)}.mp3"
    if out.exists():
        return out
    r = subprocess.run(["yt-dlp", "-q", "--no-playlist", "-x", "--audio-format", "mp3",
                        "-o", str(music_dir / f"{slug(name)}.%(ext)s"), url], capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"download failed (try another upload of the same song): {r.stderr.strip()[-300:]}")
    return out


def lyrics(track: Path) -> list[tuple[float, float, str, str]]:
    """Lyric phrases with timings. Sung vocals sometimes come back as a single [singing] event;
    then align by the waveform in a filmstrip instead."""
    t = transcribe.transcribe(track, track.parent / "transcripts")
    return transcribe.phrases(t, gap=0.6)


def offset_for(lyric_time: float, cut_time: float, lead: float = 0.04) -> float:
    """Music offset so a lyric at `lyric_time` (song time) lands `lead` s after `cut_time` (reel time)."""
    return round(lyric_time - cut_time - lead, 3)
