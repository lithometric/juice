"""Word-level transcripts (ElevenLabs Scribe), cached per source file.

Word timestamps are what make cuts accurate: a shot starts a hair before the first
word that's visibly spoken, and music is lined up against lyric timings.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import requests

SCRIBE_URL = "https://api.elevenlabs.io/v1/speech-to-text"


def api_key() -> str:
    if os.environ.get("ELEVENLABS_API_KEY"):
        return os.environ["ELEVENLABS_API_KEY"]
    for env in (Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"):
        if env.exists():
            for line in env.read_text().splitlines():
                k, _, v = line.partition("=")
                if k.strip() == "ELEVENLABS_API_KEY" and v.strip():
                    return v.strip().strip('"').strip("'")
    raise SystemExit("ELEVENLABS_API_KEY not set (env var, or .env in the current dir or reel/)")


def cache_path(media: Path, out_dir: Path) -> Path:
    return out_dir / f"{media.stem}.json"


def transcribe(media: Path, out_dir: Path, audio_track: int = 0) -> Path:
    """Transcribe once; later calls return the cached JSON (sources don't change)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = cache_path(media, out_dir)
    if out.exists() and out.stat().st_mtime >= media.stat().st_mtime:
        return out
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(media), "-map", f"0:a:{audio_track}",
                        "-ac", "1", "-ar", "16000", str(wav)], check=True)
        with open(wav, "rb") as f:
            r = requests.post(SCRIBE_URL, headers={"xi-api-key": api_key()},
                              files={"file": ("audio.wav", f, "audio/wav")},
                              data={"model_id": "scribe_v1", "diarize": "true", "tag_audio_events": "true",
                                    "timestamps_granularity": "word"}, timeout=1800)
    if r.status_code != 200:
        raise RuntimeError(f"Scribe returned {r.status_code}: {r.text[:300]}")
    out.write_text(json.dumps(r.json()))
    return out


def words(transcript: Path, start: float | None = None, end: float | None = None) -> list[dict]:
    data = json.loads(transcript.read_text())
    out = []
    for w in data.get("words", []):
        if w.get("type") not in ("word", "audio_event"):
            continue
        if start is not None and w["end"] < start:
            continue
        if end is not None and w["start"] > end:
            continue
        out.append(w)
    return out


def phrases(transcript: Path, gap: float = 0.5) -> list[tuple[float, float, str, str]]:
    """Group words into phrases on silences >= gap or speaker change."""
    out, cur = [], []
    for w in words(transcript):
        if cur and (w["start"] - cur[-1]["end"] >= gap or w.get("speaker_id") != cur[-1].get("speaker_id")):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return [(p[0]["start"], p[-1]["end"], p[0].get("speaker_id", ""), " ".join(x["text"] for x in p)) for p in out]
