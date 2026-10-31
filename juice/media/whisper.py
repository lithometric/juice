"""Local word-level transcripts (mlx-whisper on Apple silicon) when there's no ElevenLabs key.

Written in the ElevenLabs Scribe shape and to transcribe.cache_path, so words/phrases/hooks/captions
read them like any other cached transcript and never call the API.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .transcribe import cache_path

MODEL = "mlx-community/whisper-large-v3-turbo"
# Default decoding hallucinates loops ("Correction. Correction.") over quiet stretches; these don't.
ARGS = ["--word-timestamps", "True", "--output-format", "json", "--condition-on-previous-text", "False",
        "--hallucination-silence-threshold", "2", "--verbose", "False"]


def to_scribe(data: dict, speaker: str = "speaker_0") -> dict:
    words = [{"text": w["word"].strip(), "start": round(w["start"], 3), "end": round(w["end"], 3),
              "type": "word", "speaker_id": speaker}
             for seg in data.get("segments", []) for w in seg.get("words", []) if w["word"].strip()]
    return {"language_code": data.get("language"), "text": " ".join(w["text"] for w in words), "words": words}


def wav16k(media: Path, out: Path, audio_track: int = 0) -> Path:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(media), "-map", f"0:a:{audio_track}",
                    "-ac", "1", "-ar", "16000", str(out)], check=True)
    return out


def transcribe(media: Path, out_dir: Path, wav: Path | None = None, audio_track: int = 0) -> Path:
    """Transcribe once, like transcribe.transcribe; pass wav to reuse an already extracted 16 kHz track."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = cache_path(media, out_dir)
    if out.exists() and out.stat().st_mtime >= media.stat().st_mtime:
        return out
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = wav or wav16k(media, tmp / "audio.wav", audio_track)
        p = subprocess.run(["uvx", "--from", "mlx-whisper", "mlx_whisper", str(wav), "--model", MODEL, *ARGS,
                            "--output-dir", str(tmp), "--output-name", "out"], capture_output=True, text=True)
        if p.returncode != 0 or not (tmp / "out.json").exists():
            raise RuntimeError(f"mlx_whisper failed: {p.stderr.strip()[-300:]}")
        data = to_scribe(json.loads((tmp / "out.json").read_text()))
    tmp_out = out.with_suffix(".json.part")
    tmp_out.write_text(json.dumps(data))
    tmp_out.replace(out)
    return out
