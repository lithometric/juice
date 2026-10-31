"""Voice isolation: strip room noise, chatter and traffic from a take, keep the voice.

DeepFilterNet3 (full-band 48 kHz speech enhancement, runs locally on the CPU in a fraction of real
time). On the OpenBeam takes it drops the floor between words from -45 dB to -79 dB, leaves the speech
level alone, and the transcript comes back word for word. (A music separator like Demucs doesn't
work here: it hears cafe chatter as vocals.) It runs in its own uvx environment, like mlx-whisper,
because it pins an older torch. Results are cached per source file.

    isolate.voices([Path("a.mov"), Path("b.mov")])   # {source: isolated 48 kHz mono wav}
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

CACHE = Path.home() / ".cache" / "juice" / "isolated"
DF = ["uvx", "--python", "3.11", "--with", "torch==2.0.1", "--with", "torchaudio==2.0.2", "--with", "numpy<2",
      "--from", "deepfilternet==0.5.6", "deepFilter"]


def cache_path(media: Path, atten_lim: float | None = None) -> Path:
    st = Path(media).stat()
    key = f"{Path(media).resolve()}|{st.st_size}|{int(st.st_mtime)}|{atten_lim}"
    return CACHE / f"{Path(media).stem}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.wav"


def voices(media: list[Path], atten_lim: float | None = None) -> dict[Path, Path]:
    """Isolate the voice in each file (one model load for all of them). atten_lim caps the noise
    reduction in dB (e.g. 24 keeps a little room tone); None removes all it can."""
    out = {Path(m): cache_path(m, atten_lim) for m in media}
    todo = [m for m, p in out.items() if not p.exists()]
    if not todo:
        return out
    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="juice-isolate-") as tmp:
        tmp = Path(tmp)
        wavs = []
        for i, m in enumerate(todo):
            w = tmp / f"in{i}.wav"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(m), "-map", "0:a:0", "-ac", "1", "-ar", "48000",
                            str(w)], check=True)
            wavs.append(w)
        cmd = DF + [str(w) for w in wavs] + ["-o", str(tmp / "out")]
        if atten_lim is not None:
            cmd += ["--atten-lim", str(atten_lim)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"voice isolation (DeepFilterNet) failed: {r.stderr.strip()[-400:]}")
        for i, m in enumerate(todo):
            res = tmp / "out" / f"in{i}_DeepFilterNet3.wav"
            if not res.exists():
                raise RuntimeError(f"voice isolation wrote nothing for {m}")
            shutil.move(str(res), out[m])
    return out
