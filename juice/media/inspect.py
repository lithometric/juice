"""Cheap ways to look at footage without watching it.

- filmstrip: N frames across a time range + the audio waveform, as one PNG
- contact_sheet: frames at exact times in a labeled grid
- loudness: integrated LUFS and true peak of a file
(the render verifier is juice.shortform.check)
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def probe_duration(media: Path, stream: str | None = None) -> float:
    cmd = ["ffprobe", "-v", "error"]
    if stream:
        cmd += ["-select_streams", stream, "-show_entries", "stream=duration"]
    else:
        cmd += ["-show_entries", "format=duration"]
    out = subprocess.run(cmd + ["-of", "csv=p=0", str(media)], capture_output=True, text=True, check=True)
    # Phone footage can add side-data fields ("37.966667,"), so take the first number on the line.
    return float(out.stdout.strip().splitlines()[0].split(",")[0])


def frame_at(media: Path, t: float, width: int, out: Path) -> Path:
    # Seeking at or past the last frame returns nothing, so clamp inside the video stream's end
    # (the container can run longer than the picture when audio is a few ms longer).
    try:
        vend = probe_duration(media, "v:0")
    except (ValueError, IndexError):
        vend = probe_duration(media)
    t = max(0.0, min(t, vend - 0.1))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(media), "-frames:v", "1",
                    "-vf", f"scale={width}:-2", str(out)], check=True)
    return out


def filmstrip(media: Path, start: float, end: float, out: Path, n: int = 10, thumb_w: int = 180) -> Path:
    span = end - start
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        thumbs = [Image.open(frame_at(media, start + span * (i + 0.5) / n, thumb_w, tmp / f"{i}.jpg"))
                  for i in range(n)]
        th = thumbs[0].height
        wave_w, wave_h = thumb_w * n, 160
        wave = tmp / "wave.png"
        has_audio = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                    "stream=index", "-of", "csv=p=0", str(media)],
                                   capture_output=True, text=True).stdout.strip()
        if has_audio:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{start}", "-t", f"{span}", "-i", str(media),
                            "-filter_complex", f"[0:a:0]showwavespic=s={wave_w}x{wave_h}:colors=#7aa2ff",
                            "-frames:v", "1", str(wave)], check=True)
        sheet = Image.new("RGB", (wave_w, 30 + th + wave_h + 24), (18, 18, 22))
        d = ImageDraw.Draw(sheet)
        d.text((6, 8), f"{media.name}  {start:.2f}s -> {end:.2f}s  ({n} frames)", fill=(230, 230, 230))
        for i, im in enumerate(thumbs):
            sheet.paste(im, (i * thumb_w, 30))
            d.text((i * thumb_w + 4, 32), f"{start + span * (i + 0.5) / n:.2f}", fill=(255, 220, 0))
        if has_audio:
            sheet.paste(Image.open(wave).convert("RGB"), (0, 30 + th))
        for i in range(n + 1):
            x = int(i * wave_w / n)
            d.text((min(x, wave_w - 40), 30 + th + wave_h + 6), f"{start + span * i / n:.2f}",
                   fill=(150, 150, 150))
        sheet.save(out)
    return out


def contact_sheet(media: Path, times: list[float], out: Path, width: int = 480, cols: int = 4) -> Path:
    """Frames at exact times in a labeled grid: the quickest way to find the frame something happens."""
    with tempfile.TemporaryDirectory() as tmp:
        ims = [Image.open(frame_at(media, t, width, Path(tmp) / f"{i}.jpg")).convert("RGB")
               for i, t in enumerate(times)]
        w, h = ims[0].size
        rows = (len(ims) + cols - 1) // cols
        sheet = Image.new("RGB", (w * min(cols, len(ims)), h * rows), (18, 18, 22))
        d = ImageDraw.Draw(sheet)
        for i, (im, t) in enumerate(zip(ims, times)):
            x, y = (i % cols) * w, (i // cols) * h
            sheet.paste(im, (x, y))
            d.rectangle([x, y, x + 74, y + 20], fill=(0, 0, 0))
            d.text((x + 5, y + 4), f"{t:.2f}s", fill=(255, 220, 0))
        sheet.save(out)
    return out


def loudness(media: Path) -> dict:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(media), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    summary = r.stderr[r.stderr.rfind("Summary:"):]
    i = re.search(r"I:\s+(-?[\d.]+) LUFS", summary)
    p = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary)
    return {"integrated_lufs": float(i.group(1)) if i else None, "true_peak_db": float(p.group(1)) if p else None}
