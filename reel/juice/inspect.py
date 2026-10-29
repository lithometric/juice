"""Cheap ways to look at footage and verify a render without watching it.

- filmstrip: N frames across a time range + the audio waveform, as one PNG
- check: duration, loudness, a frame on each side of every cut, and text bounds
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from . import text as textmod
from .spec import Spec


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


def footage_problems(spec: Spec) -> list[str]:
    """Editorial checks on the cut itself: no moment of a clip used twice, and every clip used.
    A shot marked "replay": true (a deliberate flash-forward/replay) is allowed to repeat footage."""
    out = []
    used = {}
    for i, s in enumerate(spec.shots):
        if s.replay:
            continue
        for j, a, b in used.get(s.source, []):
            overlap = min(b, s.end) - max(a, s.start)
            if overlap > 0.05:
                out.append(f"shots {j} and {i} reuse the same footage ({s.source} {max(a, s.start):.2f}-"
                           f"{min(b, s.end):.2f}s); cut it once, or mark the repeat \"replay\": true")
        used.setdefault(s.source, []).append((i, s.start, s.end))
    unused = [k for k in spec.sources if k not in {s.source for s in spec.shots}]
    if unused:
        out.append(f"clips provided but not used: {', '.join(unused)}")
    return out


def check(spec: Spec) -> dict:
    """Verify the rendered output against the spec. Returns a report with any problems."""
    out = spec.output
    if not out.exists():
        raise SystemExit(f"not rendered yet: {out}")
    verify = spec.edit_dir / (f"verify_{spec.format}" if spec.format else "verify")
    shutil.rmtree(verify, ignore_errors=True)  # only this render's checks
    verify.mkdir()

    problems = footage_problems(spec)
    expected = round(spec.duration, 3)
    vdur = probe_duration(out, "v:0")
    adur = probe_duration(out, "a:0")
    if abs(vdur - expected) > 1.5 / spec.style["fps"]:
        problems.append(f"video is {vdur:.3f}s, spec expects {expected:.3f}s")
    if abs(adur - vdur) > 0.06:
        problems.append(f"audio ({adur:.3f}s) and video ({vdur:.3f}s) lengths differ")

    loud = loudness(out)
    target = spec.style["loudness"]
    if loud["integrated_lufs"] is not None and abs(loud["integrated_lufs"] - target["I"]) > 1.0:
        problems.append(f"loudness {loud['integrated_lufs']} LUFS, target {target['I']}")
    if loud["true_peak_db"] is not None and loud["true_peak_db"] > -1.0:
        problems.append(f"peak {loud['true_peak_db']} dB is above -1 dB")

    # Text must stay clear of the apps' UI: top bar, bottom caption area, side margins, action buttons.
    zone, plat = spec.safe_zone, spec.orientation
    for i, s in enumerate(spec.shots):
        if not s.text:
            continue
        l, t, r, b = textmod.bounds(s.text, spec.style)
        if l < zone["side"] or r > 1 - zone["side"]:
            problems.append(f"shot {i} text runs into the side margin ({l:.2f}-{r:.2f} of width)")
        if t < zone["top"]:
            problems.append(f"shot {i} text is under the top UI ({plat}: keep below {zone['top']:.2f}, top at {t:.2f})")
        if b > 1 - zone["bottom"]:
            problems.append(f"shot {i} text is in the bottom caption/UI area "
                            f"({plat}: keep above {1 - zone['bottom']:.2f}, bottom at {b:.2f})")
        btn = zone.get("buttons")
        if btn and r > btn["left"] and b > btn["top"] and t < btn["bottom"]:
            problems.append(f"shot {i} text overlaps the action buttons on the right ({plat})")

    # One frame either side of every cut, side by side, so a flash or jump is visible at a glance.
    cut_imgs = []
    for k, t in enumerate(spec.cut_points()):
        cut_imgs.append(frame_at(out, t - 0.04, 240, verify / f"cut{k}_a.jpg"))
        cut_imgs.append(frame_at(out, t + 0.02, 240, verify / f"cut{k}_b.jpg"))
    if not cut_imgs:
        cut_imgs = [frame_at(out, 0, 240, verify / "first.jpg"), frame_at(out, vdur, 240, verify / "last.jpg")]
    ims = [Image.open(p) for p in cut_imgs]
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 8 * len(ims), ims[0].height), (18, 18, 22))
    x = 0
    for k, im in enumerate(ims):
        sheet.paste(im, (x, 0))
        x += im.width + (16 if k % 2 else 0)
    sheet.save(verify / "cuts.jpg")
    for p in cut_imgs:
        p.unlink()
    filmstrip(out, 0, max(vdur - 0.05, 0.1), verify / "overview.png", n=min(16, max(4, int(vdur * 1.5))))

    from . import hooks
    from . import timeline as tlmod
    from . import retention
    tl = tlmod.build(spec)
    hook, hook_probs = hooks.triple_hook(spec, tl, out)
    problems += hook_probs
    ret = retention.score(spec, tl)
    retention.plot(ret, verify / "retention.png", tl)
    retention.write(ret, verify / "retention.json")
    ret_summary = {k: ret[k] for k in ("score", "lift_pct", "retention_3s", "retention_end", "expected_watch_s")}
    ret_summary["tips"] = ret["tips"][:4]

    report = {"output": str(out), "video_s": vdur, "audio_s": adur, "expected_s": expected, **loud,
              "cuts_at": spec.cut_points(), "hook": hook, "retention": ret_summary, "problems": problems,
              "look_at": [str(verify / "cuts.jpg"), str(verify / "overview.png"), str(verify / "retention.png")]}
    (verify / "report.json").write_text(json.dumps(report, indent=2))
    return report
