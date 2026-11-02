"""Transparent overlays for an editor: everything juice lays over the footage of a cut (captions, chips,
titles, HUDs, tags, each shot's text) on a transparent background, as ProRes 4444 with alpha, on the
edit's own timeline. Put it on a track above the cut in Resolve, Premiere or Final Cut to grade or swap the
footage under juice's graphics without rendering them in.

    juice overlays reel.json             # edit/overlays.mov (overlays_<format>.mov for a multi-format spec)

The frames are what render_shot overlays (timeline.draw_frame) plus each shot's text PNG, straight alpha.
The whole-frame finish (vignette, grain, sharpening) belongs to the footage and isn't in it.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .. import encode
from . import text as textmod
from . import timeline as tlmod
from .spec import Spec


def render(spec: Spec, out: Path | None = None) -> dict:
    style = spec.style
    fps, W, H = style["fps"], style["width"], style["height"]
    out = Path(out) if out else spec.edit_dir / f"overlays{'_' + spec.format if spec.format else ''}.mov"
    out.parent.mkdir(parents=True, exist_ok=True)
    tl = tlmod.build(spec)
    ctx = tlmod.Ctx(spec, spec.theme, tl)
    n = sum(spec.shot_frames(s) for s in spec.shots)
    with tempfile.TemporaryDirectory(prefix="juice-overlays-") as tmp:
        texts = []    # (first frame, end frame, image) for each shot's own text
        f0 = 0
        for i, shot in enumerate(spec.shots):
            nf = spec.shot_frames(shot)
            if shot.text:
                png = textmod.render_png(shot.text, style, Path(tmp) / f"{i:02d}_text.png")
                texts.append((f0, f0 + nf, Image.open(png).convert("RGBA")))
            f0 += nf
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps),
               "-i", "-", "-vf", ",".join(encode.alpha_filters(style)), *encode.alpha_args(style),
               "-r", str(fps), "-an", str(out)]
        log = Path(tmp) / "ffmpeg.log"
        with open(log, "wb") as errf:     # a file, not a pipe: a full stderr pipe would deadlock the writer
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
            try:
                for f in range(n):
                    frame = tlmod.draw_frame(ctx, tl.layers, f / fps)
                    for a, b, img in texts:
                        if a <= f < b:
                            frame.alpha_composite(img)
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
            if proc.wait() != 0:
                raise RuntimeError(f"overlay encode failed:\n{log.read_text(errors='replace')[-1500:]}")
    return {"output": str(out), "format": spec.format, "size": [W, H], "fps": fps, "frames": n,
            "duration_s": round(n / fps, 3), "encoder": encode.alpha_args(style)[1],
            "layers": sorted({L.comp for L in tl.layers}) + (["shot text"] if texts else []),
            "color": encode.color_report(out)["summary"]}
