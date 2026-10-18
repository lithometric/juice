"""On-screen text as transparent full-frame PNGs.

Text is drawn with PIL rather than ffmpeg drawtext/libass so it works with any
ffmpeg build and the font is guaranteed: a missing font is an error, never a
silent fallback.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .spec import Text


def render_png(text: Text, style: dict, out: Path) -> Path:
    w, h = style["width"], style["height"]
    f = style["font"]
    if not Path(f["path"]).exists():
        raise FileNotFoundError(f"font not found: {f['path']}")
    font = ImageFont.truetype(f["path"], f["size"], index=f.get("index", 0))
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    line_h = f["line_height"]
    y = int(h * text.y) - (line_h * len(text.lines)) // 2
    stroke = style.get("text_stroke", 0)
    for ln in text.lines:
        tw = d.textlength(ln, font=font)
        if text.align == "right":
            x = w * text.x - tw
        elif text.align == "left":
            x = w * text.x
        else:
            x = w * text.x - tw / 2
        kw = {"stroke_width": stroke, "stroke_fill": (0, 0, 0, 255)} if stroke else {}
        d.text((x, y), ln, font=font, fill=(*f["color"], 255), **kw)
        y += line_h
    img.save(out)
    return out


def bounds(text: Text, style: dict) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) of the text block as fractions of the frame."""
    f = style["font"]
    font = ImageFont.truetype(f["path"], f["size"], index=f.get("index", 0))
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    w, h = style["width"], style["height"]
    widest = max(d.textlength(ln, font=font) for ln in text.lines)
    if text.align == "right":
        left = w * text.x - widest
    elif text.align == "left":
        left = w * text.x
    else:
        left = w * text.x - widest / 2
    block_h = f["line_height"] * len(text.lines)
    top = h * text.y - block_h / 2
    return left / w, top / h, (left + widest) / w, (top + block_h) / h
