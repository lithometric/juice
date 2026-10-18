"""Themes: the look of a video, as data.

A script or a human decides the look ("bold fighting-game", "clean documentary", "loud rage bait");
juice turns that into a theme and every component draws itself from it, so one video stays
consistent and a new look is a new theme file, not new code.

A theme has:
  fonts    display / label / body faces (Google Fonts, fetched once into ~/.cache/juice/fonts)
  colors   named palette; components refer to roles (accent, ink, paper, ...) not hex values
  chip     small slanted label boxes: skew, padding, uppercase, letter spacing
  shadow   "hard" (offset block shadow) or "soft" (blurred) for big type
  finish   whole-frame look: grain, vignette, sharpen

Built-in themes live in juice_reel/themes/*.json. A spec picks one with "theme": "bold", points at a
file, or inlines overrides: "theme": {"base": "bold", "colors": {"accent": [0, 200, 255]}}.
"""
from __future__ import annotations

import json
import urllib.request
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

THEMES = Path(__file__).with_name("themes")
FONT_DIR = Path.home() / ".cache" / "juice" / "fonts"
GOOGLE = "https://github.com/google/fonts/raw/main/ofl/"
# Google Fonts we know how to fetch: name -> (repo path, local file name)
FONTS = {
    "Anton": ("anton/Anton-Regular.ttf", "Anton-Regular.ttf"),
    "Bebas Neue": ("bebasneue/BebasNeue-Regular.ttf", "BebasNeue-Regular.ttf"),
    "Archivo Black": ("archivoblack/ArchivoBlack-Regular.ttf", "ArchivoBlack-Regular.ttf"),
    "JetBrains Mono": ("jetbrainsmono/JetBrainsMono%5Bwght%5D.ttf", "JetBrainsMono[wght].ttf"),
    "Inter": ("inter/Inter%5Bopsz,wght%5D.ttf", "Inter[opsz,wght].ttf"),
}
FALLBACK = ("/System/Library/Fonts/HelveticaNeue.ttc", 1)


def font_path(name: str) -> Path | None:
    if name not in FONTS:
        p = Path(name).expanduser()
        return p if p.exists() else None
    repo, local = FONTS[name]
    p = FONT_DIR / local
    if not p.exists():
        FONT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(GOOGLE + repo, p)
        except OSError:
            return None
    return p


@lru_cache(maxsize=256)
def font(name: str, size: int, weight: int | None = None) -> ImageFont.FreeTypeFont:
    p = font_path(name)
    if p is None:  # offline and not cached: still render, in a known face, never a silent default
        return ImageFont.truetype(FALLBACK[0], max(6, size), index=FALLBACK[1])
    f = ImageFont.truetype(str(p), max(6, size))
    try:
        axes = f.get_variation_axes()
    except (OSError, AttributeError):
        axes = []
    if axes:
        # Set each axis by name (order differs per font, e.g. Inter is [opsz, wght]).
        vals = []
        for ax in axes:
            tag = ax.get("name", b"")
            tag = tag.decode() if isinstance(tag, bytes) else str(tag)
            lo, hi, default = ax["minimum"], ax["maximum"], ax["default"]
            if "eight" in tag:
                v = weight if weight is not None else default
            elif "ptical" in tag:
                v = size
            else:
                v = default
            vals.append(min(max(v, lo), hi))
        try:
            f.set_variation_by_axes(vals)
        except (OSError, ValueError):
            pass
    return f


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(spec_value, base_dir: Path | None = None) -> dict:
    """Resolve a theme name, file path or inline dict (with optional "base") to a full theme.
    Anything a theme leaves out comes from "bold", so partial themes always render."""
    raw = spec_value if spec_value is not None else "bold"
    if isinstance(raw, str):
        p = THEMES / f"{raw}.json"
        if not p.exists() and base_dir is not None:
            p = base_dir / raw
        if not p.exists():
            raise ValueError(f"unknown theme {raw!r} (built-in: {', '.join(names())})")
        raw = json.loads(p.read_text())
    if raw.get("base"):
        return _merge(load(raw["base"], base_dir), {k: v for k, v in raw.items() if k != "base"})
    if raw.get("name") == "bold":
        return raw
    return _merge(load("bold"), raw)


def names() -> list[str]:
    return sorted(p.stem for p in THEMES.glob("*.json"))


def color(theme: dict, c, alpha: int = 255) -> tuple:
    """A role name ("accent") or an [r, g, b(, a)] list -> RGBA tuple."""
    if isinstance(c, str):
        c = theme["colors"][c]
    c = tuple(c)
    return c if len(c) == 4 else c + (alpha,)


# --- drawing primitives shared by every component -------------------------------------------------

def text_size(text: str, f, spacing: float = 0) -> tuple[int, int]:
    d = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    w = sum(d.textlength(c, font=f) for c in text) + spacing * max(0, len(text) - 1) if spacing else d.textlength(text, font=f)
    asc, desc = f.getmetrics()
    return int(w), asc + desc


def draw_text(img: Image.Image, xy, text: str, f, fill, spacing: float = 0) -> None:
    d = ImageDraw.Draw(img)
    if not spacing:
        d.text(xy, text, font=f, fill=fill)
        return
    x, y = xy
    for c in text:
        d.text((x, y), c, font=f, fill=fill)
        x += d.textlength(c, font=f) + spacing


def skew(img: Image.Image, deg: float) -> Image.Image:
    """Shear horizontally (slanted labels). Positive leans right."""
    if not deg:
        return img
    import math
    k = math.tan(math.radians(deg))
    extra = int(abs(k) * img.height)
    out = img.transform((img.width + extra, img.height), Image.AFFINE,
                        (1, k, -extra if k > 0 else 0, 0, 1, 0), resample=Image.BICUBIC)
    return out


def chip(theme: dict, text: str, size: float, fill="ink", ink="paper", role: str = "label") -> Image.Image:
    """A small slanted label box (e.g. SLOW-MO 0.4x, OPENAI IS DOWN, WINGS: YES)."""
    c = theme["chip"]
    fspec = theme["fonts"][role]
    f = font(fspec["family"], int(size), fspec.get("weight"))
    t = text.upper() if c.get("uppercase", True) else text
    sp = size * c.get("tracking", 0.08)
    tw, th = text_size(t, f, sp)
    px, py = size * c.get("pad_x", 0.55), size * c.get("pad_y", 0.28)
    w, h = int(tw + 2 * px), int(size * 1.0 + 2 * py)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).rectangle([0, 0, w, h], fill=color(theme, fill))
    asc, _ = f.getmetrics()
    draw_text(img, (px, py + (size - asc) * 0.5 + size * 0.02), t, f, color(theme, ink), sp)
    return skew(img, c.get("skew", 0))


def headline(theme: dict, text: str, size: float, fill="paper", role: str = "display",
             shadow: str | None = None, outline=None) -> Image.Image:
    """Big type on its own canvas with the theme's shadow (hard block or soft blur)."""
    fspec = theme["fonts"][role]
    f = font(fspec["family"], int(size), fspec.get("weight"))
    t = text.upper() if fspec.get("uppercase", role == "display") else text
    sp = size * fspec.get("tracking", 0)
    tw, th = text_size(t, f, sp)
    sh = theme["shadow"]
    kind = shadow or sh.get("kind", "hard")
    off = int(size * sh.get("offset", 0.06))
    pad = int(size * 0.3) + off
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
    ol = outline if outline is not None else theme.get("outline")
    if kind == "hard":
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_text(layer, (pad + off, pad + off), t, f, color(theme, sh.get("color", "ink")), sp)
        if ol:  # thicken the block shadow so it reads as an outline too
            layer = layer.filter(ImageFilter.MaxFilter(ol * 2 + 1))
        img.alpha_composite(layer)
        if ol:
            o = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw_text(o, (pad, pad), t, f, color(theme, sh.get("color", "ink")), sp)
            img.alpha_composite(o.filter(ImageFilter.MaxFilter(ol * 2 + 1)))
    elif kind == "soft":
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_text(layer, (pad + off // 2, pad + off), t, f, color(theme, sh.get("color", "ink"), 170), sp)
        img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(2, off))))
    draw_text(img, (pad, pad), t, f, color(theme, fill), sp)
    return img


def paste(canvas: Image.Image, layer: Image.Image, cx: float, cy: float, scale: float = 1.0,
          alpha: float = 1.0, angle: float = 0.0, anchor: str = "center") -> None:
    """Composite `layer` centered (or anchored) at (cx, cy) with scale/opacity/rotation."""
    if alpha <= 0.002 or scale <= 0.002:
        return
    if scale != 1.0:
        layer = layer.resize((max(1, int(layer.width * scale)), max(1, int(layer.height * scale))), Image.BICUBIC)
    if angle:
        layer = layer.rotate(angle, resample=Image.BICUBIC, expand=True)
    if alpha < 1.0:
        layer = layer.copy()
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * alpha)))
    x = cx - layer.width / 2 if anchor in ("center", "top", "bottom") else (cx if anchor.endswith("left") or anchor == "left" else cx - layer.width)
    y = cy - layer.height / 2 if anchor in ("center", "left", "right") else (cy if anchor.startswith("top") else cy - layer.height)
    canvas.alpha_composite(layer, (int(x), int(y)))
