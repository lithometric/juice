"""Preview every component in a theme on one sheet, without rendering a video.

    juice-reel gallery --theme loud --bg frame.jpg -o gallery.png

The quickest way for a human (or the agent) to judge a look before committing to a render.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from . import theme as th
from .components import COMPONENTS

DEMO = [
    ("versus", {"left": "CLAUDE", "right": "OPENAI", "premise": "FIRST ONE ON THE GROUND LOSES"}, 2.0),
    ("slam", {"text": "K.O.", "sub": "OPENAI IS DOWN"}, 1.0),
    ("title", {"kicker": "WINNER", "text": "CLAUDE", "sub": "STILL STANDING"}, 1.0),
    ("lower_third", {"name": "Kevin Vera", "role": "Founder, Lithometric"}, 1.0),
    ("captions", {}, 1.0),
    ("meters", {"left": {"name": "CLAUDE", "meter": "a", "label": "BALANCE"},
                "right": {"name": "OPENAI", "meter": "b", "label": "BALANCE", "color": "paper"},
                "counter": {"from": 99}}, 1.0),
    ("callout", {"target": "demo", "text": "-1 LEG", "label": "OPENAI"}, 1.0),
    ("tags", {"targets": {"CLAUDE": {"sub": "WINGS: YES"}}}, 1.0),
    ("ticker", {"text": "EVERY HIT SIMULATED // OPENBEAM", "dot": "danger"}, 1.0),
    ("endcard", {"title": "OPENBEAM", "tagline": "Scaling world-model physics.", "link": "openbeam.world",
                 "handle": "@openbeamworld"}, 1.5),
]


def render(theme_value, out: Path, bg: Path | None = None, W: int = 1920, H: int = 1080, cols: int = 3) -> Path:
    theme = th.load(theme_value)
    base = (Image.open(bg).convert("RGBA").resize((W, H)) if bg
            else Image.new("RGBA", (W, H), (120, 112, 104, 255)))
    ctx = SimpleNamespace(theme=theme, W=W, H=H, u=min(W, H) / 1080, fps=30, total=10.0, t_out=3.2,
                          meter=lambda m: {"a": 92, "b": 0}.get(m, 100),
                          target=lambda n: (W * 0.55, H * 0.55, 90, 90),
                          words=[{"text": "who", "start": 3.0, "end": 3.2}, {"text": "asks", "start": 3.2, "end": 3.4},
                                 {"text": "that?", "start": 3.4, "end": 3.8}])
    tiles = []
    for name, p, t in DEMO:
        c = base.copy()
        COMPONENTS[name](c, t, 3.0, p, ctx)
        ImageDraw.Draw(c).text((12, 10), name, fill=(255, 255, 0, 255))
        tiles.append(c.convert("RGB").resize((W // 2, H // 2)))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W // 2, rows * H // 2), (18, 18, 22))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % cols) * W // 2, (i // cols) * H // 2))
    sheet.save(out)
    return out
