"""Static font files for any Google Fonts family and weight, fetched once into ~/.cache/juice/fonts/static.

Variable fonts render at their default weight in skia (and need per-axis setup in PIL), so the
motion graphics ask Google's CSS API for one static TTF per weight instead. Monospace families are
never used: they come back as NO_MONO.
"""
from __future__ import annotations

import re
import urllib.parse
import urllib.request
from pathlib import Path

STATIC_DIR = Path.home() / ".cache" / "juice" / "fonts" / "static"
NO_MONO = "Inter"
MONO_HINTS = ("mono", "code", "courier", "menlo", "consol", "typewriter", "fixed")
# An old browser user agent makes the CSS API answer with TrueType instead of WOFF2.
UA = "Mozilla/4.0"


def is_mono(name: str) -> bool:
    return any(h in Path(name).name.lower() for h in MONO_HINTS)


def static(family: str, weight: int = 400) -> Path | None:
    """A static TTF for family at weight (a file path passes through), or None offline/unknown."""
    if is_mono(family):
        family = NO_MONO
    p = Path(family).expanduser()
    if p.suffix.lower() in (".ttf", ".otf", ".ttc") and p.exists():
        return p
    weight = int(round(weight / 100) * 100)
    out = STATIC_DIR / f"{family.replace(' ', '')}-{weight}.ttf"
    if out.exists():
        return out
    q = urllib.parse.quote(family)
    try:
        req = urllib.request.Request(f"https://fonts.googleapis.com/css2?family={q}:wght@{weight}",
                                     headers={"User-Agent": UA})
        css = urllib.request.urlopen(req, timeout=10).read().decode()
        m = re.search(r"url\((https://[^)]+\.ttf)\)", css)
        if not m:
            return None
        STATIC_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(m.group(1), tmp)
        tmp.replace(out)
        return out
    except OSError:
        return None
