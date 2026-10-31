"""A project folder the agent works in, and the result of one tool call.

Writes stay inside the project (reel.json, scripts, the edit/ output). Reads of media can point anywhere,
since footage usually starts life in ~/Downloads. Pictures the tools make (filmstrips, contact sheets,
the verifier's sheets) are what the model gets to see instead of the video, so a result carries both
text and a list of images.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

MAX_SIDE = 1600       # images shown to the model are downscaled to this (long side), as JPEG


@dataclass
class Result:
    text: str
    images: list[Path] = field(default_factory=list)


class Workspace:
    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.look = self.root / "agent" / "look"     # pictures made for the model
        self.look.mkdir(parents=True, exist_ok=True)

    def path(self, p: str | Path) -> Path:
        """Resolve a path the model gave: relative to the project, ~ expanded."""
        q = Path(p).expanduser()
        return (q if q.is_absolute() else self.root / q).resolve()

    def inside(self, p: str | Path) -> Path:
        """Like path(), but the result must be inside the project (for writes)."""
        q = self.path(p)
        if self.root != q and self.root not in q.parents:
            raise ValueError(f"{p} is outside the project folder {self.root}")
        return q

    def rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    def media(self, p: str | Path) -> Path:
        q = self.path(p)
        if not q.exists():
            raise FileNotFoundError(f"no such file: {p} (looked at {q})")
        return q


def image_data_url(path: Path, max_side: int = MAX_SIDE) -> str:
    """The image as a data: URL, downscaled and JPEG-encoded so a contact sheet costs a few hundred KB."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = max_side / max(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_bytes(path: Path, max_side: int = MAX_SIDE) -> tuple[bytes, str]:
    url = image_data_url(path, max_side)
    return base64.b64decode(url.split(",", 1)[1]), "image/jpeg"
