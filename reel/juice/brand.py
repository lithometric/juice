"""Brand kits: juice.json (the Juice Brand Kit Protocol, gathered by `npx juice-press`).

A kit names the brand's colors, fonts, logo, line and link. Juice turns it into a theme (so every
component is on-brand) and hands the rest (logo files, name, url) to the motion graphics.

Discovery, first hit wins: an explicit path / URL / domain, then juice.json next to the edit, then
https://<domain>/.well-known/juice.json for a domain.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

ROLES = {"primary": "accent", "secondary": "accent2", "background": "ink", "text": "paper",
         "muted": "muted", "surface": "surface"}


def rgb(hex_: str) -> list[int]:
    h = hex_.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def _fetch(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "juice"})
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    except (OSError, ValueError):
        return None


def load(path: str | Path) -> dict:
    """Read a kit and resolve its asset paths against the kit's folder."""
    path = Path(path).expanduser().resolve()
    kit = json.loads(path.read_text())
    return _resolve(kit, path.parent)


def _resolve(kit: dict, base: Path | None) -> dict:
    if kit.get("juice") != "1" or not kit.get("name") or not kit.get("colors", {}).get("primary"):
        raise ValueError("not a juice brand kit (needs juice: \"1\", name and colors.primary); "
                         "check it with `npx juice-press validate`")
    kit = dict(kit)

    def res(v):
        if not isinstance(v, str) or v.startswith(("http://", "https://")) or base is None:
            return v
        return str((base / v).resolve())
    kit["logo"] = {k: res(v) for k, v in kit.get("logo", {}).items()}
    kit["images"] = [res(v) for v in kit.get("images", [])]
    for f in kit.get("fonts", {}).values():
        if f.get("file"):
            f["file"] = res(f["file"])
    return kit


def find(value: str | None, base: Path) -> dict | None:
    if value:
        if value.startswith(("http://", "https://")) or ("." in value and "/" not in value
                                                         and not value.endswith(".json")):
            url = value if value.startswith("http") else f"https://{value}/.well-known/juice.json"
            kit = _fetch(url)
            if kit is None:
                raise ValueError(f"no brand kit at {url}")
            return _resolve(kit, None)
        return load(base / value)
    local = base / "juice.json"
    return load(local) if local.exists() else None


def colors(kit: dict) -> dict:
    """Theme color roles from the kit, deriving what it leaves out."""
    c = {ROLES.get(k, k): rgb(v) for k, v in kit["colors"].items()}
    dark = kit.get("style", {}).get("mode", "dark") == "dark"
    c.setdefault("accent2", c["accent"])
    c.setdefault("ink", [8, 9, 11] if dark else [250, 250, 248])
    c.setdefault("paper", [255, 255, 255] if dark else [16, 17, 19])
    c.setdefault("surface", [int(a * 0.85 + b * 0.15) for a, b in zip(c["ink"], c["paper"])])
    c.setdefault("muted", [int(a * 0.6 + b * 0.4) for a, b in zip(c["paper"], c["ink"])])
    return c


def font(kit: dict, role: str, default_weight: int) -> dict:
    """{"family", "weight"} for a role ("display" | "body"); a font file wins over the family name."""
    f = kit.get("fonts", {}).get(role) or kit.get("fonts", {}).get("body") or {"family": "Inter"}
    return {"family": f.get("file") or f.get("family", "Inter"), "weight": f.get("weight", default_weight)}


def theme(kit: dict, base_theme=None) -> dict:
    """A theme overlay for juice's components: the kit's colors and fonts over base_theme (default clean)."""
    if isinstance(base_theme, dict) and "base" not in base_theme:
        base_theme = {"base": "clean", **base_theme}
    over = base_theme if isinstance(base_theme, dict) else {"base": base_theme or "clean"}
    body = font(kit, "body", 500)
    fonts = {"display": {**font(kit, "display", 800)},
             "label": {"family": body["family"], "weight": 800},
             "body": body}
    out = {**over, "colors": {**colors(kit), **over.get("colors", {})}}
    out["fonts"] = {k: {**fonts[k], **over.get("fonts", {}).get(k, {})} for k in fonts}
    return out
