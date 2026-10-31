"""Drawing primitives for motion graphics: brand context, type, easing, footage, 3D panels, stage.

Everything draws in design units (1920x1080 or 1080x1920); a preview render scales the canvas
down, so scenes never need to know the output resolution. `u` is 1 at 1080 on the short side.
"""
from __future__ import annotations

import math
import re
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
import skia

from .. import brand as brand_mod
from .. import fonts as fonts_mod

SAMP = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)


# --- easing ---------------------------------------------------------------------------------------

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, t0, d):
    """0 -> 1 as t goes from t0 to t0 + d."""
    return clamp((t - t0) / max(d, 1e-6))


def lerp(a, b, x):
    return a + (b - a) * x


def e_out(x):
    return 1 - (1 - x) ** 3


def e_in(x):
    return x ** 3


def e_inout(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def e_expo(x):
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def spring(x, k=1.7):
    """Lands past 1 and settles: the pop of kinetic type, not a fade."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return 1 - math.exp(-6.5 * x) * math.cos(k * math.pi * x * 2.2)


# --- brand context --------------------------------------------------------------------------------

class Ctx:
    """What a scene can ask: frame size, scale unit, brand colors / fonts / logo, output fps."""

    def __init__(self, W: int, H: int, fps: int, kit: dict | None, scale: float = 1.0):
        self.W, self.H, self.fps, self.scale = W, H, fps, scale
        self.u = min(W, H) / 1080
        self.portrait = H > W
        self.kit = kit or {"juice": "1", "name": "", "colors": {"primary": "#FFC400"}}
        self.colors = {k: tuple(v) for k, v in brand_mod.colors(self.kit).items()}
        self.display = brand_mod.font(self.kit, "display", 800)
        self.body = brand_mod.font(self.kit, "body", 500)
        self.radius = float(self.kit.get("style", {}).get("corner_radius", 0.2))
        self.name = self.kit.get("name", "")

    def c(self, role, a=1.0):
        """skia color for a role name ("accent", "ink", ...) or an rgb tuple, at alpha a."""
        rgb = self.colors[role] if isinstance(role, str) else role
        return col(rgb, a)

    def font(self, size: float, weight: int | None = None, role: str = "body") -> skia.Font:
        f = self.display if role == "display" else self.body
        return font(f["family"], weight or f["weight"], size)


def col(rgb, a=1.0):
    return skia.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(clamp(a) * 255))


@lru_cache(maxsize=64)
def _typeface(family: str, weight: int) -> skia.Typeface:
    p = fonts_mod.static(family, weight)
    if p is not None:
        tf = skia.Typeface.MakeFromFile(str(p))
        if tf is not None:
            return tf
    # Offline and not cached: a system face at the right weight, never a monospace default.
    return skia.Typeface("Helvetica Neue", skia.FontStyle(weight, skia.FontStyle.kNormal_Width,
                                                         skia.FontStyle.kUpright_Slant))


def font(family: str, weight: int, size: float) -> skia.Font:
    f = skia.Font(_typeface(family, int(round(weight / 100) * 100)), size)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setSubpixel(True)
    return f


def tw(s: str, f: skia.Font, track: float = 0) -> float:
    return f.measureText(s) + track * max(len(s) - 1, 0)


def text(c, s, x, y, f, rgb=(255, 255, 255), a=1.0, align="l", track=0.0, blur=0.0) -> float:
    """Draw s with its baseline at y; align l/c/r on x. Returns the width."""
    p = skia.Paint(AntiAlias=True, Color=col(rgb, a))
    if blur > 0.3:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    w = tw(s, f, track)
    x0 = x - w if align == "r" else x - w / 2 if align == "c" else x
    if not track:
        c.drawString(s, x0, y, f, p)
    else:
        for ch in s:
            c.drawString(ch, x0, y, f, p)
            x0 += f.measureText(ch) + track
    return w


def fit(s: str, f: skia.Font, max_w: float) -> skia.Font:
    """The same font, smaller if s would be wider than max_w."""
    w = f.measureText(s)
    if w <= max_w:
        return f
    g = skia.Font(f.getTypeface(), f.getSize() * max_w / w)
    g.setEdging(skia.Font.Edging.kAntiAlias)
    g.setSubpixel(True)
    return g


# --- footage --------------------------------------------------------------------------------------

class Source:
    """A range of a video file, read sequentially on the output clock. Opens ffmpeg lazily at the
    first frame asked for, so a render chunk starting mid-scene seeks straight there."""

    def __init__(self, path, start=0.0, end=None, speed=1.0, crop=None):
        self.path, self.start, self.end, self.speed, self.crop = str(path), float(start), end, float(speed), crop
        self.p = None
        self.fps = 30
        self.size = (1280, 720)

    def configure(self, fps: int, size: tuple[int, int]):
        self.fps, self.size = fps, (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)

    @property
    def duration(self) -> float:
        return ((self.end or self.start + 3600) - self.start) / self.speed

    def _open(self, lt: float):
        s0 = self.start + lt * self.speed
        vf = [f"setpts=(PTS-STARTPTS)/{self.speed}", f"fps={self.fps}"]
        if self.crop:
            vf.append("crop={2}:{3}:{0}:{1}".format(*self.crop))
        vf += [f"scale={self.size[0]}:{self.size[1]}:force_original_aspect_ratio=increase:flags=bicubic",
               f"crop={self.size[0]}:{self.size[1]}", "format=rgba"]
        dur = [] if self.end is None else ["-t", f"{max(0.05, self.end - s0) + 0.2:.3f}"]
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{s0:.3f}", *dur, "-i", self.path, "-an",
               "-vf", ",".join(vf), "-f", "rawvideo", "-"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10 ** 7)
        self.t0, self.idx, self.arr, self.img = lt, -1, None, None

    def get(self, lt: float) -> skia.Image | None:
        lt = max(0.0, min(lt, self.duration - 1 / self.fps))
        if self.p is None or lt < self.t0 + self.idx / self.fps - 1e-6:
            self.close()
            self._open(lt)
        i = int(round((lt - self.t0) * self.fps))
        fb = self.size[0] * self.size[1] * 4
        while self.idx < i:
            buf = self.p.stdout.read(fb)
            if len(buf) < fb:
                break
            self.arr = np.frombuffer(buf, np.uint8).reshape(self.size[1], self.size[0], 4)
            self.idx += 1
            self.img = None
        if self.img is None and self.arr is not None:
            self.img = skia.Image.fromarray(self.arr, colorType=skia.kRGBA_8888_ColorType)
        return self.img

    def close(self):
        if self.p:
            self.p.kill()
            self.p.wait()
            self.p = None


@lru_cache(maxsize=32)
def still(path: str) -> skia.Image | None:
    """A still image (PNG/JPG, or SVG rasterized at 1024px)."""
    if path.lower().endswith(".svg"):
        return svg(path, 1024)
    return skia.Image.open(path) if Path(path).exists() else None


@lru_cache(maxsize=32)
def svg(path: str, size: int) -> skia.Image | None:
    if not Path(path).exists():
        return None
    dom = skia.SVGDOM.MakeFromStream(skia.Stream.MakeFromFile(path))
    if dom is None:
        return None
    w, h = size, size
    vb = re.search(r'viewBox="\s*[-\d.]+[\s,]+[-\d.]+[\s,]+([\d.]+)[\s,]+([\d.]+)', Path(path).read_text(errors="ignore"))
    if vb and float(vb.group(1)) > 0:
        h = int(size * float(vb.group(2)) / float(vb.group(1)))
    dom.setContainerSize(skia.Size(w, h))
    surf = skia.Surface(w, h)
    surf.getCanvas().clear(skia.ColorTRANSPARENT)
    dom.render(surf.getCanvas())
    return surf.makeImageSnapshot()


def cover(c, img, x, y, w, h, zoom=1.0, cx=0.5, cy=0.5, alpha=1.0, blur=0.0, dim=0.0, dim_rgb=(0, 0, 0)):
    """Fill the box with img (crop to fit), zoomed around (cx, cy)."""
    if img is None:
        return
    iw, ih = img.width(), img.height()
    s = max(w / iw, h / ih) * zoom
    sw, sh = w / s, h / s
    sx, sy = clamp(cx * iw - sw / 2, 0, iw - sw), clamp(cy * ih - sh / 2, 0, ih - sh)
    p = skia.Paint(AntiAlias=True, Alphaf=alpha)
    if blur > 0.3:
        p.setImageFilter(skia.ImageFilters.Blur(blur, blur, skia.TileMode.kClamp))
    c.save()
    c.clipRect(skia.Rect.MakeXYWH(x, y, w, h))
    c.drawImageRect(img, skia.Rect.MakeXYWH(sx, sy, sw, sh), skia.Rect.MakeXYWH(x, y, w, h), SAMP, p)
    if dim > 0:
        c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), skia.Paint(Color=col(dim_rgb, dim)))
    c.restore()


def contain(c, img, x, y, w, h, alpha=1.0):
    """Fit img inside the box, centered, keeping its aspect."""
    if img is None:
        return
    s = min(w / img.width(), h / img.height())
    dw, dh = img.width() * s, img.height() * s
    c.drawImageRect(img, skia.Rect.MakeXYWH(x + (w - dw) / 2, y + (h - dh) / 2, dw, dh), SAMP,
                    skia.Paint(AntiAlias=True, Alphaf=alpha))


# --- 3D -------------------------------------------------------------------------------------------

FOC = 1500.0


def rot(rx, ry, rz):
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def proj(ctx, p, cam=(0, 0, 0)):
    x, y, z = p[0] - cam[0], p[1] - cam[1], p[2] - cam[2]
    s = FOC * ctx.u / max(FOC * ctx.u + z, 1.0)
    return ctx.W / 2 + x * s, ctx.H / 2 + y * s


def quad(ctx, center, w, h, rx=0, ry=0, rz=0, cam=(0, 0, 0)):
    R = rot(rx, ry, rz)
    out = []
    for lx, ly in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        out.append(proj(ctx, R @ np.array([lx, ly, 0.0]) + np.array(center), cam))
    return out


def panel(c, ctx, media, center, w, h, rx=0, ry=0, rz=0, cam=(0, 0, 0), alpha=1.0, radius=None, glow=0.0,
          border=True):
    """A rounded card in 3D (center is x, y, z from the frame center, z away from the camera).
    media is a skia.Image or a callable draw(c, w, h). Returns the 4 projected corners."""
    radius = 18 * ctx.u if radius is None else radius
    q = quad(ctx, center, w, h, rx, ry, rz, cam)
    m = skia.Matrix()
    if not m.setPolyToPoly([skia.Point(0, 0), skia.Point(w, 0), skia.Point(w, h), skia.Point(0, h)],
                           [skia.Point(*p) for p in q]):
        return q
    c.save()
    c.concat(m)
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeWH(w, h), radius, radius)
    c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=col((0, 0, 0), 0.55 * alpha),
                               MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 40 * ctx.u)))
    if glow > 0:
        c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c("accent", glow * alpha), StrokeWidth=6 * ctx.u,
                                   Style=skia.Paint.kStroke_Style,
                                   MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 18 * ctx.u)))
    c.save()
    c.clipRRect(rr, True)
    c.drawRect(skia.Rect.MakeWH(w, h), skia.Paint(Color=ctx.c("surface", alpha)))
    if callable(media):
        media(c, w, h)
    elif media is not None:
        cover(c, media, 0, 0, w, h, alpha=alpha)
    c.restore()
    if border:
        c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c("paper", 0.14 * alpha),
                                   Style=skia.Paint.kStroke_Style, StrokeWidth=2 * ctx.u))
    c.restore()
    return q


def at(q, fx, fy):
    """A point inside a projected panel, by fraction of its width/height."""
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = q
    top = (lerp(x0, x1, fx), lerp(y0, y1, fx))
    bot = (lerp(x3, x2, fx), lerp(y3, y2, fx))
    return lerp(top[0], bot[0], fy), lerp(top[1], bot[1], fy)


# --- stage (the brand's motion-graphics background) -----------------------------------------------

_PART = np.random.default_rng(3).uniform([-1400, -1400, 0], [1400, 1400, 2600], (140, 3))


def stage(c, ctx, t, glow=0.35, grid=1.0, speed=120.0, horizon=0.64, parts=1.0):
    """Dark gradient, a perspective floor grid running toward the camera, drifting sparks."""
    W, H, u = ctx.W, ctx.H, ctx.u
    ink = ctx.colors["ink"]
    c.clear(col(ink))
    lift = tuple(min(255, v + 20) for v in ink)
    g = skia.Paint(AntiAlias=True)
    g.setShader(skia.GradientShader.MakeRadial(skia.Point(W / 2, H * 0.46), max(W, H) * 0.62,
                                               [col(lift), col(ink)], [0.0, 1.0]))
    c.drawRect(skia.Rect.MakeWH(W, H), g)
    if glow > 0:
        gp = skia.Paint(AntiAlias=True)
        gp.setShader(skia.GradientShader.MakeRadial(skia.Point(W / 2, H * horizon), max(W, H) * 0.45,
                                                    [ctx.c("accent", 0.10 * glow), ctx.c("accent", 0)], [0.0, 1.0]))
        c.drawRect(skia.Rect.MakeWH(W, H), gp)
    if grid > 0:
        hy, camh, f = H * horizon, 260.0 * u, 900.0 * u
        lp = skia.Paint(AntiAlias=True, StrokeWidth=1.4 * u, Style=skia.Paint.kStroke_Style)
        off = (t * speed) % 120.0
        for k in range(1, 40):
            z = k * 120.0 - off
            if z < 30:
                continue
            y = hy + f * camh / z / u
            a = 0.22 * grid * clamp(1 - z / 4200) * clamp((y - hy) / (40 * u))
            lp.setColor(ctx.c("accent", a))
            c.drawLine(0, y, W, y, lp)
        for k in range(-24, 25):
            x = k * 160.0 * u
            x0, y0 = W / 2 + f * x / 30 / u, hy + f * camh / 30 / u
            x1, y1 = W / 2 + f * x / 4200 / u, hy + f * camh / 4200 / u
            sh = skia.Paint(AntiAlias=True, StrokeWidth=1.4 * u)
            sh.setShader(skia.GradientShader.MakeLinear([skia.Point(x1, y1), skia.Point(x0, y0)],
                                                        [ctx.c("accent", 0.0), ctx.c("accent", 0.20 * grid)]))
            c.drawLine(x0, y0, x1, y1, sh)
        hz = skia.Paint(AntiAlias=True)
        hz.setShader(skia.GradientShader.MakeLinear([skia.Point(0, hy - 120 * u), skia.Point(0, hy + 60 * u)],
                                                    [col(ink, 0), col(ink, 0.95), col(ink, 0)], [0, 0.6, 1]))
        c.drawRect(skia.Rect.MakeXYWH(0, hy - 120 * u, W, 180 * u), hz)
    if parts > 0:
        pp = skia.Paint(AntiAlias=True)
        spark = tuple(min(255, int(v * 0.5 + 128)) for v in ctx.colors["accent"])
        for x, y, z in _PART:
            zz = (z - t * 160) % 2600
            sx, sy = proj(ctx, (x * u, (y + math.sin(t * 0.7 + x) * 20) * u, zz * u))
            a = parts * 0.55 * clamp(1 - zz / 2600) * clamp(zz / 200)
            pp.setColor(col(spark, a))
            c.drawCircle(sx, sy, (1.2 + 2.4 * (1 - zz / 2600)) * u, pp)


def vignette(c, ctx, a=0.5):
    p = skia.Paint(AntiAlias=True)
    p.setShader(skia.GradientShader.MakeRadial(skia.Point(ctx.W / 2, ctx.H / 2), max(ctx.W, ctx.H) * 0.72,
                                               [col((0, 0, 0), 0), col((0, 0, 0), a)], [0.55, 1.0]))
    c.drawRect(skia.Rect.MakeWH(ctx.W, ctx.H), p)


# --- small UI pieces ------------------------------------------------------------------------------

def pill(c, ctx, x, y, w, h, fill="surface", a=0.9, border=0.16, r=None, shadow=True):
    r = h / 2 if r is None else r
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)
    if shadow:
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y + 10 * ctx.u, w, h), r, r),
                    skia.Paint(AntiAlias=True, Color=col((0, 0, 0), 0.45 * a),
                               MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 22 * ctx.u)))
    c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c(fill, a)))
    if border:
        c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c("paper", border * a),
                                   Style=skia.Paint.kStroke_Style, StrokeWidth=1.5 * ctx.u))


def button(c, ctx, cx, cy, label, k=1.0, fill="accent2", size=32, a=1.0):
    """A rounded call-to-action button centered on (cx, cy), popping in with k."""
    if k <= 0:
        return
    u = ctx.u
    f = ctx.font(size * u, 700)
    w, h = tw(label, f) + 70 * u, size * 2.1 * u
    c.save()
    c.translate(cx, cy)
    c.scale(k, k)
    r = h * max(ctx.radius, 0.12) * 2 if ctx.radius < 0.5 else h / 2
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-w / 2, -h / 2, w, h), r, r),
                skia.Paint(AntiAlias=True, Color=ctx.c(fill, a)))
    text(c, label, 0, size * 0.36 * u, f, readable_on(ctx.colors[fill]), a, align="c")
    c.restore()


def readable_on(rgb):
    """Black or white, whichever reads on rgb."""
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (12, 13, 15) if lum > 150 else (255, 255, 255)


def brackets(c, ctx, x, y, w, h, a=1.0, L=34, sw=3):
    p = skia.Paint(AntiAlias=True, Color=ctx.c("accent", a), StrokeWidth=sw * ctx.u, Style=skia.Paint.kStroke_Style)
    L *= ctx.u
    for px, py, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        c.drawLine(px, py, px + sx * L, py, p)
        c.drawLine(px, py, px, py + sy * L, p)


def sheen(c, x, y, w, h, k, r=0):
    """A light sweep across a box, k = 0..1."""
    if not 0 < k < 1:
        return
    sx = lerp(x - w * 0.5, x + w * 1.5, k)
    sh = skia.Paint(AntiAlias=True)
    sh.setShader(skia.GradientShader.MakeLinear([skia.Point(sx - w * 0.15, y), skia.Point(sx + w * 0.15, y + h)],
                                                [col((255, 255, 255), 0), col((255, 255, 255), 0.35),
                                                 col((255, 255, 255), 0)], [0, 0.5, 1]))
    c.save()
    c.clipRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r), True)
    c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), sh)
    c.restore()


def logo_image(ctx, which=("mark", "icon", "wordmark")) -> skia.Image | None:
    """The brand's logo for reveals and tiles. A large square icon (an app icon: the mark as people
    see it, usually on its tile) beats a bare mark, which sites often draw in currentColor."""
    icon = ctx.kit.get("logo", {}).get("icon")
    if which[0] == "mark" and icon and not str(icon).startswith("http"):
        img = still(str(icon))
        if img is not None and img.width() >= 256 and abs(img.width() - img.height()) <= 2:
            return img
    for k in which:
        p = ctx.kit.get("logo", {}).get(k)
        if p and not str(p).startswith("http"):
            img = still(str(p))
            if img is not None:
                return img
    return None


def mark(c, ctx, cx, cy, size, alpha=1.0, tile=None):
    """The brand mark centered on (cx, cy), size px tall. With no logo file, a primary-colored tile
    with the brand's initial. tile=True puts the mark on a rounded primary tile."""
    img = logo_image(ctx)
    r = size * max(0.12, ctx.radius)
    if img is None or tile:
        rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(cx - size / 2, cy - size / 2, size, size), r, r)
        c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=col((0, 0, 0), 0.5 * alpha),
                                   MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, size * 0.08)))
        c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c("accent", alpha)))
        hl = skia.Paint(AntiAlias=True)
        hl.setShader(skia.GradientShader.MakeLinear([skia.Point(cx, cy - size / 2), skia.Point(cx, cy + size / 2)],
                                                    [col((255, 255, 255), 0.22 * alpha), col((255, 255, 255), 0)]))
        c.drawRRect(rr, hl)
        if img is None:
            f = ctx.font(size * 0.62, 900, "display")
            text(c, (ctx.name or "?")[0].upper(), cx, cy + size * 0.22, f, readable_on(ctx.colors["accent"]),
                 alpha, align="c")
            return
        contain(c, img, cx - size * 0.36, cy - size * 0.36, size * 0.72, size * 0.72, alpha)
        return
    s = size / img.height()
    contain(c, img, cx - img.width() * s / 2, cy - size / 2, img.width() * s, size, alpha)
