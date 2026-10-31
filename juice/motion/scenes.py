"""Scenes (fill the frame) and overlays (draw on top), brand-driven and layout-aware.

Every piece is an object with draw(c, ctx, lt, dur, t): lt = seconds since it started, dur = how long
it lives, t = output time (for things timed to the voice). Sizes are in design units times ctx.u,
and layouts switch on ctx.portrait, so the same script renders portrait and landscape.
"""
from __future__ import annotations

import math

import numpy as np
import skia

from .core import (Source, at, brackets, button, clamp, col, contain, cover, e_expo, e_in, e_inout, e_out,
                   fit, lerp, logo_image, mark, panel, pill, prog, readable_on, sheen, spring, stage, still, text,
                   tw, vignette)


def _media(m):
    """A Source passes through; a path to an image becomes a still."""
    if m is None or isinstance(m, Source) or callable(m):
        return m
    return still(str(m))


def _pos(ctx, pos, landscape, portrait):
    """pos if given, else the default for this orientation (portrait keeps clear of the apps' UI)."""
    return pos if pos is not None else (portrait if ctx.portrait else landscape)


def _frame(m, lt):
    if isinstance(m, Source):
        return m.get(lt)
    return m


class Piece:
    def sources(self) -> list[Source]:
        out = []
        for v in vars(self).values():
            for x in (v if isinstance(v, (list, tuple)) else [v]):
                if isinstance(x, Source):
                    out.append(x)
                elif isinstance(x, Piece):
                    out += x.sources()
                elif isinstance(x, dict):
                    out += [y for y in x.values() if isinstance(y, Source)]
        return out

    def draw(self, c, ctx, lt, dur, t):
        raise NotImplementedError


# --- scenes ---------------------------------------------------------------------------------------

class Footage(Piece):
    """Full-frame footage with a slow push (zoom from -> to), optional blur/dim for text over it."""

    def __init__(self, src: Source, zoom=(1.0, 1.05), cx=0.5, cy=0.45, blur=0.0, dim=0.0, vignette=0.4):
        self.src, self.zoom, self.cx, self.cy, self.blur, self.dim, self.vig = src, zoom, cx, cy, blur, dim, vignette

    def draw(self, c, ctx, lt, dur, t):
        z = self.zoom if isinstance(self.zoom, (int, float)) else lerp(*self.zoom, e_inout(clamp(lt / max(dur, 0.01))))
        cover(c, self.src.get(lt), 0, 0, ctx.W, ctx.H, zoom=z, cx=self.cx, cy=self.cy, blur=self.blur * ctx.u,
              dim=self.dim, dim_rgb=ctx.colors["ink"])
        if self.vig:
            vignette(c, ctx, self.vig)


class Stage(Piece):
    """Just the brand's motion background (grid floor, glow, sparks)."""

    def __init__(self, glow=0.5, grid=1.0, speed=100):
        self.glow, self.grid, self.speed = glow, grid, speed

    def draw(self, c, ctx, lt, dur, t):
        stage(c, ctx, t, glow=self.glow, grid=self.grid, speed=self.speed)


class LogoReveal(Piece):
    """The mark flies in out of a glow with sparks and a shockwave, springs onto its tile, then the
    name wipes on beside it (below it in portrait), with an optional badge and a typed tagline."""

    def __init__(self, badge: str | None = None, tagline: str | None = None, tile: bool | None = None):
        self.badge, self.tagline, self.tile = badge, tagline, tile

    def draw(self, c, ctx, lt, dur, t):
        W, H, u = ctx.W, ctx.H, ctx.u
        stage(c, ctx, t, glow=0.8 + 0.6 * math.exp(-lt * 3), grid=e_out(prog(lt, 0.0, 0.8)), speed=260)
        bp = prog(lt, 0.0, 0.9)
        if 0 < bp < 1:  # spark burst
            sp = skia.Paint(AntiAlias=True, StrokeCap=skia.Paint.kRound_Cap, StrokeWidth=3 * u)
            r0 = np.random.default_rng(11)
            for _ in range(46):
                ang, v, L = r0.uniform(0, 2 * math.pi), r0.uniform(300, 1100) * u, r0.uniform(20, 90) * u
                d = v * e_out(bp)
                x0, y0 = W / 2 + math.cos(ang) * d, H / 2 + math.sin(ang) * d
                sp.setColor(ctx.c("accent", (1 - bp) * 0.9))
                c.drawLine(x0, y0, x0 + math.cos(ang) * L * (1 - bp), y0 + math.sin(ang) * L * (1 - bp), sp)
        for tt, g in ((0.0, 1.0), (0.82, 0.8)):  # shockwaves
            s_ = prog(lt, tt, 0.7)
            if 0 < s_ < 1:
                c.drawCircle(W / 2, H / 2, (60 + 900 * e_out(s_)) * u,
                             skia.Paint(AntiAlias=True, Color=ctx.c("accent", 0.5 * g * (1 - s_)),
                                        Style=skia.Paint.kStroke_Style, StrokeWidth=(6 * (1 - s_) + 1) * u))
        name = ctx.name
        fo = ctx.font((168 if not ctx.portrait else 150) * u, None, "display")
        fo = fit(name, fo, (W - 330 * u - 300 * u) if not ctx.portrait else W * 0.86)
        wordw = tw(name, fo)
        msize = (330 if not ctx.portrait else 360) * u
        fly = e_expo(prog(lt, 0.0, 0.8))
        size = lerp(msize * 4.2, msize, fly)
        settle = e_expo(prog(lt, 0.0, 1.0))
        k_tile = spring(prog(lt, 0.72, 0.55))
        slide = e_inout(prog(lt, 1.02, 0.45))
        if ctx.portrait:
            mx, my = W / 2, lerp(H / 2, H / 2 - 230 * u, slide)
            wx, wy = W / 2 - wordw / 2, my + msize / 2 + 190 * u
        else:
            group = msize + 50 * u + wordw
            mx, my = lerp(W / 2, W / 2 - group / 2 + msize / 2, slide), H / 2 - 20 * u
            wx, wy = mx + msize / 2 + 50 * u, my + 60 * u
        out = e_in(prog(lt, dur - 0.2, 0.2))
        c.save()
        c.translate(W / 2, H / 2)
        c.scale(1 + out * 0.25, 1 + out * 0.25)
        c.translate(-W / 2, -H / 2)
        # Mark: a glowing silhouette that spins in, then the real mark on its tile springs in.
        c.save()
        c.translate(mx, my)
        c.rotate((1 - settle) * 140)
        c.translate(-mx, -my)
        glow = 1 - clamp(k_tile * 1.3)
        img = logo_image(ctx)
        if glow > 0 and img is not None:
            gl = skia.Paint(AntiAlias=True, ColorFilter=skia.ColorFilters.Blend(ctx.c("accent"), skia.BlendMode.kSrcIn),
                            ImageFilter=skia.ImageFilters.Blur(18 * u, 18 * u), Alphaf=glow)
            s = size / img.height()
            c.drawImageRect(img, skia.Rect.MakeXYWH(mx - img.width() * s / 2, my - size / 2, img.width() * s, size),
                            skia.SamplingOptions(), gl)
        elif glow > 0:
            c.drawCircle(mx, my, size * 0.4, skia.Paint(AntiAlias=True, Color=ctx.c("accent", glow * 0.8),
                                                        MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 30 * u)))
        if k_tile > 0:
            mark(c, ctx, mx, my, size * lerp(0.6, 1.0, k_tile), alpha=clamp(k_tile * 1.6), tile=self.tile)
        c.restore()
        wp = e_expo(prog(lt, 1.12, 0.55))
        if wp > 0:
            c.save()
            c.clipRect(skia.Rect.MakeXYWH(wx - 10 * u, wy - 220 * u, (wordw + 40 * u) * wp, 440 * u))
            text(c, name, wx + (1 - wp) * -60 * u, wy, fo, ctx.colors["paper"])
            c.restore()
            bk = spring(prog(lt, 1.38, 0.5))
            if self.badge and bk > 0:
                fb = ctx.font(40 * u, 700)
                bw, bh = tw(self.badge, fb) + 40 * u, 56 * u
                bx, by = (wx + wordw + 22 * u, wy - fo.getSize() * 0.55 - bh / 2) if not ctx.portrait else \
                    (W / 2 - bw / 2, wy + 50 * u)
                c.save()
                c.translate(bx + bw / 2, by + bh / 2)
                c.scale(bk, bk)
                c.translate(-(bx + bw / 2), -(by + bh / 2))
                badge_bg = tuple(int(v * 0.3 + i * 0.7) for v, i in zip(ctx.colors["accent2"], ctx.colors["ink"]))
                c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(bx, by, bw, bh), bh / 2, bh / 2),
                            skia.Paint(AntiAlias=True, Color=col(badge_bg)))
                text(c, self.badge, bx + bw / 2, by + bh * 0.72, fb, ctx.colors["accent2"], align="c")
                c.restore()
            if self.tagline:
                n = int(len(self.tagline) * clamp((lt - 1.40) / 0.45))
                ft = fit(self.tagline, ctx.font(32 * u, 600), W * 0.86)
                ty = wy + (90 if not ctx.portrait else 170) * u
                if n > 0:
                    if ctx.portrait:
                        text(c, self.tagline[:n], W / 2 - tw(self.tagline, ft, 2 * u) / 2, ty, ft, ctx.colors["muted"],
                             track=2 * u)
                    else:
                        text(c, self.tagline[:n], wx + 4 * u, ty, ft, ctx.colors["muted"], track=2 * u)
        c.restore()


class Panels(Piece):
    """Product screens on floating 3D cards with a drifting camera, callouts pinned to them, and an
    optional status chip. callouts: [(panel_index, fx, fy, label)] with fx/fy fractions of the card."""

    def __init__(self, media: list, callouts=(), chip: str | None = None):
        self.media = [_media(m) for m in media][:3]
        self.callouts, self.chip = list(callouts), chip

    def layout(self, ctx, n):
        u = ctx.u
        if ctx.portrait:
            return [((0, -380 * u, 0), 900 * u, 506 * u, 0.30, 0.10), ((0, 300 * u, -40 * u), 960 * u, 540 * u, -0.25, -0.08),
                    ((0, 860 * u, 60 * u), 820 * u, 461 * u, 0.2, 0.06)][:n] if n > 1 else \
                [((0, 0, 0), 980 * u, 551 * u, 0.0, 0.0)]
        if n == 1:
            return [((0, 0, 0), 1400 * u, 788 * u, 0.04, -0.08)]
        return [((-560 * u, 10 * u, 300 * u), 1000 * u, 628 * u, 0.05, 0.38),
                ((330 * u, -20 * u, -40 * u), 1160 * u, 652 * u, 0.04, -0.16),
                ((620 * u, 330 * u, -140 * u), 700 * u, 394 * u, 0.03, -0.2)][:n]

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        stage(c, ctx, t, glow=0.5, grid=0.8, speed=90)
        cam = (lerp(-40, 40, lt / max(dur, 0.1)) * u, lerp(-10, 10, lt / max(dur, 0.1)) * u,
               lerp(0, -90, e_inout(clamp(lt / max(dur, 0.1)))) * u)
        outk = e_in(prog(lt, dur - 0.28, 0.28))
        qs = []
        for i, (m, (ctr, w, h, rx, ry)) in enumerate(zip(self.media, self.layout(ctx, len(self.media)))):
            a = e_expo(prog(lt, 0.12 * i, 0.6))
            side = -1 if i % 2 == 0 else 1
            x, y, z = ctr
            if ctx.portrait:
                pos = (x + (1 - a) * side * 500 * u, y, z + (1 - a) * 600 * u - outk * 800 * u)
                rot = (rx * (1 - 0.5 * a) + 0.04, ry + (1 - a) * side * 0.3)
            else:
                pos = (x + (1 - a) * side * 400 * u, y, z + (1 - a) * 700 * u - outk * 800 * u)
                rot = (rx, ry - (1 - a) * side * 0.4)
            qs.append(panel(c, ctx, _frame(m, lt), pos, w, h, rx=rot[0], ry=rot[1], cam=cam, alpha=a,
                            glow=0.35 if i == len(self.media) - 1 else 0))
        if outk < 0.3:
            for k, (pi, fx, fy, label) in enumerate(self.callouts):
                if pi < len(qs):
                    ax, ay = at(qs[pi], fx, fy)
                    dx = (-1 if ax > ctx.W / 2 else 1) * (150 if not ctx.portrait else 90) * u
                    dy = (-1 if ay > ctx.H / 2 else 1) * -170 * u * (1 if k % 2 == 0 else -0.6)
                    Callout((ax / ctx.W, ay / ctx.H), label, dx=dx, dy=dy).draw(c, ctx, lt - 0.35 - 0.4 * k, dur, t)
        ck = e_expo(prog(lt, 0.08, 0.5)) * (1 - outk)
        if self.chip and ck > 0:
            Chip(self.chip).draw(c, ctx, lt - 0.08, dur, t)


class Linked(Piece):
    """Two screens joined by an animated link with a label on it, e.g. an AI agent driving an app:
    left (the controller), right (the thing being controlled)."""

    def __init__(self, left, right, label="MCP", left_label=None, title=None, sub=None):
        self.left, self.right = _media(left), _media(right)
        self.label, self.left_label, self.title, self.sub = label, left_label, title, sub

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        stage(c, ctx, t, glow=0.6, grid=0.7, speed=100)
        cam = (lerp(30, -30, lt / max(dur, 0.1)) * u, 0, lerp(0, -70, e_inout(clamp(lt / max(dur, 0.1)))) * u)
        a1, a2 = e_expo(prog(lt, 0.0, 0.5)), e_expo(prog(lt, 0.08, 0.5))
        if ctx.portrait:
            qc = panel(c, ctx, _frame(self.left, lt), (0, -150 * u, 200 * u + (1 - a1) * 600 * u), 700 * u, 560 * u,
                       rx=-0.12, ry=0.1, cam=cam, alpha=a1)
            qs = panel(c, ctx, _frame(self.right, lt), (0, 480 * u, -40 * u + (1 - a2) * 500 * u), 940 * u, 528 * u,
                       rx=0.1, ry=-0.06, cam=cam, alpha=a2, glow=0.45)
            p0 = ((qc[3][0] + qc[2][0]) / 2, (qc[3][1] + qc[2][1]) / 2)
            p1 = ((qs[0][0] + qs[1][0]) / 2, (qs[0][1] + qs[1][1]) / 2)
            bend = (160 * u, 0)
        else:
            qc = panel(c, ctx, _frame(self.left, lt), (-640 * u, 190 * u, 200 * u + (1 - a1) * 600 * u), 520 * u, 610 * u,
                       rx=0.04, ry=0.30, cam=cam, alpha=a1)
            qs = panel(c, ctx, _frame(self.right, lt), (300 * u + (1 - a2) * 400 * u, 200 * u, -40 * u + (1 - a2) * 500 * u),
                       1060 * u, 596 * u, rx=0.03, ry=-0.12, cam=cam, alpha=a2, glow=0.45)
            p0 = ((qc[1][0] + qc[2][0]) / 2, (qc[1][1] + qc[2][1]) / 2)
            p1 = ((qs[0][0] + qs[3][0]) / 2, (qs[0][1] + qs[3][1]) / 2)
            bend = (120 * u, -160 * u)
        lk = e_expo(prog(lt, 0.3, 0.5))
        if lk > 0:
            path = skia.Path()
            path.moveTo(*p0)
            path.cubicTo(p0[0] + bend[0], p0[1] + bend[1], p1[0] - bend[0], p1[1] + bend[1], *p1)
            meas = skia.PathMeasure(path, False)
            L = meas.getLength()
            seg = skia.Path()
            meas.getSegment(0, L * lk, seg, True)
            c.drawPath(seg, skia.Paint(AntiAlias=True, Color=ctx.c("accent", 0.35), StrokeWidth=10 * u,
                                       Style=skia.Paint.kStroke_Style,
                                       MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 8 * u)))
            c.drawPath(seg, skia.Paint(AntiAlias=True, Color=ctx.c("accent"), StrokeWidth=3 * u,
                                       Style=skia.Paint.kStroke_Style))
            for k in range(4):
                pos, _ = meas.getPosTan(((lt * 0.9 + k / 4) % 1.0) * L * lk)
                c.drawCircle(pos.x(), pos.y(), 6 * u, skia.Paint(AntiAlias=True, Color=ctx.c("paper")))
            mid, _ = meas.getPosTan(L * 0.5)
            mk = spring(prog(lt, 0.55, 0.5))
            if mk > 0 and self.label:
                fo = ctx.font(26 * u, 800)
                w = tw(self.label, fo, 2 * u) + 40 * u
                c.save()
                c.translate(mid.x(), mid.y())
                c.scale(mk, mk)
                pill(c, ctx, -w / 2, -24 * u, w, 48 * u, fill="accent", a=1.0, border=0)
                text(c, self.label, 0, 10 * u, fo, readable_on(ctx.colors["accent"]), align="c", track=2 * u)
                c.restore()
            if self.left_label:
                text(c, self.left_label.upper(), qc[0][0], qc[0][1] - 20 * u, ctx.font(22 * u, 700), ctx.colors["muted"],
                     lk, track=3 * u)
        if self.title:
            y = 150 * u if not ctx.portrait else 230 * u
            Slam([self.title], pos=(0.5, y / ctx.H), size=150 if not ctx.portrait else 120).draw(c, ctx, lt - 0.1, dur, t)
        if self.sub:
            sk = e_expo(prog(lt, 0.55, 0.4))
            y = (290 if not ctx.portrait else 450) * u
            if sk > 0:
                f = fit(self.sub, ctx.font(36 * u, 500), ctx.W * 0.88)
                text(c, self.sub, ctx.W / 2, y + (1 - sk) * 20 * u, f, ctx.colors["muted"], sk, align="c")


class Card:
    """One feature card: a title over media (Source / image), or a drawn card when media is None
    (title + body + optional url field and button). rec=True adds a recording indicator."""

    def __init__(self, title, media=None, body=None, url=None, button=None, rec=False):
        self.title, self.media, self.body, self.url, self.button, self.rec = title, _media(media), body, url, button, rec


class FeatureGrid(Piece):
    """A headline that springs in word by word over a grid of feature cards (2x2; 1 column of up to
    4 in portrait), each card flying in on its own beat with a light sweep."""

    def __init__(self, title, cards: list[Card], kicker=None):
        self.title, self.cards, self.kicker = title, cards[:4], kicker

    def sources(self):
        return [cd.media for cd in self.cards if isinstance(cd.media, Source)]

    def _drawn(self, ctx, cd, lt):
        u = ctx.u

        def draw(c, w, h):
            g = skia.Paint(AntiAlias=True)
            g.setShader(skia.GradientShader.MakeRadial(skia.Point(w * 0.8, h * 0.2), w * 0.7,
                                                       [ctx.c("accent", 0.12), ctx.c("accent", 0)], [0, 1]))
            c.drawRect(skia.Rect.MakeWH(w, h), g)
            pad = 48 * u
            f1 = fit(cd.title, ctx.font(38 * u, 700), w - 2 * pad)
            text(c, cd.title, pad, 88 * u, f1, ctx.colors["paper"])
            y = 134 * u
            compact = h < 330 * u
            if compact:
                y = 112 * u
            if cd.body and not compact:
                f2 = fit(cd.body, ctx.font(25 * u, 500), w - 2 * pad)
                text(c, cd.body, pad, y, f2, ctx.colors["muted"])
                y += 34 * u
            if cd.url:
                box = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(pad, y, w - 2 * pad, 70 * u), 14 * u, 14 * u)
                c.drawRRect(box, skia.Paint(AntiAlias=True, Color=ctx.c("ink")))
                n = int(len(cd.url) * clamp((lt - 0.9) / 0.9))
                fu = fit(cd.url, ctx.font(30 * u, 500), w - 2 * pad - 60 * u)
                wu = text(c, cd.url[:n], pad + 26 * u, y + 46 * u, fu, ctx.colors["paper"])
                if int(lt * 4) % 2 == 0:
                    c.drawRect(skia.Rect.MakeXYWH(pad + 30 * u + wu, y + 18 * u, 3 * u, 36 * u),
                               skia.Paint(Color=ctx.c("accent")))
                y += (84 if compact else 94) * u
            if cd.button:
                bk = spring(prog(lt, 1.9, 0.5))
                f3 = ctx.font(26 * u, 600)
                bw = tw(cd.button, f3) + 60 * u
                button(c, ctx, pad + bw / 2, y + 28 * u, cd.button, k=bk, size=26)
        return draw

    def draw(self, c, ctx, lt, dur, t):
        W, H, u = ctx.W, ctx.H, ctx.u
        stage(c, ctx, t, glow=0.4, grid=0.6, speed=80, horizon=0.9)
        x0 = (120 if not ctx.portrait else 70) * u
        y_title = (196 if not ctx.portrait else 300) * u
        if self.kicker:
            hk = e_expo(prog(lt, 0.0, 0.5))
            text(c, self.kicker.upper(), x0, y_title - 78 * u, ctx.font(24 * u, 700), ctx.colors["accent"], hk, track=4 * u)
        fo = ctx.font((68 if not ctx.portrait else 64) * u, 800, "display")
        Headline(self.title, pos=(x0 / W, y_title / H), size=fo.getSize() / u, align="l",
                 max_w=W - 2 * x0).draw(c, ctx, lt - 0.06, dur, t)
        n = len(self.cards)
        if ctx.portrait:
            cols, cw, ch, gap = 1, W - 2 * x0, 262 * u, 24 * u
            top = y_title + 70 * u
        else:
            cols, cw, ch, gap = 2, 820 * u, 372 * u, 40 * u
            top = 262 * u
        rows = math.ceil(n / cols)
        gx = (W - cols * cw - (cols - 1) * gap) / 2
        push = lerp(1.0, 1.035, e_inout(clamp(lt / max(dur, 0.1))))
        c.save()
        c.translate(W / 2, top + rows * ch / 2)
        c.scale(push, push)
        c.translate(-W / 2, -(top + rows * ch / 2))
        r = 22 * u
        for k, cd in enumerate(self.cards):
            cx = gx + (k % cols) * (cw + gap) + cw / 2
            cy = top + (k // cols) * (ch + gap) + ch / 2
            p = prog(lt, 0.15 + k * 0.25, 0.7)
            s, a = spring(p), clamp(p * 3)
            if a <= 0:
                continue

            def media(cc, w, h, cd=cd, k=k):
                if cd.media is None:
                    self._drawn(ctx, cd, lt)(cc, w, h)
                else:
                    cover(cc, _frame(cd.media, lt), 0, 0, w, h, zoom=1.02 + 0.04 * clamp(lt / max(dur, 0.1)))
                    gr = skia.Paint(AntiAlias=True)
                    gr.setShader(skia.GradientShader.MakeLinear([skia.Point(0, h * 0.35), skia.Point(0, h)],
                                                                [ctx.c("ink", 0), ctx.c("ink", 0.92)]))
                    cc.drawRect(skia.Rect.MakeWH(w, h), gr)
                    f = fit(cd.title, ctx.font(40 * u, 700), w - 88 * u)
                    text(cc, cd.title, 44 * u, h - 46 * u, f, ctx.colors["paper"])
                num = f"{k + 1:02d}"
                text(cc, num, w - 44 * u, 62 * u if cd.media is not None else h - 40 * u, ctx.font(28 * u, 800),
                     ctx.colors["accent"], align="r", track=2 * u)
                if cd.rec:
                    on = int(lt * 2.5) % 2 == 0
                    cc.drawCircle(52 * u, 52 * u, 10 * u, skia.Paint(AntiAlias=True, Color=col((235, 64, 52), 1 if on else 0.35)))
                    sec = 3 * 60 + 12 + lt
                    text(cc, f"REC {int(sec // 60):02d}:{sec % 60:05.2f}", 74 * u, 62 * u, ctx.font(24 * u, 700),
                         ctx.colors["paper"], 0.95, track=1 * u)
            panel(c, ctx, media, (cx - W / 2, cy - H / 2 + (1 - s) * 140 * u, (1 - s) * 500 * u), cw, ch,
                  rx=(1 - s) * 0.5, ry=((k % 2) - 0.5) * 0.06 * (1 - s), alpha=a, radius=r)
            sheen(c, cx - cw / 2, cy - ch / 2, cw, ch, prog(lt, 0.7 + k * 0.25, 0.6), r)
        c.restore()


class Notification(Piece):
    """A message card over blurred footage (e.g. "Your account is ready."), with the brand mark,
    a sender line and a button that pops at `button_at` seconds into the scene."""

    def __init__(self, src: Source | None, title, body=None, button_label=None, sender=None, to="to: you",
                 button_at=1.4):
        self.src, self.title, self.body, self.button_label = src, title, body, button_label
        self.sender, self.to, self.button_at = sender, to, button_at

    def draw(self, c, ctx, lt, dur, t):
        W, H, u = ctx.W, ctx.H, ctx.u
        if self.src is not None:
            cover(c, self.src.get(lt), 0, 0, W, H, zoom=1.12, cy=0.45, blur=26 * u, dim=0.55, dim_rgb=ctx.colors["ink"])
        else:
            stage(c, ctx, t, glow=0.4, grid=0.5)
        k, a = spring(prog(lt, 0.18, 0.7)), clamp(prog(lt, 0.18, 0.25))
        cw = min(980 * u, W - 80 * u)
        ch = 440 * u
        cx, cy = W / 2 - cw / 2, H / 2 - ch / 2 - 20 * u + (1 - k) * -160 * u
        out = e_in(prog(lt, dur - 0.25, 0.25))
        c.save()
        c.translate(W / 2, H / 2)
        c.scale(1 - 0.08 * out, 1 - 0.08 * out)
        c.translate(-W / 2, -H / 2)
        a *= 1 - out
        pill(c, ctx, cx, cy, cw, ch, fill="surface", a=a * 0.96, border=0.14, r=34 * u)
        mark(c, ctx, cx + 92 * u, cy + 96 * u, 84 * u, alpha=a)
        text(c, self.sender or ctx.name, cx + 160 * u, cy + 86 * u, ctx.font(32 * u, 700), ctx.colors["paper"], a)
        text(c, "now", cx + cw - 56 * u, cy + 86 * u, ctx.font(26 * u, 500), ctx.colors["muted"], a, align="r")
        if self.to:
            text(c, self.to, cx + 160 * u, cy + 124 * u, ctx.font(26 * u, 500), ctx.colors["muted"], a)
        c.drawRect(skia.Rect.MakeXYWH(cx + 48 * u, cy + 172 * u, cw - 96 * u, 1.5 * u), skia.Paint(Color=ctx.c("paper", 0.10 * a)))
        ft = fit(self.title, ctx.font(58 * u, 800, "display"), cw - 112 * u)
        text(c, self.title, cx + 56 * u, cy + 252 * u, ft, ctx.colors["paper"], a)
        if self.body:
            fb = fit(self.body, ctx.font(28 * u, 500), cw - 112 * u)
            text(c, self.body, cx + 56 * u, cy + 304 * u, fb, ctx.colors["muted"], a)
        if self.button_label:
            bk = spring(prog(lt, self.button_at, 0.5))
            fb = ctx.font(28 * u, 700)
            bw = tw(self.button_label, fb) + 70 * u
            pulse = 1 + 0.06 * math.exp(-max(0, lt - self.button_at - 0.4) * 6) * (lt > self.button_at + 0.4)
            button(c, ctx, cx + 56 * u + bw / 2, cy + 374 * u, self.button_label, k=bk * pulse, size=28, a=a)
        c.restore()


class EndCard(Piece):
    """Mark + name (+ badge), a line with the link highlighted, and a small sub line."""

    def __init__(self, line=None, url=None, sub=None, badge=None):
        self.line, self.url, self.sub, self.badge = line, url, sub, badge

    def draw(self, c, ctx, lt, dur, t):
        W, H, u = ctx.W, ctx.H, ctx.u
        stage(c, ctx, t, glow=0.9 + 0.8 * math.exp(-lt * 2.5), grid=1.0, speed=60)
        k = spring(prog(lt, 0.0, 0.8))
        fo = fit(ctx.name, ctx.font(150 * u, None, "display"), W * (0.55 if not ctx.portrait else 0.86))
        wordw = tw(ctx.name, fo)
        ms = (270 if not ctx.portrait else 300) * u
        if ctx.portrait:
            mx, my = W / 2, H / 2 - 300 * u
            wx, wy = W / 2 - wordw / 2, my + ms / 2 + 170 * u
            y_line = wy + 170 * u
        else:
            gw = ms + 46 * u + wordw
            mx, my = W / 2 - gw / 2 + ms / 2, H / 2 - 70 * u
            wx, wy = mx + ms / 2 + 46 * u, my + 54 * u
            y_line = my + 250 * u
        mark(c, ctx, mx, my, ms * k, alpha=clamp(k * 2))
        sheen(c, mx - ms / 2, my - ms / 2, ms, ms, prog(lt, 0.7, 0.7), ms * max(0.12, ctx.radius))
        wp = e_expo(prog(lt, 0.15, 0.6))
        c.save()
        c.clipRect(skia.Rect.MakeXYWH(wx - 10 * u, wy - 200 * u, (wordw + 40 * u) * wp, 400 * u))
        text(c, ctx.name, wx + (1 - wp) * -50 * u, wy, fo, ctx.colors["paper"])
        c.restore()
        bk = spring(prog(lt, 0.45, 0.5))
        if self.badge and bk > 0:
            fb = ctx.font(36 * u, 700)
            bw, bh = tw(self.badge, fb) + 36 * u, 50 * u
            bx, by = (wx + wordw + 18 * u, wy - fo.getSize() * 0.55 - bh / 2) if not ctx.portrait else \
                (W / 2 - bw / 2, wy + 40 * u)
            if ctx.portrait:
                y_line += 60 * u
            c.save()
            c.translate(bx + bw / 2, by + bh / 2)
            c.scale(bk, bk)
            c.translate(-(bx + bw / 2), -(by + bh / 2))
            badge_bg = tuple(int(v * 0.3 + i * 0.7) for v, i in zip(ctx.colors["accent2"], ctx.colors["ink"]))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(bx, by, bw, bh), bh / 2, bh / 2),
                        skia.Paint(AntiAlias=True, Color=col(badge_bg)))
            text(c, self.badge, bx + bw / 2, by + bh * 0.72, fb, ctx.colors["accent2"], align="c")
            c.restore()
        uk = e_expo(prog(lt, 0.55, 0.6))
        url = self.url or (ctx.kit.get("url") or "").replace("https://", "").replace("http://", "").rstrip("/")
        s1 = (self.line + " ") if self.line else ""
        if uk > 0 and (s1 or url):
            f1 = fit(s1 + url, ctx.font(46 * u, 600), W * 0.88)
            tot = tw(s1 + url, f1)
            xx, yy = W / 2 - tot / 2, y_line + (1 - uk) * 24 * u
            text(c, s1, xx, yy, f1, ctx.colors["paper"], uk * 0.85)
            text(c, url, xx + f1.measureText(s1), yy, f1, ctx.colors["accent"], uk)
            c.drawRect(skia.Rect.MakeXYWH(xx + f1.measureText(s1), yy + 14 * u,
                                          f1.measureText(url) * e_expo(prog(lt, 0.8, 0.6)), 3 * u),
                       skia.Paint(Color=ctx.c("accent", uk)))
        tk = e_expo(prog(lt, 0.9, 0.6))
        if self.sub and tk > 0:
            fs = fit(self.sub.upper(), ctx.font(24 * u, 700), W * 0.88)
            text(c, self.sub.upper(), W / 2, y_line + 80 * u, fs, ctx.colors["muted"], tk, align="c", track=4 * u)


# --- overlays -------------------------------------------------------------------------------------

class Slam(Piece):
    """Giant words: each letter rises out of a blur with an overshoot; an accent bar wipes under the
    last line. pos = (x, y) fractions of the frame (the block's center, or left edge for align="l")."""

    def __init__(self, lines, pos=(0.5, 0.5), size=200, align="c", out=None, accent=True, color="paper"):
        self.lines = [lines] if isinstance(lines, str) else list(lines)
        self.pos, self.size, self.align, self.out, self.accent, self.color = pos, size, align, out, accent, color

    def draw(self, c, ctx, lt, dur, t):
        if lt < 0:
            return
        u = ctx.u
        fo = ctx.font(self.size * u, 900, "display")
        width = max(ctx.W * 0.9 - (ctx.W * self.pos[0] if self.align == "l" else 0), 100)
        fo = min((fit(ln, fo, width) for ln in self.lines), key=lambda f: f.getSize())
        size = fo.getSize()
        lh = size * 0.98
        cx, cy = ctx.W * self.pos[0], ctx.H * self.pos[1]
        y0 = cy - (len(self.lines) - 1) * lh / 2 + size * 0.36
        out_t = self.out if self.out is not None else dur - 0.18
        fade = 1.0 - e_in(prog(lt, out_t, 0.18))
        for li, line in enumerate(self.lines):
            track = lerp(-size * 0.02, 0, e_out(prog(lt, 0, 1.2)))
            w = tw(line, fo, track)
            x = cx - w / 2 if self.align == "c" else cx if self.align == "l" else cx - w
            bx, y = x, y0 + li * lh
            for k, ch in enumerate(line):
                p = prog(lt, li * 0.07 + k * 0.018, 0.42)
                if p > 0:
                    c.save()
                    c.translate(x, y + (1 - e_expo(p)) * size * 0.45)
                    text(c, ch, 0, 0, fo, ctx.colors[self.color], clamp(p * 2.5) * fade, blur=(1 - e_out(p)) * 18 * u)
                    c.restore()
                x += fo.measureText(ch) + track
            if self.accent and li == len(self.lines) - 1:
                ap = e_expo(prog(lt, 0.25, 0.5))
                c.drawRect(skia.Rect.MakeXYWH(bx, y + size * 0.16, w * ap, size * 0.06), skia.Paint(Color=ctx.c("accent", fade)))


class Headline(Piece):
    """A line that springs in word by word. highlight: {word: color role}."""

    def __init__(self, text_, pos=None, size=92, align="l", highlight=None, max_w=None, weight=800):
        self.text, self.pos, self.size, self.align = text_, pos, size, align
        self.highlight, self.max_w, self.weight = highlight or {}, max_w, weight

    def draw(self, c, ctx, lt, dur, t):
        if lt < 0:
            return
        u = ctx.u
        fo = fit(self.text, ctx.font(self.size * u, self.weight, "display"), self.max_w or ctx.W * 0.9)
        words = self.text.split()
        total = fo.measureText(self.text)
        pos = _pos(ctx, self.pos, (0.05, 0.86), (0.07, 0.72))
        x = ctx.W * pos[0] - (total / 2 if self.align == "c" else total if self.align == "r" else 0)
        y = ctx.H * pos[1]
        out = e_in(prog(lt, dur - 0.2, 0.2))
        for i, wd in enumerate(words):
            q = spring(prog(lt, i * 0.12, 0.5))
            c.save()
            c.translate(x, y + (1 - q) * 40 * u - out * 30 * u)
            text(c, wd, 0, 0, fo, ctx.colors[self.highlight.get(wd, "paper")], clamp(q * 2) * (1 - out))
            c.restore()
            x += fo.measureText(wd + " ")


class Words(Piece):
    """Spoken words appearing as they're said: words = [(word, output_time)], cleared at `until`."""

    def __init__(self, words, pos=None, size=58, until=None, align="l"):
        self.words, self.pos, self.size, self.until, self.align = words, pos, size, until, align

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        fo = ctx.font(self.size * u, 700)
        line = " ".join(w for w, _ in self.words)
        fo = fit(line, fo, ctx.W * 0.88)
        pos = _pos(ctx, self.pos, (0.05, 0.89), (0.07, 0.72))
        x = ctx.W * pos[0] - (fo.measureText(line) / 2 if self.align == "c" else 0)
        y = ctx.H * pos[1]
        gone = 1 - e_in(prog(t, self.until - 0.1, 0.15)) if self.until else 1
        for wd, st in self.words:
            q = e_expo(prog(t, st - 0.04, 0.3)) * gone
            if q > 0:
                text(c, wd, x, y + (1 - q) * 24 * u, fo, ctx.colors["paper"], q)
            x += fo.measureText(wd + " ")


class Chip(Piece):
    """A small status pill with a colored dot, e.g. NOW ONBOARDING · BETA."""

    def __init__(self, label, pos=None, dot="accent2"):
        self.label, self.pos, self.dot = label, pos, dot

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        k = e_expo(prog(lt, 0, 0.5)) * (1 - e_in(prog(lt, dur - 0.2, 0.2)))
        if k <= 0:
            return
        fo = ctx.font(22 * u, 700)
        w, h = tw(self.label, fo, 2.5 * u) + 58 * u, 46 * u
        x, y = (ctx.W * self.pos[0], ctx.H * self.pos[1]) if self.pos else \
            ((96 if not ctx.portrait else 60) * u, (90 if not ctx.portrait else 220) * u)
        x -= (1 - k) * 60 * u
        pill(c, ctx, x, y, w, h, a=k, border=0.12)
        c.drawCircle(x + 24 * u, y + h / 2, 5.5 * u, skia.Paint(AntiAlias=True, Color=ctx.c(self.dot, k)))
        text(c, self.label, x + 40 * u, y + h / 2 + 8 * u, fo, ctx.colors["paper"], k * 0.9, track=2.5 * u)


class Callout(Piece):
    """A pulsing dot on a point, an elbow leader that draws on, and a label pill.
    anchor = (x, y) fractions of the frame; dx, dy = where the label sits, in px from the anchor."""

    def __init__(self, anchor, label, dx=150, dy=-160):
        self.anchor, self.label, self.dx, self.dy = anchor, label, dx, dy

    def draw(self, c, ctx, lt, dur, t):
        if lt <= 0:
            return
        u = ctx.u
        ax, ay = ctx.W * self.anchor[0], ctx.H * self.anchor[1]
        k = e_out(prog(lt, 0, 0.5))
        c.drawCircle(ax, ay, 7 * u * k, skia.Paint(AntiAlias=True, Color=ctx.c("accent")))
        ring = (lt * 1.4) % 1.0
        c.drawCircle(ax, ay, (7 + 26 * ring) * u, skia.Paint(AntiAlias=True, Color=ctx.c("accent", 0.6 * (1 - ring)),
                                                            Style=skia.Paint.kStroke_Style, StrokeWidth=2 * u))
        ex, ey = ax + self.dx, ay + self.dy
        mx, my = ax + self.dx * 0.35, ey
        lp = e_expo(prog(lt, 0.05, 0.35))
        L1, L2 = math.dist((ax, ay), (mx, my)), math.dist((mx, my), (ex, ey))
        Lt = (L1 + L2) * lp
        paint = skia.Paint(AntiAlias=True, Color=ctx.c("accent", 0.9), StrokeWidth=2.2 * u, Style=skia.Paint.kStroke_Style)
        if Lt <= L1:
            f = Lt / max(L1, 1)
            c.drawLine(ax, ay, lerp(ax, mx, f), lerp(ay, my, f), paint)
        else:
            c.drawLine(ax, ay, mx, my, paint)
            f = (Lt - L1) / max(L2, 1)
            c.drawLine(mx, my, lerp(mx, ex, f), lerp(my, ey, f), paint)
        lk = spring(prog(lt, 0.22, 0.6))
        if lk > 0:
            fo = ctx.font(30 * u, 600)
            w, h = tw(self.label, fo) + 64 * u, 62 * u
            x = ex if self.dx > 0 else ex - w
            x = clamp(x, 20 * u, ctx.W - w - 20 * u)
            pivot = x if self.dx > 0 else x + w
            c.save()
            c.translate(pivot, ey)
            c.scale(lerp(0.7, 1, lk), lerp(0.7, 1, lk))
            c.translate(-pivot, -ey)
            pill(c, ctx, x, ey - h / 2, w, h, a=clamp(lk * 1.4))
            c.drawCircle(x + 26 * u, ey, 6 * u, skia.Paint(AntiAlias=True, Color=ctx.c("accent", clamp(lk * 1.4))))
            text(c, self.label, x + 44 * u, ey + 11 * u, fo, ctx.colors["paper"], clamp(lk * 1.4))
            c.restore()


class Tracker(Piece):
    """Corner brackets locking onto a box (x, y, w, h fractions) with a small label above."""

    def __init__(self, box, label=None, out=None):
        self.box, self.label, self.out = box, label, out

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        k = e_expo(prog(lt, 0, 0.4)) * (1 - e_expo(prog(lt, self.out if self.out is not None else dur - 0.2, 0.3)))
        if k <= 0:
            return
        x, y, w, h = self.box
        bw, bh = w * ctx.W * lerp(1.3, 1, k), h * ctx.H * lerp(1.3, 1, k)
        bx, by = (x + w / 2) * ctx.W - bw / 2, (y + h / 2) * ctx.H - bh / 2
        brackets(c, ctx, bx, by, bw, bh, a=k)
        if self.label:
            text(c, self.label.upper(), bx, by - 16 * u, ctx.font(20 * u, 700), ctx.colors["accent"], k, track=2 * u)


class Letterbox(Piece):
    def __init__(self, amount=0.125):
        self.amount = amount

    def draw(self, c, ctx, lt, dur, t):
        k = e_expo(prog(lt, 0, 0.3)) * (1 - e_in(prog(lt, dur - 0.12, 0.12)))
        if k <= 0:
            return
        bh = min(ctx.W, ctx.H) * self.amount * k
        p = skia.Paint(Color=col((0, 0, 0)))
        c.drawRect(skia.Rect.MakeXYWH(0, 0, ctx.W, bh), p)
        c.drawRect(skia.Rect.MakeXYWH(0, ctx.H - bh, ctx.W, bh), p)


class Label(Piece):
    """A small line of text that slides in: a section label, a speed tag (0.5x), a caption."""

    def __init__(self, text_, pos=None, size=22, color="paper", align="l", caps=True, weight=700):
        self.text, self.pos, self.size, self.color, self.align = text_, pos, size, color, align
        self.caps, self.weight = caps, weight

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        k = e_out(prog(lt, 0, 0.3)) * (1 - e_in(prog(lt, dur - 0.15, 0.15)))
        if k <= 0:
            return
        s = self.text.upper() if self.caps else self.text
        pos = _pos(ctx, self.pos, (0.05, 0.95) if self.align == "l" else (0.95, 0.085),
                   (0.07, 0.76) if self.align == "l" else (0.93, 0.14))
        text(c, s, ctx.W * pos[0], ctx.H * pos[1], ctx.font(self.size * u, self.weight),
             ctx.colors[self.color], k * 0.95, align=self.align[0], track=(3 if self.caps else 0) * u)


class Bug(Piece):
    """The brand's corner bug: small mark + name, top left."""

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        k = e_expo(prog(lt, 0.12, 0.5)) * (1 - e_in(prog(lt, dur - 0.15, 0.15)))
        if k <= 0:
            return
        x, y = (96 if not ctx.portrait else 60) * u, (96 if not ctx.portrait else 200) * u
        mark(c, ctx, x + 36 * u, y + 36 * u, 72 * u * k, alpha=k)
        text(c, ctx.name.upper(), x + 92 * u, y + 46 * u, ctx.font(26 * u, 800), ctx.colors["paper"], k, track=4 * u)


class UrlBar(Piece):
    """A browser address bar that types the url between typing=(t0, t1) (output times, e.g. while the
    link is spoken), then pops a button."""

    def __init__(self, url, typing, button_label="Sign up", out=None):
        self.url, self.typing, self.button_label, self.out = url, typing, button_label, out

    def draw(self, c, ctx, lt, dur, t):
        W, H, u = ctx.W, ctx.H, ctx.u
        u0, u1 = self.typing
        k = spring(prog(t, u0 - 0.28, 0.6)) * (1 - e_in(prog(lt, (self.out or dur) - 0.05, 0.05)))
        if k <= 0:
            return
        a = clamp(k * 1.5)
        bw, bh = min(820 * u, W - 80 * u), 104 * u
        bx, by = W / 2 - bw / 2, H - (170 if not ctx.portrait else 520) * u + (1 - k) * 80 * u
        pill(c, ctx, bx, by, bw, bh, a=a, border=0.18)
        lx, ly = bx + 52 * u, by + bh / 2
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(lx - 12 * u, ly - 2 * u, 24 * u, 20 * u), 4 * u, 4 * u),
                    skia.Paint(AntiAlias=True, Color=ctx.c("muted", a)))
        c.drawArc(skia.Rect.MakeXYWH(lx - 8 * u, ly - 16 * u, 16 * u, 24 * u), 180, 180, False,
                  skia.Paint(AntiAlias=True, Color=ctx.c("muted", a), Style=skia.Paint.kStroke_Style, StrokeWidth=3 * u))
        n = int(len(self.url) * clamp((t - u0) / max(u1 - u0, 0.01)))
        fo = fit(self.url, ctx.font(50 * u, 600), bw - 330 * u)
        w = text(c, self.url[:n], bx + 92 * u, ly + 18 * u, fo, ctx.colors["paper"], a)
        if int(t * 3) % 2 == 0 or n < len(self.url):
            c.drawRect(skia.Rect.MakeXYWH(bx + 96 * u + w, ly - 22 * u, 3.5 * u, 46 * u), skia.Paint(Color=ctx.c("accent", a)))
        if self.button_label:
            sk = spring(prog(t, u1 + 0.02, 0.45))
            fb = ctx.font(32 * u, 700)
            gw = tw(self.button_label, fb) + 70 * u
            button(c, ctx, bx + bw - gw / 2 - 18 * u, ly, self.button_label, k=sk, size=32)
