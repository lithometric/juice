"""The clean look: restrained, editorial motion for launches and enterprise trailers.

Where scenes.py punches (slams, springs, 3D cards, sparks), these pieces hold still and let the cut
do the work: type rises into place through a mask, product footage sits in flat windows on a plain
background, everything lines up on one margin, and color is spent on one word at a time.

    from juice.motion import Video, clean as c
    v.scene(c.Lockup(line="Soft-body physics, in the cloud."), 1.6, 3.6)
    v.scene(c.Frame(v.source("app.mp4", 2, 4), kicker="OpenBeam Cloud", title=["A physics lab", "in a browser tab."]),
            3.6, 5.6)
    v.overlay(c.Title("Now in beta.", kicker="OpenBeam", highlight={"beta.": "accent"}, scrim=True), 0.3, 1.6)
    v.overlay(c.Terms("If your research involves", [("World models", 10.2), ("Robotics", 11.1)]), 8.6, 14.0)

Pair them with the soft transitions (cut, fade, dissolve, wipe), the quiet hits (sub, tick, air,
swell) and the "keynote" bed. Footage, Card and the overlays in scenes.py still mix in.
"""
from __future__ import annotations

import skia

from .core import Source, button, clamp, col, contain, cover, fit, lerp, mark, prog, still, text, tw
from .scenes import Card, Footage, Piece, _frame, _media, _pos  # noqa: F401  (Footage, Card: re-exported)


def e_quint(x):
    """Fast start, long soft landing: the clean look's only ease."""
    return 1 - (1 - clamp(x)) ** 5


def margin(ctx) -> float:
    """The one left margin everything aligns to (px)."""
    return (80 if ctx.portrait else 120) * ctx.u


def plain(c, ctx, glow=1.0):
    """The clean background: the brand's ink, lifted a touch at the top, a faint warmth at the foot."""
    W, H = ctx.W, ctx.H
    ink = ctx.colors["ink"]
    c.clear(col(ink))
    lift = tuple(min(255, v + 18) for v in ink)
    g = skia.Paint(AntiAlias=True)
    g.setShader(skia.GradientShader.MakeRadial(skia.Point(W / 2, -H * 0.15), max(W, H) * 0.95,
                                               [col(lift), col(ink)], [0.0, 1.0]))
    c.drawRect(skia.Rect.MakeWH(W, H), g)
    if glow > 0:
        a = skia.Paint(AntiAlias=True)
        a.setShader(skia.GradientShader.MakeRadial(skia.Point(W / 2, H * 1.15), max(W, H) * 0.6,
                                                   [ctx.c("accent", 0.06 * glow), ctx.c("accent", 0)], [0.0, 1.0]))
        c.drawRect(skia.Rect.MakeWH(W, H), a)


def scrim(c, ctx, a=1.0, side="bottom"):
    """A soft dark gradient behind type over footage (bottom, or left in landscape)."""
    if a <= 0:
        return
    W, H = ctx.W, ctx.H
    p = skia.Paint(AntiAlias=True)
    if side == "left" and not ctx.portrait:
        pts = [skia.Point(0, 0), skia.Point(W * 0.62, 0)]
    else:
        pts = [skia.Point(0, H), skia.Point(0, H * (0.35 if ctx.portrait else 0.4))]
    p.setShader(skia.GradientShader.MakeLinear(pts, [ctx.c("ink", 0.78 * a), ctx.c("ink", 0)]))
    c.drawRect(skia.Rect.MakeWH(W, H), p)


def rise(c, segs, x, y, f, k, a=1.0, align="l", up=0.0):
    """One line of type rising into place through a mask (k 0 -> 1); up 0 -> 1 carries it out the top.
    segs = [(text, rgb)] drawn left to right; returns the line's width."""
    w = sum(f.measureText(s) for s, _ in segs)
    if k <= 0 or up >= 1 or a <= 0:
        return w
    size = f.getSize()
    x0 = x - w if align == "r" else x - w / 2 if align == "c" else x
    c.save()
    c.clipRect(skia.Rect.MakeLTRB(x0 - size * 0.2, y - size * 1.02, x0 + w + size * 0.2, y + size * 0.3))
    dy = (1 - k) * size * 1.25 - up * size * 1.35
    for s, rgb in segs:
        text(c, s, x0, y + dy, f, rgb, a)
        x0 += f.measureText(s)
    c.restore()
    return w


def kicker(c, ctx, label, x, y, a=1.0, align="l", color="muted"):
    """A small uppercase label with an accent square: the section name above a title."""
    if a <= 0 or not label:
        return
    u = ctx.u
    f = ctx.font(22 * u, 600)
    s = label.upper()
    w = tw(s, f, 3 * u) + 26 * u
    x0 = x - w if align == "r" else x - w / 2 if align == "c" else x
    c.drawRect(skia.Rect.MakeXYWH(x0, y - 15 * u, 12 * u, 12 * u), skia.Paint(AntiAlias=True, Color=ctx.c("accent", a)))
    text(c, s, x0 + 26 * u, y, f, ctx.colors[color], a, track=3 * u)


def _segs(ctx, line, highlight, color):
    words = line.split(" ")
    return [(wd + (" " if i < len(words) - 1 else ""), ctx.colors[highlight.get(wd, color)]) for i, wd in enumerate(words)]


# --- overlays -------------------------------------------------------------------------------------

class Title(Piece):
    """A title that rises in line by line through a mask, under an optional kicker, with an optional
    body line below. pos = (x, y) fractions: the left edge (or center / right, by align) and the
    baseline of the last title line. highlight = {word: color role}, used sparingly."""

    def __init__(self, lines, kicker=None, body=None, pos=None, size=84, weight=600, color="paper", highlight=None,
                 align="l", scrim=False, stagger=0.08, out=True, max_w=None):
        self.lines = [lines] if isinstance(lines, str) else list(lines)
        self.body = [body] if isinstance(body, str) else list(body or [])
        self.kicker, self.pos, self.size, self.weight, self.max_w = kicker, pos, size, weight, max_w
        self.color, self.highlight, self.align, self.scrim = color, highlight or {}, align, scrim
        self.stagger, self.out = stagger, out

    def draw(self, c, ctx, lt, dur, t):
        if lt < 0:
            return
        u = ctx.u
        m = margin(ctx)
        f = ctx.font(self.size * u, self.weight, "display")
        mw = self.max_w or ctx.W - 2 * m
        f = min((fit(ln, f, mw) for ln in self.lines), key=lambda g: g.getSize())
        size = f.getSize()
        px, py = _pos(ctx, self.pos, (m / ctx.W, 0.86), (m / ctx.W, 0.70))
        x, y_last = ctx.W * px, ctx.H * py
        lh = size * 1.08
        y0 = y_last - (len(self.lines) - 1) * lh
        a = 1 - e_quint(prog(lt, dur - 0.25, 0.25)) if self.out else 1.0
        if self.scrim:
            scrim(c, ctx, e_quint(prog(lt, 0, 0.4)) * a)
        kicker(c, ctx, self.kicker, x, y0 - size * 1.0, e_quint(prog(lt, 0.0, 0.5)) * a, self.align)
        for i, ln in enumerate(self.lines):
            k = e_quint(prog(lt, 0.05 + i * self.stagger, 0.7))
            rise(c, _segs(ctx, ln, self.highlight, self.color), x, y0 + i * lh, f, k, a, self.align)
        if self.body:
            fb = min((fit(ln, ctx.font(32 * u, 400), mw) for ln in self.body), key=lambda g: g.getSize())
            for i, ln in enumerate(self.body):
                k = e_quint(prog(lt, 0.2 + (len(self.lines) + i) * self.stagger, 0.7))
                rise(c, [(ln, ctx.colors["muted"])], x, y_last + size * 0.7 + 22 * u + i * fb.getSize() * 1.35, fb, k, a,
                     self.align)


class Caption(Piece):
    """A quiet line in the corner of full-bleed footage: an index, a hairline, a label."""

    def __init__(self, label, index=None, pos=None):
        self.label, self.index, self.pos = label, index, pos

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        k = e_quint(prog(lt, 0, 0.6))
        a = k * (1 - e_quint(prog(lt, dur - 0.2, 0.2)))
        if a <= 0:
            return
        m = margin(ctx)
        px, py = _pos(ctx, self.pos, (m / ctx.W, 0.925), (m / ctx.W, 0.74))
        x, y = ctx.W * px, ctx.H * py
        scrim(c, ctx, 0.6 * a)
        f = ctx.font(30 * u, 500)
        if self.index:
            fi = ctx.font(30 * u, 600)
            text(c, self.index, x, y, fi, ctx.colors["accent"], a)
            x += fi.measureText(self.index) + 18 * u
            c.drawRect(skia.Rect.MakeXYWH(x, y - 9 * u, 40 * u * k, 1.5 * u), skia.Paint(Color=ctx.c("paper", 0.6 * a)))
            x += 58 * u
        rise(c, [(self.label, ctx.colors["paper"])], x, y, f, k, a)


class Terms(Piece):
    """A lead line, then terms that replace one another as they're spoken: each rises in through a
    mask and lifts the last one out, with progress ticks under them. lead = a string, or
    [(word, output_time)] to appear with the voice; terms = [(text, output_time)]."""

    def __init__(self, lead, terms, pos=None, size=132, ticks=True, scrim=True):
        self.lead, self.terms, self.pos, self.size, self.ticks, self.scrim = lead, list(terms), pos, size, ticks, scrim

    def draw(self, c, ctx, lt, dur, t):
        u = ctx.u
        m = margin(ctx)
        a = 1 - e_quint(prog(lt, dur - 0.22, 0.22))
        if self.scrim:
            scrim(c, ctx, e_quint(prog(lt, 0, 0.4)) * a)
        px, py = _pos(ctx, self.pos, (m / ctx.W, 0.84), (m / ctx.W, 0.72))
        x, y = ctx.W * px, ctx.H * py
        f = ctx.font(self.size * u, 600, "display")
        f = min((fit(s, f, ctx.W - 2 * m) for s, _ in self.terms), key=lambda g: g.getSize())
        size = f.getSize()
        fl = ctx.font((40 if not ctx.portrait else 38) * u, 500)
        yl = y - size * 1.08
        if isinstance(self.lead, str):
            rise(c, [(self.lead, ctx.colors["muted"])], x, yl, fl, e_quint(prog(lt, 0, 0.6)), a)
        else:
            xx = x
            for wd, st in self.lead:
                k = e_quint(prog(t, st - 0.05, 0.45))
                rise(c, [(wd, ctx.colors["muted"])], xx, yl, fl, k, a)
                xx += fl.measureText(wd + " ")
        for i, (s, ti) in enumerate(self.terms):
            k = e_quint(prog(t, ti - 0.04, 0.55))
            up = e_quint(prog(t, self.terms[i + 1][1] - 0.04, 0.45)) if i + 1 < len(self.terms) else 0.0
            rise(c, [(s, ctx.colors["paper"])], x, y, f, k, a, up=up)
        if self.ticks and self.terms:
            n = len(self.terms)
            first = e_quint(prog(t, self.terms[0][1], 0.5)) * a
            cur = max([i for i, (_, ti) in enumerate(self.terms) if t >= ti - 0.04] or [-1])
            for i in range(n):
                on = i == cur
                c.drawRect(skia.Rect.MakeXYWH(x + i * 58 * u, y + 44 * u, 44 * u, 4 * u),
                           skia.Paint(Color=ctx.c("accent" if on else "paper", first * (1 if on else 0.25))))


class Notice(Piece):
    """A system notification that slides in (top right; lower third in portrait, clear of the
    speaker's face): the brand mark, a sender, a title and a body line."""

    def __init__(self, title, body=None, sender=None, when="now"):
        self.title, self.body, self.sender, self.when = title, body, sender, when

    def draw(self, c, ctx, lt, dur, t):
        if lt < 0:
            return
        u = ctx.u
        k = e_quint(prog(lt, 0, 0.6))
        a = k * (1 - e_quint(prog(lt, dur - 0.2, 0.2)))
        if a <= 0:
            return
        w, h = (min(960 * u, ctx.W - 120 * u), 150 * u) if ctx.portrait else (600 * u, 150 * u)
        x = ctx.W / 2 - w / 2 if ctx.portrait else ctx.W - w - 56 * u
        y = ctx.H * 0.62 if ctx.portrait else 56 * u
        x += (1 - k) * 40 * u * (0 if ctx.portrait else 1)
        y -= (1 - k) * 30 * u * (1 if ctx.portrait else 0)
        rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), 26 * u, 26 * u)
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y + 14 * u, w, h), 26 * u, 26 * u),
                    skia.Paint(AntiAlias=True, Color=col((0, 0, 0), 0.4 * a),
                               MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 28 * u)))
        c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c("surface", 0.94 * a)))
        c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c("paper", 0.12 * a), Style=skia.Paint.kStroke_Style,
                                   StrokeWidth=1.5 * u))
        mark(c, ctx, x + 64 * u, y + h / 2, 68 * u, alpha=a)
        tx = x + 122 * u
        fs = ctx.font(22 * u, 600)
        text(c, (self.sender or ctx.name).upper(), tx, y + 46 * u, fs, ctx.colors["muted"], a, track=2 * u)
        text(c, self.when, x + w - 30 * u, y + 46 * u, ctx.font(22 * u, 500), ctx.colors["muted"], a, align="r")
        text(c, self.title, tx, y + 88 * u, fit(self.title, ctx.font(30 * u, 600), w - 160 * u), ctx.colors["paper"], a)
        if self.body:
            text(c, self.body, tx, y + 124 * u, fit(self.body, ctx.font(24 * u, 400), w - 160 * u),
                 ctx.colors["muted"], a)


# --- scenes ---------------------------------------------------------------------------------------

def window(c, ctx, x, y, w, h, media=None, url=None, a=1.0, zoom=1.0, chrome=True, label=None):
    """A flat app window: soft shadow, hairline border, an optional browser bar with the url, the
    media filling the rest. Returns the content rect."""
    u = ctx.u
    r = 18 * u
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y + 30 * u, w, h), r, r),
                skia.Paint(AntiAlias=True, Color=col((0, 0, 0), 0.55 * a),
                           MaskFilter=skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 50 * u)))
    c.save()
    c.clipRRect(rr, True)
    c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), skia.Paint(Color=ctx.c("surface", a)))
    top = y
    if chrome:
        bh = 46 * u
        for i in range(3):
            c.drawCircle(x + (26 + i * 22) * u, y + bh / 2, 6 * u, skia.Paint(AntiAlias=True, Color=ctx.c("paper", 0.18 * a)))
        if url:
            f = ctx.font(19 * u, 500)
            uw = min(tw(url, f) + 60 * u, w * 0.5)
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x + w / 2 - uw / 2, y + 9 * u, uw, bh - 18 * u), 8 * u, 8 * u),
                        skia.Paint(AntiAlias=True, Color=ctx.c("ink", 0.6 * a)))
            text(c, url, x + w / 2, y + bh / 2 + 7 * u, f, ctx.colors["muted"], a, align="c")
        top = y + bh
    if media is not None:
        cover(c, media, x, top, w, y + h - top, zoom=zoom, alpha=a)
    c.restore()
    c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c("paper", 0.13 * a), Style=skia.Paint.kStroke_Style,
                               StrokeWidth=1.5 * u))
    if label:
        kicker(c, ctx, label, x, y - 22 * u, a)
    return x, top, w, y + h - top


def _enter(lt, delay=0.0, d=0.8):
    return e_quint(prog(lt, delay, d))


class Frame(Piece):
    """Product footage in a flat window on the plain background, beside its copy (kicker, title,
    body; above it in portrait). enter=False holds the layout still, for cutting between clips
    inside the same window. url puts a browser bar on the window."""

    def __init__(self, media, kicker=None, title=None, body=None, url=None, enter=True, zoom=(1.0, 1.05)):
        self.media, self.kicker, self.body, self.url, self.enter, self.zoom = _media(media), kicker, body, url, enter, zoom
        self.title = [title] if isinstance(title, str) else list(title or [])

    def _layout(self, ctx):
        u, W, H, m = ctx.u, ctx.W, ctx.H, margin(ctx)
        copy = bool(self.title or self.kicker)
        if ctx.portrait:
            w = W - 2 * m
            h = w / 1.6 + (46 * u if self.url else 0)
            return (m, 760 * u if copy else H / 2 - h / 2, w, h), (m / W, 560 / 1920)
        if not copy:
            w = 1500 * u
            h = w / 1.6 + (46 * u if self.url else 0)
            return (W / 2 - w / 2, H / 2 - h / 2, w, h), None
        w = 1040 * u
        h = w / 1.6 + (46 * u if self.url else 0)
        return (W - m - w, H / 2 - h / 2, w, h), (m / W, 0.5 if self.body else 0.54)

    def draw(self, c, ctx, lt, dur, t):
        plain(c, ctx)
        (x, y, w, h), tpos = self._layout(ctx)
        k = _enter(lt, 0.0, 0.9) if self.enter else 1.0
        z = lerp(*self.zoom, clamp(lt / max(dur, 0.01)))
        c.save()
        s = lerp(0.965, 1.0, k)
        c.translate(x + w / 2, y + h / 2 + (1 - k) * 60 * ctx.u)
        c.scale(s, s)
        c.translate(-(x + w / 2), -(y + h / 2))
        window(c, ctx, x, y, w, h, _frame(self.media, lt), self.url, a=k, zoom=z, chrome=bool(self.url))
        c.restore()
        if tpos and self.title:
            size = 68 if not ctx.portrait else 72
            col_w = ctx.W - 2 * margin(ctx) if ctx.portrait else x - margin(ctx) - 80 * ctx.u
            Title(self.title, kicker=self.kicker, body=self.body, pos=tpos, size=size, out=False, stagger=0.08,
                  max_w=col_w).draw(c, ctx, lt + (0 if self.enter else 10), dur, t)


class Split(Piece):
    """Two windows joined by a labeled link (side by side; stacked in portrait), e.g. an AI agent
    driving the product: data dots run along the link."""

    def __init__(self, left, right, link="MCP", left_label=None, right_label=None, left_url=None, right_url=None):
        self.left, self.right, self.link = _media(left), _media(right), link
        self.left_label, self.right_label, self.left_url, self.right_url = left_label, right_label, left_url, right_url

    def draw(self, c, ctx, lt, dur, t):
        plain(c, ctx)
        u, W, m = ctx.u, ctx.W, margin(ctx)
        if ctx.portrait:
            A = (m, 200 * u, W - 2 * m, 320 * u)
            B = (m, 600 * u, W - 2 * m, (W - 2 * m) * 9 / 16 + 46 * u)
            p0, p1 = (W / 2, A[1] + A[3]), (W / 2, B[1])
        else:
            A = (m, 150 * u, 560 * u, 560 * u)
            B = (m + 680 * u, 150 * u, W - 2 * m - 680 * u, 560 * u)
            p0, p1 = (A[0] + A[2], A[1] + A[3] / 2), (B[0], B[1] + B[3] / 2)
        z = lerp(1.0, 1.04, clamp(lt / max(dur, 0.01)))
        for i, (box, med, lab, url) in enumerate(((A, self.left, self.left_label, self.left_url),
                                                  (B, self.right, self.right_label, self.right_url))):
            k = _enter(lt, 0.08 * i, 0.8)
            x, y, w, h = box
            c.save()
            c.translate(0, (1 - k) * 50 * u)
            window(c, ctx, x, y, w, h, _frame(med, lt), url, a=k, zoom=z, chrome=bool(url), label=lab)
            c.restore()
        lk = e_quint(prog(lt, 0.3, 0.6))
        if lk <= 0 or not self.link:
            return
        line = skia.Paint(AntiAlias=True, Color=ctx.c("accent", 0.9), StrokeWidth=2 * u)
        c.drawLine(p0[0], p0[1], lerp(p0[0], p1[0], lk), lerp(p0[1], p1[1], lk), line)
        for j in range(3):
            q = (lt * 0.8 + j / 3) % 1.0
            c.drawCircle(lerp(p0[0], p1[0], q * lk), lerp(p0[1], p1[1], q * lk), 4.5 * u,
                         skia.Paint(AntiAlias=True, Color=ctx.c("accent", lk)))
        f = ctx.font(20 * u, 700)
        lw, lh = tw(self.link, f, 2 * u) + 34 * u, 40 * u
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        pk = e_quint(prog(lt, 0.55, 0.5))
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(mx - lw / 2, my - lh / 2, lw, lh), lh / 2, lh / 2),
                    skia.Paint(AntiAlias=True, Color=ctx.c("accent", pk)))
        text(c, self.link, mx, my + 7 * u, f, ctx.colors["ink"], pk, align="c", track=2 * u)


class Bento(Piece):
    """A kicker and title over a bento grid of feature cells (Card: a title over footage, or a drawn
    card with body, url field and button). Cells appear one per `every` seconds (put it on the beat)."""

    def __init__(self, title, cards: list[Card], kicker=None, every=0.5):
        self.title, self.cards, self.kicker, self.every = title, cards[:4], kicker, every

    def sources(self):
        return [cd.media for cd in self.cards if isinstance(cd.media, Source)]

    def _cells(self, ctx):
        u, W, H, m = ctx.u, ctx.W, ctx.H, margin(ctx)
        g = 24 * u
        if ctx.portrait:
            top, bot = 480 * u, 1460 * u
            h = (bot - top - 3 * g) / 4
            return [(m, top + i * (h + g), W - 2 * m, h) for i in range(len(self.cards))]
        top, bot = 290 * u, H - 90 * u
        h = (bot - top - g) / 2
        cw = (W - 2 * m - 2 * g) / 3
        spans = [(0, 0, 2), (2, 0, 1), (0, 1, 1), (1, 1, 2)]
        return [(m + col_ * (cw + g), top + row * (h + g), cw * span + g * (span - 1), h)
                for col_, row, span in spans[:len(self.cards)]]

    def _drawn(self, c, ctx, cd, x, y, w, h, a):
        u = ctx.u
        pad = 40 * u
        text(c, cd.title, x + pad, y + 76 * u, fit(cd.title, ctx.font(36 * u, 600, "display"), w - 2 * pad),
             ctx.colors["paper"], a)
        yy = y + 120 * u
        if cd.body and h > 300 * u:
            text(c, cd.body, x + pad, yy, fit(cd.body, ctx.font(24 * u, 400), w - 2 * pad), ctx.colors["muted"], a)
            yy += 40 * u
        if cd.url:
            fh, gap = 64 * u, 20 * u
            fb = ctx.font(24 * u, 600)
            bw = tw(cd.button, fb) + 60 * u if cd.button else 0
            beside = w - 2 * pad - bw - gap >= 400 * u   # the button beside the field, else under it
            fw = min(w - 2 * pad - (bw + gap if beside else 0), 760 * u)
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x + pad, yy, fw, fh), 12 * u, 12 * u),
                        skia.Paint(AntiAlias=True, Color=ctx.c("ink", 0.8 * a)))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x + pad, yy, fw, fh), 12 * u, 12 * u),
                        skia.Paint(AntiAlias=True, Color=ctx.c("paper", 0.12 * a), Style=skia.Paint.kStroke_Style,
                                   StrokeWidth=1.5 * u))
            text(c, cd.url, x + pad + 24 * u, yy + fh / 2 + 9 * u, fit(cd.url, ctx.font(26 * u, 500), fw - 48 * u),
                 ctx.colors["paper"], a)
            if cd.button:
                bx, by = (x + pad + fw + gap, yy) if beside else (x + pad, yy + fh + gap)
                button(c, ctx, bx + bw / 2, by + fh / 2, cd.button, fill="accent", size=24, a=a)

    def draw(self, c, ctx, lt, dur, t):
        plain(c, ctx)
        u = ctx.u
        m = margin(ctx)
        ty = (200 if not ctx.portrait else 380) * u
        Title(self.title, kicker=self.kicker, pos=(m / ctx.W, ty / ctx.H), size=64 if not ctx.portrait else 66,
              out=False).draw(c, ctx, lt, dur, t)
        push = lerp(1.0, 1.02, clamp(lt / max(dur, 0.01)))
        cells = self._cells(ctx)
        cx = ctx.W / 2
        cy = (cells[0][1] + cells[-1][1] + cells[-1][3]) / 2 if cells else ctx.H / 2
        c.save()
        c.translate(cx, cy)
        c.scale(push, push)
        c.translate(-cx, -cy)
        r = 20 * u
        for k, (cd, (x, y, w, h)) in enumerate(zip(self.cards, cells)):
            p = _enter(lt, 0.15 + k * self.every, 0.7)
            if p <= 0:
                continue
            y += (1 - p) * 36 * u
            rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)
            c.save()
            c.clipRRect(rr, True)
            c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), skia.Paint(Color=ctx.c("surface", p)))
            if cd.media is None:
                g = skia.Paint(AntiAlias=True)
                g.setShader(skia.GradientShader.MakeRadial(skia.Point(x + w * 0.85, y), w * 0.6,
                                                           [ctx.c("accent", 0.08 * p), ctx.c("accent", 0)], [0, 1]))
                c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), g)
                self._drawn(c, ctx, cd, x, y, w, h, p)
            else:
                cover(c, _frame(cd.media, lt), x, y, w, h, zoom=1.0 + 0.04 * clamp(lt / max(dur, 0.01)), alpha=p)
                gr = skia.Paint(AntiAlias=True)
                gr.setShader(skia.GradientShader.MakeLinear([skia.Point(0, y + h * 0.45), skia.Point(0, y + h)],
                                                            [ctx.c("ink", 0), ctx.c("ink", 0.85 * p)]))
                c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), gr)
                text(c, cd.title, x + 36 * u, y + h - 36 * u, fit(cd.title, ctx.font(32 * u, 600), w - 72 * u),
                     ctx.colors["paper"], p)
                if cd.rec:
                    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x + 24 * u, y + 22 * u, 104 * u, 38 * u), 19 * u, 19 * u),
                                skia.Paint(AntiAlias=True, Color=ctx.c("ink", 0.7 * p)))
                    c.drawCircle(x + 46 * u, y + 41 * u, 7 * u, skia.Paint(AntiAlias=True, Color=col((235, 72, 60), p)))
                    text(c, "REC", x + 62 * u, y + 49 * u, ctx.font(20 * u, 600), ctx.colors["paper"], p, track=2 * u)
            c.restore()
            c.drawRRect(rr, skia.Paint(AntiAlias=True, Color=ctx.c("paper", 0.12 * p), Style=skia.Paint.kStroke_Style,
                                       StrokeWidth=1.5 * u))
        c.restore()


class Lockup(Piece):
    """The brand lockup on the plain background: the wordmark (or mark + name) settling in, a line
    under it with the link in the brand color, and a small sub line at the foot. url=True uses the
    kit's url."""

    def __init__(self, line=None, url=None, sub=None):
        self.line, self.url, self.sub = line, url, sub

    def draw(self, c, ctx, lt, dur, t):
        plain(c, ctx, glow=1.5)
        u, W, H = ctx.u, ctx.W, ctx.H
        k = e_quint(prog(lt, 0.0, 1.0))
        drift = lerp(1.0, 1.025, clamp(lt / max(dur, 0.01)))
        s = lerp(0.95, 1.0, k) * drift
        cy = H * (0.44 if not ctx.portrait else 0.42)
        wm = ctx.kit.get("logo", {}).get("wordmark")
        img = still(str(wm)) if wm and not str(wm).startswith("http") else None
        c.save()
        c.translate(W / 2, cy)
        c.scale(s, s)
        if img is not None:
            ww = (880 if not ctx.portrait else 860) * u
            wh = ww * img.height() / img.width()
            contain(c, img, -ww / 2, -wh / 2, ww, wh, alpha=k)
            bottom = wh / 2
        else:
            ms = 150 * u
            f = ctx.font(132 * u, 700, "display")
            nw = f.measureText(ctx.name)
            gw = ms + 40 * u + nw
            mark(c, ctx, -gw / 2 + ms / 2, 0, ms, alpha=k)
            text(c, ctx.name, -gw / 2 + ms + 40 * u, 46 * u, f, ctx.colors["paper"], k)
            bottom = ms / 2
        c.restore()
        url = self.url if isinstance(self.url, str) else \
            (ctx.kit.get("url") or "").replace("https://", "").replace("http://", "").rstrip("/") if self.url else ""
        y = cy + bottom * s + (110 if not ctx.portrait else 130) * u
        segs = ([(self.line + (" " if url else ""), ctx.colors["muted"])] if self.line else []) + \
               ([(url, ctx.colors["accent"])] if url else [])
        if segs:
            f = ctx.font(40 * u, 500)
            whole = "".join(s_ for s_, _ in segs)
            f = fit(whole, f, W - 2 * margin(ctx))
            rise(c, segs, W / 2, y, f, e_quint(prog(lt, 0.35, 0.8)), align="c")
        if self.sub:
            fs = fit(self.sub.upper(), ctx.font(20 * u, 600), W - 2 * margin(ctx))
            a = e_quint(prog(lt, 0.7, 0.8))
            text(c, self.sub.upper(), W / 2, H * (0.88 if not ctx.portrait else 0.78), fs, ctx.colors["muted"], a * 0.8,
                 align="c", track=4 * u)
