"""Components: themed, animated overlay building blocks.

Every component is `draw(canvas, t, dur, p, ctx)`:
  canvas  RGBA frame (output size) the component draws onto
  t       seconds since this layer started; dur = how long the layer lives
  p       the component's params from the spec / event
  ctx     render context: ctx.theme, ctx.W, ctx.H, ctx.u (1.0 at 1080px short side), ctx.t_out
          (absolute output time), ctx.meter(id) (current meter value 0-100), ctx.target(name)
          (tracked box in output pixels, or None when hidden/out of frame), ctx.words (captions)

Components are genre-neutral. A fight uses versus + meters + slam + callout; a story uses
lower_third + captions + title; rage bait uses captions(loud) + slam + chip. Colors are theme roles
("accent", "ink", "paper", "danger", "accent2") or [r, g, b].

  flash        {dur=0.25, color="paper", strength=0.9}
  slam         {text, color="paper", sub?, sub_fill="danger", y=0.42, size=1.0}
               giant word that fills the screen faintly, then snaps to size (FIGHT!, K.O., WAIT...)
  versus       {left, right, left_color="accent", right_color="paper", premise?}
  title        {text, kicker?, sub?, color="accent", y=0.62, size=1.0}       (WINNER / CLAUDE / STILL STANDING)
  lower_third  {name, role?, side="left", y=0.78}
  chip         {text, x=0.5, y=0.5, fill="ink", ink="paper", size=1.0, anchor="center", slide=true}
  caption      {text, y=0.75, size=1.0, color="paper"}                      static subtitle line
  captions     {y=0.72, size=1.0, max_words=3, active=theme.caption_active, color="paper", uppercase=true}
  meters       {left: {name, meter, color, label}, right: {...}, counter?: {from=99, rate=1}}
  progress     {y=0.0 | 1.0, color="accent", height=8}                      story progress bar
  callout      {target, text, label?, dx=0.12, dy=-0.18, color="accent"}  circle + leader + label
  tags         {targets: {NAME: {sub?, color?, until?}}}                   name tags pinned over tracked heads
  ticker       {text, dot="danger", y=0.93, x=0.03}
  watermark    {text, corner="bottom-right"}
  letterbox    {amount=0.09}                                               bars slide in
  endcard      {title, tagline?, link?, handle?, dim=0.72}
"""
from __future__ import annotations

import math

from PIL import Image, ImageDraw

from . import theme as th

_cache: dict = {}


def cached(key, fn):
    if key not in _cache:
        if len(_cache) > 600:
            _cache.clear()
        _cache[key] = fn()
    return _cache[key]


def ease_out(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_back(t, s=1.6):
    t = min(max(t, 0.0), 1.0) - 1
    return 1 + (s + 1) * t ** 3 + s * t ** 2


def _key(ctx, *a):
    return (id(ctx.theme), ctx.W, ctx.H) + a


def _headline(ctx, text, size, fill, role="display", shadow=None, outline=None):
    return cached(_key(ctx, "h", text, int(size), str(fill), role, shadow, outline),
                  lambda: th.headline(ctx.theme, text, size, fill=fill, role=role, shadow=shadow, outline=outline))


def _chip(ctx, text, size, fill="ink", ink="paper", role="label"):
    return cached(_key(ctx, "c", text, int(size), str(fill), str(ink), role),
                  lambda: th.chip(ctx.theme, text, size, fill=fill, ink=ink, role=role))


def _fit(img: Image.Image, max_w: float) -> float:
    return min(1.0, max_w / max(1, img.width))


# --- components -----------------------------------------------------------------------------------

def flash(c, t, dur, p, ctx):
    d = p.get("dur", 0.25)
    if t < d:
        a = (1 - ease_out(t / d)) * p.get("strength", 0.9)
        c.alpha_composite(Image.new("RGBA", c.size, th.color(ctx.theme, p.get("color", "paper"), int(255 * a))))


def slam(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    size = 300 * u * p.get("size", 1.0)
    img = _headline(ctx, p["text"], size, p.get("color", "paper"), outline=p.get("outline", max(3, int(7 * u))))
    final = _fit(img, W * 0.86)
    y = H * p.get("y", 0.42)
    # Phase 1: the word fills the frame, faint, then rushes down to size (0 - 0.22s).
    k = ease_out(t / 0.22)
    huge = (W * 1.15) / img.width
    scale = huge + (final - huge) * k
    th.paste(c, img, W / 2, y + (H / 2 - y) * (1 - k), scale=scale, alpha=0.25 + 0.75 * k)
    if p.get("sub") and t > 0.18:
        ks = ease_out((t - 0.18) / 0.2)
        ch = _chip(ctx, p["sub"], 40 * u, fill=p.get("sub_fill", "danger"), ink="paper")
        th.paste(c, ch, W / 2, y + img.height * final * 0.5 + 40 * u + (1 - ks) * 20 * u, alpha=ks)


def versus(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    vertical = H > W
    c.alpha_composite(Image.new("RGBA", c.size, (10, 10, 12, int(110 * ease_out(t / 0.2)))))
    size = (190 if vertical else 170) * u
    L = _headline(ctx, p["left"], size, p.get("left_color", "accent"))
    R = _headline(ctx, p["right"], size, p.get("right_color", "paper"))
    vs = _chip(ctx, "VS", 70 * u, fill="paper", ink="ink", role="display")
    kl, kr, kv = ease_out(t / 0.28), ease_out((t - 0.1) / 0.28), ease_back((t - 0.3) / 0.25)
    if vertical:
        s = min(_fit(L, W * 0.86), _fit(R, W * 0.86))
        th.paste(c, L, -W / 2 + (W) * kl, H * 0.37, scale=s, alpha=min(1, kl * 2))
        th.paste(c, R, W * 1.5 - W * kr, H * 0.57, scale=s, alpha=min(1, max(0, kr) * 2))
        vy, py = H * 0.47, H * 0.66
    else:
        s = min(_fit(L, W * 0.36), _fit(R, W * 0.36))
        th.paste(c, L, -W * 0.3 + W * 0.56 * kl, H * 0.44, scale=s, alpha=min(1, kl * 2))
        th.paste(c, R, W * 1.3 - W * 0.56 * kr, H * 0.44, scale=s, alpha=min(1, max(0, kr) * 2))
        vy, py = H * 0.44, H * 0.6
    if t > 0.3:
        th.paste(c, vs, W / 2, vy, scale=max(0.01, 2.2 - 1.2 * kv))
    if p.get("premise") and t > 0.55:
        # Typed on, monospace: the rule/stakes of the video.
        text = p["premise"]
        n = int(len(text) * min(1, (t - 0.55) / 0.6))
        cursor = "_" if n < len(text) and int(t * 6) % 2 == 0 else ""
        pc = _chip(ctx, text[:n] + cursor, 34 * u, fill="ink", ink="paper") if n else None
        if pc:
            th.paste(c, pc, W / 2, py)


def title(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    y = H * p.get("y", 0.62)
    big = _headline(ctx, p["text"], 230 * u * p.get("size", 1.0), p.get("color", "accent"),
                    outline=p.get("outline", max(3, int(6 * u))))
    s = _fit(big, W * 0.82)
    k = ease_back(t / 0.3)
    th.paste(c, big, W / 2, y, scale=s * (0.6 + 0.4 * k), alpha=min(1, t / 0.12))
    bh = big.height * s
    if p.get("kicker"):
        kk = ease_out((t - 0.12) / 0.25)
        kc = _chip(ctx, p["kicker"], 38 * u, fill=p.get("color", "accent"), ink="ink")
        th.paste(c, kc, W / 2, y - bh * 0.42 + (1 - kk) * 16 * u, alpha=kk)
    if p.get("sub"):
        ks = ease_out((t - 0.25) / 0.25)
        sc = _chip(ctx, p["sub"], 34 * u, fill="ink", ink="paper")
        th.paste(c, sc, W / 2, y + bh * 0.42 + (1 - ks) * 16 * u, alpha=ks)


def lower_third(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    k = ease_out(t / 0.35)
    out = ease_out((t - (dur - 0.3)) / 0.3) if dur > 1 else 0
    left = p.get("side", "left") == "left"
    name = _headline(ctx, p["name"], 84 * u, "paper", role="display")
    x0 = W * 0.06 if left else W * 0.94
    y = H * p.get("y", 0.78)
    shift = (1 - k + out) * 60 * u * (-1 if left else 1)
    th.paste(c, name, x0 + shift, y, alpha=k * (1 - out), anchor="left" if left else "right")
    if p.get("role"):
        rc = _chip(ctx, p["role"], 30 * u, fill="accent", ink="ink")
        th.paste(c, rc, x0 + shift, y + name.height * 0.42 + 18 * u, alpha=k * (1 - out),
                 anchor="left" if left else "right")


def chip(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    img = _chip(ctx, p["text"], 34 * u * p.get("size", 1.0), fill=p.get("fill", "ink"), ink=p.get("ink", "paper"))
    k = ease_out(t / 0.25) if p.get("slide", True) else 1
    th.paste(c, img, W * p.get("x", 0.5) + (1 - k) * 30 * u, H * p.get("y", 0.5), alpha=k,
             anchor=p.get("anchor", "center"))


def caption(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    img = _headline(ctx, p["text"], 64 * u * p.get("size", 1.0), p.get("color", "paper"), role="body", shadow="soft")
    th.paste(c, img, W / 2, H * p.get("y", 0.75), scale=_fit(img, W * 0.9))


def captions(c, t, dur, p, ctx):
    """Word-by-word captions from the transcript: the spoken chunk, active word in the accent color."""
    W, H, u = ctx.W, ctx.H, ctx.u
    words = ctx.words or []
    now = ctx.t_out
    i = next((k for k, w in enumerate(words) if w["start"] <= now < w["end"] + 0.25), None)
    if i is None:
        return
    m = p.get("max_words", 3)
    a = (i // m) * m
    chunk = words[a:a + m]
    size = 86 * u * p.get("size", 1.0)
    up = p.get("uppercase", True)
    imgs = []
    for k, w in enumerate(chunk):
        txt = w["text"].upper() if up else w["text"]
        active = (a + k) == i
        hi = p.get("active") or ctx.theme.get("caption_active", "accent")
        imgs.append(_headline(ctx, txt, size, hi if active else p.get("color", "paper"),
                              role=p.get("role", "display")))
    gap = 18 * u
    total = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    s = _fit(Image.new("RGBA", (int(total), 1)), W * 0.9)
    x = W / 2 - total * s / 2
    pop = 1 + 0.12 * (1 - ease_out((now - words[i]["start"]) / 0.12))
    for k, im in enumerate(imgs):
        sc = s * (pop if (a + k) == i else 1)
        th.paste(c, im, x + im.width * s / 2, H * p.get("y", 0.72), scale=sc)
        x += im.width * s + gap * s


def meters(c, t, dur, p, ctx):
    """Two-sided stat bars (health/balance/hype/rage...) with an optional center counter."""
    W, H, u = ctx.W, ctx.H, ctx.u
    k = ease_out(t / 0.35)
    top = H * (0.06 if W > H else 0.12) - (1 - k) * 80 * u
    cw = 90 * u
    bar_w, bar_h = (W - cw) / 2 - W * 0.05, 22 * u
    d = ImageDraw.Draw(c)
    for side, sign in (("left", -1), ("right", 1)):
        s = p.get(side)
        if not s:
            continue
        v = max(0.0, min(100.0, ctx.meter(s["meter"]))) / 100 if s.get("meter") else 1.0
        col = th.color(ctx.theme, s.get("color", "accent" if side == "left" else "paper"))
        ink = th.color(ctx.theme, "ink")
        x_in = W / 2 + sign * cw / 2 + sign * 10 * u   # edge next to the counter
        x_out = x_in + sign * bar_w
        skew = 12 * u
        y0, y1 = top + 34 * u, top + 34 * u + bar_h
        # frame
        frame = [(x_out, y1), (x_in, y1), (x_in + sign * skew, y0), (x_out + sign * skew, y0)]
        d.polygon(frame, fill=ink)
        # fill drains toward the outer edge like a fighting game
        inner = 4 * u
        fx_out = x_out - sign * inner
        fx_in = fx_out + (x_in - sign * inner - fx_out) * v
        if v > 0.005:
            d.polygon([(fx_out, y1 - inner), (fx_in, y1 - inner), (fx_in + sign * (skew * 0.8), y0 + inner),
                       (fx_out + sign * (skew * 0.8), y0 + inner)], fill=col)
        name = _headline(ctx, s["name"], 46 * u, s.get("color", "accent" if side == "left" else "paper"))
        # Name sits on top of its bar, flush with the outer edge (headline canvases carry padding).
        pad = name.height * 0.18
        th.paste(c, name, x_out - sign * pad, y0 + pad, anchor="bottomleft" if sign < 0 else "bottomright")
        lab = s.get("label")
        state = s.get("down_label", "KO") if v <= 0.005 else lab
        if state:
            lc = _chip(ctx, state, 20 * u, fill="ink", ink="danger" if v <= 0.005 else "paper")
            th.paste(c, lc, x_out, y1 + 18 * u, anchor="left" if sign < 0 else "right")
    cnt = p.get("counter")
    if cnt:
        val = max(0, int(cnt.get("from", 99) - ctx.t_out * cnt.get("rate", 1.0)))
        r = cw * 0.5
        cx, cy = W / 2, top + 34 * u + bar_h / 2
        hexa = [(cx + r * math.cos(math.pi / 3 * i + math.pi / 6), cy + r * math.sin(math.pi / 3 * i + math.pi / 6)) for i in range(6)]
        d.polygon(hexa, fill=th.color(ctx.theme, "ink"))
        hexb = [(cx + (r - 5 * u) * math.cos(math.pi / 3 * i + math.pi / 6),
                 cy + (r - 5 * u) * math.sin(math.pi / 3 * i + math.pi / 6)) for i in range(6)]
        d.polygon(hexb, outline=th.color(ctx.theme, "accent"), width=max(1, int(3 * u)))
        num = _headline(ctx, str(val), 54 * u, "paper", shadow="none")
        th.paste(c, num, cx, cy)


def progress(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    frac = min(1.0, ctx.t_out / max(0.01, ctx.total))
    h = p.get("height", 8) * u
    y = 0 if p.get("y", 0.0) < 0.5 else H - h
    ImageDraw.Draw(c).rectangle([0, y, W * frac, y + h], fill=th.color(ctx.theme, p.get("color", "accent")))


def callout(c, t, dur, p, ctx):
    """Circle a tracked thing and point a label at it (-1 LEG, THIS GUY, $4,000 WATCH)."""
    W, H, u = ctx.W, ctx.H, ctx.u
    box = ctx.target(p["target"])
    if box is None:
        return
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    k = ease_out(t / 0.3)
    col = th.color(ctx.theme, p.get("color", "accent"))
    d = ImageDraw.Draw(c)
    r = max(w, h) * 0.75 * (0.6 + 0.4 * k)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=max(2, int(4 * u)))
    lx, ly = cx + W * p.get("dx", 0.12) * (1 if cx < W * 0.7 else -1), cy + H * p.get("dy", -0.18)
    ex = cx + (lx - cx) * k
    ey = cy + (ly - cy) * k
    ang = math.atan2(ey - cy, ex - cx)
    d.line([(cx + r * math.cos(ang), cy + r * math.sin(ang)), (ex, ey)], fill=col, width=max(2, int(3 * u)))
    if k > 0.6:
        a = ease_out((k - 0.6) / 0.4)
        big = _headline(ctx, p["text"], 80 * u, "paper")
        box_img = _chip(ctx, p["text"], 64 * u, fill=p.get("color", "accent"), ink="ink", role="display")
        th.paste(c, box_img, ex, ey, alpha=a)
        if p.get("label"):
            lc = _chip(ctx, p["label"], 26 * u, fill="ink", ink="paper")
            th.paste(c, lc, ex - box_img.width * 0.25, ey - box_img.height * 0.5 - 16 * u, alpha=a)


def tags(c, t, dur, p, ctx):
    """Name tags (and an optional stat line) pinned above each tracked head, constant size on screen."""
    u = ctx.u
    for name, cfg in p.get("targets", {}).items():
        if cfg.get("until") is not None and ctx.t_out > cfg["until"]:
            sub = None
        else:
            sub = cfg.get("sub")
        box = ctx.target(name)
        if box is None:
            continue
        x, y, w, h = box
        col = cfg.get("color", "accent")
        nm = _chip(ctx, name, 40 * u * cfg.get("size", 1.0), fill=col, ink="ink", role="display")
        cx = x + w / 2
        tip = y - h * 0.25
        d = ImageDraw.Draw(c)
        tri = 12 * u
        d.polygon([(cx - tri, tip - tri * 1.4), (cx + tri, tip - tri * 1.4), (cx, tip)], fill=th.color(ctx.theme, col))
        base = tip - tri * 1.3
        if sub:
            sc = _chip(ctx, sub, 22 * u, fill="ink", ink=col)
            th.paste(c, sc, cx, base - sc.height / 2)
            base -= sc.height + 2 * u
        th.paste(c, nm, cx, base - nm.height / 2)


def ticker(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    k = ease_out(t / 0.3)
    img = _chip(ctx, p["text"], 22 * u, fill="ink", ink="paper")
    x, y = W * p.get("x", 0.03), H * p.get("y", 0.93 if W > H else 0.76)
    th.paste(c, img, x + 26 * u - (1 - k) * 40 * u, y, alpha=k, anchor="left")
    if p.get("dot"):
        on = int(ctx.t_out * 2) % 2 == 0
        r = 7 * u
        ImageDraw.Draw(c).ellipse([x + 4 * u, y - r, x + 4 * u + 2 * r, y + r],
                                  fill=th.color(ctx.theme, p["dot"], 255 if on else 90))


def watermark(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    img = _chip(ctx, p["text"], 22 * u, fill="ink", ink="paper")
    corner = p.get("corner", "bottom-right")
    mx, my = W * 0.025, H * (0.03 if W > H else 0.23)
    x = W - mx if "right" in corner else mx
    y = H - my if "bottom" in corner else my
    th.paste(c, img, x, y, alpha=p.get("alpha", 0.85),
             anchor=("bottom" if "bottom" in corner else "top") + ("right" if "right" in corner else "left"))


def letterbox(c, t, dur, p, ctx):
    W, H = ctx.W, ctx.H
    k = ease_out(t / 0.2) * (1 - ease_out((t - (dur - 0.15)) / 0.15))
    bar = H * p.get("amount", 0.09) * k
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, W, bar], fill=(0, 0, 0, 255))
    d.rectangle([0, H - bar, W, H], fill=(0, 0, 0, 255))


def endcard(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    k = ease_out(t / 0.35)
    c.alpha_composite(Image.new("RGBA", c.size, (18, 18, 20, int(255 * p.get("dim", 0.72) * k))))
    y = H * 0.42
    big = _headline(ctx, p["title"], 170 * u, "paper", shadow="soft")
    s = _fit(big, W * 0.8)
    th.paste(c, big, W / 2, y + (1 - k) * 30 * u, scale=s, alpha=k)
    bw = big.width * s * 0.7 * ease_out((t - 0.2) / 0.4)
    ry = y + big.height * s * 0.42
    ImageDraw.Draw(c).rectangle([W / 2 - bw / 2, ry, W / 2 + bw / 2, ry + 4 * u], fill=th.color(ctx.theme, "accent"))
    yy = ry + 50 * u
    for i, (key, fill, ink, size, role) in enumerate((("tagline", None, "paper", 40, "body"),
                                                        ("link", "accent", "ink", 34, "label"),
                                                        ("handle", None, "muted", 26, "label"))):
        if not p.get(key):
            continue
        a = ease_out((t - 0.35 - i * 0.12) / 0.3)
        if fill:
            img = _chip(ctx, p[key], size * u, fill=fill, ink=ink)
        else:
            img = _headline(ctx, p[key], size * u, ink, role=role, shadow="none")
        th.paste(c, img, W / 2, yy + img.height / 2, alpha=a)
        yy += img.height + 22 * u


# Stacking order (higher draws later, on top): labels pinned to things < HUD/branding < big type < transitions.
Z = {"tags": 10, "callout": 12, "captions": 14, "caption": 14, "lower_third": 16, "progress": 18,
     "meters": 20, "ticker": 22, "watermark": 22, "chip": 24, "letterbox": 26, "versus": 30, "title": 30,
     "slam": 32, "stamp": 32, "vs": 30, "endcard": 40, "flash": 50}

COMPONENTS = {f.__name__: f for f in (flash, slam, versus, title, lower_third, chip, caption, captions, meters,
                                      progress, callout, tags, ticker, watermark, letterbox, endcard)}
# Back-compat names from the first motion module.
COMPONENTS["vs"] = lambda c, t, d, p, ctx: versus(c, t, d, {**p, "left_color": p.get("left_color", "accent"),
                                                             "right_color": p.get("right_color", "paper")}, ctx)
COMPONENTS["stamp"] = slam
