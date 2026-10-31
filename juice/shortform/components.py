"""Components: themed, animated overlay building blocks.

Every component is `draw(canvas, t, dur, p, ctx)`:
  canvas  RGBA frame (output size) the component draws onto
  t       seconds since this layer started; dur = how long the layer lives
  p       the component's params from the spec / event
  ctx     render context: ctx.theme, ctx.W, ctx.H, ctx.u (1.0 at 1080px short side), ctx.t_out
          (absolute output time), ctx.meter(id) (current meter value 0-100), ctx.target(name)
          (tracked box in output pixels, or None when hidden/out of frame), ctx.words (captions)

`"style": {"animate": false}` turns off every text animation (title, chip, captions appear and leave on a
frame, no pop on the active caption word) for creators who want static type.

Components are genre-neutral. A fight uses versus + meters + slam + callout; a story uses
lower_third + captions + title; rage bait uses captions(loud) + slam + chip. Colors are theme roles
("accent", "ink", "paper", "danger", "accent2") or [r, g, b].

  flash        {dur=0.25, color="paper", strength=0.9}
  slam         {text, color="paper", sub?, sub_fill="danger", y=0.42, size=1.0}
               giant word that fills the screen faintly, then snaps to size (FIGHT!, K.O., WAIT...)
  versus       {left, right, left_color="accent", right_color="paper", premise?}
  title        {text | lines[], kicker?, sub?, color="accent", bar="accent2" | false, y=0.62, size=1.0, stagger=0.09}
               kinetic: lines spring in staggered, accent bar wipes, kicker drops, all slide out
  lower_third  {name, role?, side="left", y=0.78}
  chip         {text, badge?, x=0.5, y=0.5, fill="ink", ink="paper", size=1.0, anchor="center", slide=true}
  caption      {text, y=0.75, size=1.0, color="paper"}                      static subtitle line
  captions     {y=0.72, size=1.0, max_words=3, active=theme.caption_active, color="paper", uppercase=true,
                box=true, gap=0.18, inactive?}  active word on a rounded accent block; gap in em;
                inactive = color for the other words of the chunk (karaoke look: box=false, inactive="muted")
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


def ease_spring(t, s=1.9):
    """Snappy overshoot (lands past 1 then settles): the feel of a kinetic-type pop, not a fade."""
    return ease_back(t, s)


def ease_expo(t):
    t = min(max(t, 0.0), 1.0)
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)


def animate(ctx) -> bool:
    """style.animate = false turns every text animation off: things appear and leave on a frame."""
    sp = getattr(ctx, "spec", None)
    return True if sp is None else bool(sp.style.get("animate", True))


def exit_k(t, dur, d=0.22):
    """0 -> 1 over the last `d` seconds of a layer, so things leave on purpose instead of vanishing."""
    return ease_out((t - (dur - d)) / d) if dur > d * 2 else 0.0


def _rounded(w: int, h: int, r: int, fill) -> Image.Image:
    img = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=fill)
    return img


def _pill(ctx, text, size, fill, ink, role="display"):
    """A word on a rounded accent block (the active caption word)."""
    def mk():
        base = th.headline(ctx.theme, text, size, fill=ink, role=role, shadow="none")
        f = ctx.theme["fonts"][role]
        px, py = int(size * 0.22), int(size * 0.02)
        box = _rounded(base.width - int(size * 0.3) + 2 * px, int(size * 1.18) + 2 * py, int(size * 0.18),
                       th.color(ctx.theme, fill))
        out = Image.new("RGBA", (max(box.width, base.width), max(box.height, base.height)), (0, 0, 0, 0))
        out.alpha_composite(box, ((out.width - box.width) // 2, (out.height - box.height) // 2))
        out.alpha_composite(base, ((out.width - base.width) // 2, (out.height - base.height) // 2))
        return out
    return cached(_key(ctx, "pill", text, int(size), str(fill), str(ink), role), mk)


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
        # Typed on: the rule/stakes of the video.
        text = p["premise"]
        n = int(len(text) * min(1, (t - 0.55) / 0.6))
        cursor = "_" if n < len(text) and int(t * 6) % 2 == 0 else ""
        pc = _chip(ctx, text[:n] + cursor, 34 * u, fill="ink", ink="paper") if n else None
        if pc:
            th.paste(c, pc, W / 2, py)


def title(c, t, dur, p, ctx):
    """Kinetic title: each line springs in (staggered), an accent bar wipes under it, the kicker drops
    in from above, the sub chip rises from below; everything slides out on the last 0.22s."""
    W, H, u = ctx.W, ctx.H, ctx.u
    lines = [str(x) for x in (p.get("lines") or [p["text"]])]
    size = 230 * u * p.get("size", 1.0) * (0.72 if len(lines) > 1 else 1.0)
    col = p.get("color", "accent")
    imgs = [_headline(ctx, ln, size, col, outline=p.get("outline", max(3, int(6 * u)))) for ln in lines]
    s = min(_fit(im, W * 0.84) for im in imgs)
    pitch = max(im.height for im in imgs) * s * 0.74
    total = pitch * len(lines)
    A = animate(ctx)
    out = exit_k(t, dur) if A else 0.0
    y0 = H * p.get("y", 0.62) - total / 2 + pitch / 2 - out * 60 * u
    stagger = p.get("stagger", 0.09) if A else 0.0
    for i, im in enumerate(imgs):
        k = ease_spring((t - stagger * i) / 0.42) if A else 1.0
        a = (min(1.0, max(0.0, (t - stagger * i) / 0.08)) if A else 1.0) * (1 - out)
        th.paste(c, im, W / 2, y0 + i * pitch + (1 - k) * 70 * u, scale=s * (0.66 + 0.34 * k), alpha=a)
    # accent bar wipes left -> right under the last line
    kb = ease_expo((t - 0.18 - stagger * (len(lines) - 1)) / 0.45) if A else 1.0
    bar_w = min(W * 0.5, max(im.width for im in imgs) * s * 0.6) * kb
    if bar_w > 2 and out < 1 and p.get("bar", "accent2"):
        by = y0 + (len(lines) - 1) * pitch + pitch * 0.5 + 4 * u
        bar = _rounded(int(bar_w), int(10 * u), int(5 * u), th.color(ctx.theme, p.get("bar", "accent2")))
        th.paste(c, bar, W / 2 - (W * 0.5 * (1 - kb)) * 0, by, alpha=1 - out)
    if p.get("kicker"):
        kk = ease_spring((t - 0.04) / 0.36) if A else 1.0
        kc = _chip(ctx, p["kicker"], 38 * u, fill=p.get("kicker_fill", "accent2"), ink="ink")
        th.paste(c, kc, W / 2, y0 - pitch * 0.5 - 30 * u - (1 - kk) * 60 * u,
                 alpha=(min(1.0, max(0.0, (t - 0.04) / 0.08)) if A else 1.0) * (1 - out), scale=0.8 + 0.2 * kk)
    if p.get("sub"):
        ks = ease_spring((t - 0.3 - stagger * (len(lines) - 1)) / 0.36) if A else 1.0
        sc = _chip(ctx, p["sub"], 34 * u, fill="ink", ink="paper")
        th.paste(c, sc, W / 2, y0 + (len(lines) - 0.5) * pitch + 50 * u + (1 - ks) * 40 * u,
                 alpha=(min(1.0, max(0.0, (t - 0.3) / 0.08)) if A else 1.0) * (1 - out), scale=0.8 + 0.2 * ks)


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
    """Label chip that pops in with overshoot (and slides out at the end). `badge` glues a small
    accent block to its left ("01", "NEW", "?")."""
    W, H, u = ctx.W, ctx.H, ctx.u
    size = 34 * u * p.get("size", 1.0)
    img = _chip(ctx, p["text"], size, fill=p.get("fill", "ink"), ink=p.get("ink", "paper"))
    if p.get("badge"):
        b = _chip(ctx, str(p["badge"]), size, fill=p.get("badge_fill", "accent2"), ink="ink", role="display")
        joined = Image.new("RGBA", (img.width + b.width, max(img.height, b.height)), (0, 0, 0, 0))
        joined.alpha_composite(b, (0, (joined.height - b.height) // 2))
        joined.alpha_composite(img, (b.width, (joined.height - img.height) // 2))
        img = joined
    A = animate(ctx)
    out = exit_k(t, dur) if A else 0.0
    k = ease_spring(t / 0.34) if (A and p.get("slide", True)) else 1.0
    th.paste(c, img, W * p.get("x", 0.5) - (1 - k) * 50 * u, H * p.get("y", 0.5) + out * 30 * u,
             scale=(0.6 + 0.4 * k) * (1 - 0.1 * out), alpha=(min(1.0, t / 0.06) if A else 1.0) * (1 - out),
             anchor=p.get("anchor", "center"))


def caption(c, t, dur, p, ctx):
    W, H, u = ctx.W, ctx.H, ctx.u
    img = _headline(ctx, p["text"], 64 * u * p.get("size", 1.0), p.get("color", "paper"), role="body", shadow="soft")
    th.paste(c, img, W / 2, H * p.get("y", 0.75), scale=_fit(img, W * 0.9))


def _caption_chunks(words, m, cuts=()):
    """For each word index, the (start, end) index range of the chunk it belongs to.
    A chunk never crosses a cut, a pause > 0.6s, or sentence punctuation."""
    out, a = [], 0
    for k, w in enumerate(words):
        last = k == len(words) - 1
        ends = w["text"].rstrip()[-1:] in ".?!," if w["text"].strip() else False
        pause = (not last) and words[k + 1]["start"] - w["end"] > 0.6
        cut = (not last) and any(w["start"] < c <= words[k + 1]["start"] for c in cuts)
        if last or ends or pause or cut or (k - a + 1) >= m:
            out.extend([(a, k + 1)] * (k + 1 - a))
            a = k + 1
    return out


def captions(c, t, dur, p, ctx):
    """Word-by-word captions from the transcript: the spoken chunk in plain white with a thin black
    stroke, no colors by default. Opt in to a highlight with `"active": "accent"` (the spoken word in that
    color) or `"box": true` (the spoken word on a rounded accent block); `size`, `gap` and `stroke` scale
    the type, the word spacing and the stroke."""
    W, H, u = ctx.W, ctx.H, ctx.u
    words = ctx.words or []
    now = ctx.t_out
    i = next((k for k, w in enumerate(words) if w["start"] <= now < w["end"] + 0.25), None)
    if i is None:
        return
    m = p.get("max_words", 3)
    # Chunks follow the phrasing: break on punctuation or a pause, never mid-phrase just to fill a slot.
    cuts = tuple(getattr(getattr(ctx, "tl", None), "starts", ())[1:])
    chunks = cached(("capchunks", id(words), m, cuts), lambda: _caption_chunks(words, m, cuts))
    a, b = chunks[i]
    chunk = words[a:b]
    size = 58 * u * p.get("size", 1.0)
    up = p.get("uppercase", True)
    box = p.get("box", False)
    hi = p.get("active") or (ctx.theme.get("caption_active", "accent") if box else None)
    role = p.get("role", "display")
    stroke = p.get("stroke", max(1, round(size * 0.035)))
    base, imgs = [], []
    for k, w in enumerate(chunk):
        txt = w["text"].upper() if up else w["text"]
        plain = _headline(ctx, txt, size, p.get("color", "paper"), role=role, shadow="none", outline=stroke)
        base.append(plain)
        active = (a + k) == i
        if active and box:
            imgs.append(_pill(ctx, txt, size, hi, p.get("box_ink", "ink"), role=role))
        elif active and hi:
            imgs.append(_headline(ctx, txt, size, hi, role=role, shadow="none", outline=stroke))
        elif p.get("inactive"):     # karaoke: the words not yet / already spoken in a quieter color
            imgs.append(_headline(ctx, txt, size, p["inactive"], role=role, shadow="none", outline=stroke))
        else:
            imgs.append(plain)
    # Space words by their glyph widths, not their canvases (every headline carries shadow padding).
    pad = int(size * 0.3) + int(size * ctx.theme["shadow"].get("offset", 0.06))
    widths = [max(1, im.width - 2 * pad) for im in base]
    gap = size * p.get("gap", 0.28)
    total = sum(widths) + gap * (len(base) - 1)
    s = _fit(Image.new("RGBA", (int(total), 1)), W * 0.9)
    x = W / 2 - total * s / 2
    pop = 1 + 0.06 * (1 - ease_spring((now - words[i]["start"]) / 0.16)) if animate(ctx) else 1.0
    y = H * p.get("y", 0.72)
    for k, (im, wd) in enumerate(zip(imgs, widths)):
        active = (a + k) == i
        sc = s * (pop if active else 1)
        th.paste(c, im, x + wd * s / 2, y, scale=sc)
        x += wd * s + gap * s


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
