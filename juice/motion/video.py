"""Video: scenes and overlays on one output clock, voice-over, a music bed and hits, rendered fast.

    from juice.motion import Video, scenes as s
    v = Video("landscape", "short", brand="juice.json")
    take = v.source("takes/a.mov")
    vo = v.voice(take, 0.0, 1.6, at=0.0)                  # voice from a take, placed on the output clock
    v.scene(s.Footage(v.source("takes/a.mov", 0, 1.6)), 0, 1.6)
    v.scene(s.LogoReveal(badge="beta"), 1.6, 3.6, transition="flash")
    v.overlay(s.Headline("Now in beta.", highlight={"beta.": "accent"}), 0.3, 1.6)
    v.music("hype")
    v.render("out.mp4", preview=True)   # 480p, every other frame, seconds
    v.render("out.mp4")                 # full quality, split across CPU cores

Every scene draws in design units (1920x1080 or 1080x1920), so preview and full renders are the
same picture at different sizes. Renders split the frames into chunks, one process per chunk, and
concatenate the encoded pieces; the audio mix runs at the same time.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import skia

from .. import brand as brand_mod
from .. import encode
from .core import Ctx, Source, clamp, col, e_in, e_inout, lerp, logo_image, prog
from .scenes import Piece

ORIENTATIONS = {"portrait": (1080, 1920), "landscape": (1920, 1080)}
TRANSITIONS = ("cut", "flash", "iris", "logo", "whip", "glitch", "fade", "dissolve", "wipe")
# The soft transitions and how long they take: fade dips through black around the cut; dissolve and
# wipe draw the outgoing scene under the incoming one.
SOFT = {"fade": 0.2, "dissolve": 0.3, "wipe": 0.45}
# What a hit does to the picture: (flash alpha, shake px at 1080, chroma px), and its sound.
HITS = {"boom": (0.6, 12, 12), "slam": (0.25, 9, 0), "land": (0.3, 22, 10), "flash": (0.5, 0, 0),
        "shake": (0, 14, 0), "glitch": (0, 0, 12), "whoosh": (0, 0, 0), "pop": (0, 0, 0), "click": (0, 0, 0),
        "riser": (0, 0, 0), "type": (0, 0, 0), "braam": (0, 0, 0),
        # quiet ones, for the clean look: no picture effect
        "sub": (0, 0, 0), "tick": (0, 0, 0), "air": (0, 0, 0), "swell": (0, 0, 0)}


@dataclass
class Placed:
    piece: Piece
    start: float
    end: float
    transition: str = "cut"
    energy: int | None = None


@dataclass
class Voice:
    path: str
    start: float
    end: float
    at: float
    isolate: bool = True

    def out(self, src_t: float) -> float:
        """Output time of a moment in the take (for timing graphics to a spoken word)."""
        return self.at + (src_t - self.start)


@dataclass
class Hit:
    at: float
    kind: str
    amount: float = 1.0
    until: float | None = None   # for "type" (key ticks) and "riser" (ends at `at`, starts at `until`)


@dataclass
class Video:
    orientation: str | None = None   # default: $JUICE_ORIENTATION (set by `juice motion --orientation`), else landscape
    length: str = "short"
    brand: str | dict | None = None
    fps: int = 30
    scenes: list = field(default_factory=list)
    overlays: list = field(default_factory=list)
    voices: list = field(default_factory=list)
    hits: list = field(default_factory=list)
    bed: dict | None = None

    def __post_init__(self):
        self.orientation = self.orientation or os.environ.get("JUICE_ORIENTATION", "landscape")
        if self.orientation not in ORIENTATIONS:
            raise ValueError("orientation must be portrait or landscape")
        if self.length not in ("short", "long"):
            raise ValueError("length must be short or long")
        self.W, self.H = ORIENTATIONS[self.orientation]
        if isinstance(self.brand, (str, Path)):
            p = Path(self.brand).expanduser()
            self.kit = brand_mod.load(p) if p.exists() else brand_mod.find(str(self.brand), Path.cwd())
        else:
            self.kit = self.brand

    # --- building ---------------------------------------------------------------------------------

    def source(self, path, start=0.0, end=None, speed=1.0, crop=None) -> Source:
        """A range of a video file; crop = (x, y, w, h) in source pixels."""
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return Source(path, start, end, speed, crop)

    def scene(self, piece: Piece, start: float, end: float, transition="cut", energy: int | None = None):
        """A full-frame scene. transition: cut | flash (white hit + boom) | iris (circle opens) |
        logo (opens through the brand mark) | whip (motion blur + whoosh) | glitch, or the soft ones:
        fade (dip through black) | dissolve | wipe (slides over the last scene, with air). energy 0-3
        shapes the music under it (default: 2; the first scene 0)."""
        if transition not in TRANSITIONS:
            raise ValueError(f"transition must be one of {', '.join(TRANSITIONS)}")
        self.scenes.append(Placed(piece, start, end, transition, energy))
        return piece

    def overlay(self, piece: Piece, start: float, end: float):
        self.overlays.append(Placed(piece, start, end))
        return piece

    def voice(self, take: Source | str, start: float, end: float, at: float, isolate=True) -> Voice:
        """Voice from a take's audio (start..end in the take) placed at output time `at`: the speaker
        can keep talking over other scenes. isolate strips the room (chatter, traffic, hum) first,
        see juice/isolate.py; isolate=False keeps the take's own ambience."""
        v = Voice(str(take.path if isinstance(take, Source) else take), start, end, at, isolate)
        self.voices.append(v)
        return v

    def hit(self, at: float, kind="boom", amount=1.0, until=None):
        """A punctuation: boom | slam | land | flash | shake | glitch | whoosh | pop | click | braam |
        riser (builds from `until` to `at`) | type (key ticks from at to until); quiet ones for the
        clean look: sub (soft low thump) | tick | air (soft whoosh) | swell (breath in, ending at `at`)."""
        if kind not in HITS:
            raise ValueError(f"hit kind must be one of {', '.join(HITS)}")
        self.hits.append(Hit(at, kind, amount, until))

    def music(self, style="hype", level=0.0, duck=12.0, beat0=None, bpm=None, file=None):
        """A bed: a built-in style (hype | tech | calm | keynote | future, see beds.py) arranged to the
        scenes' energies, bars starting at beat0 (default: the first transition, so the downbeat lands
        on a cut); or a music file. The mix sets its volume by loudness, not a fader: between lines it
        plays `level` dB from the voice's loudness (0 = as loud as the voice), under a line it sits
        `duck` dB lower."""
        self.bed = {"style": style, "level": level, "duck": duck, "beat0": beat0, "bpm": bpm, "file": file}

    def beat(self, k: float) -> float:
        """Output time of beat k of the bed (k may be fractional or negative)."""
        from ..media.beds import STYLES
        bpm = (self.bed or {}).get("bpm") or STYLES[(self.bed or {}).get("style") or "hype"]["bpm"]
        return self._beat0() + k * 60 / bpm

    def _beat0(self) -> float:
        if self.bed and self.bed.get("beat0") is not None:
            return self.bed["beat0"]
        starts = sorted(p.start for p in self.scenes if p.start > 0)
        return starts[0] if starts else 0.0

    @property
    def duration(self) -> float:
        return max([p.end for p in self.scenes + self.overlays] + [v.at + v.end - v.start for v in self.voices] + [0])

    def sections(self) -> list[tuple[float, int]]:
        out = []
        for i, p in enumerate(sorted(self.scenes, key=lambda p: p.start)):
            e = p.energy if p.energy is not None else (0 if i == 0 else 2)
            if not out or out[-1][1] != e:
                out.append((p.start, e))
        return out

    def _all_hits(self) -> list[Hit]:
        """Explicit hits plus the ones each transition implies."""
        implied = {"flash": ["boom"], "iris": ["whoosh", "pop"], "logo": ["whoosh", "boom"], "whip": ["whoosh"],
                   "glitch": ["glitch"], "wipe": ["air"]}
        out = list(self.hits)
        for p in self.scenes:
            for k in implied.get(p.transition, []):
                out.append(Hit(p.start - (0.35 if k in ("whoosh", "air") else 0), k, 0.6 if p.transition != "flash" else 1.0))
        return out

    # --- drawing ----------------------------------------------------------------------------------

    def _ctx(self, scale=1.0) -> Ctx:
        return Ctx(self.W, self.H, self.fps, self.kit, scale)

    def _post_amounts(self, t: float, hits: list[Hit]):
        flash = shake = chroma = 0.0
        for h in hits:
            fa, sa, ca = HITS[h.kind]
            dt = t - h.at
            if fa and 0 <= dt < 0.16:
                flash = max(flash, fa * h.amount * (1 - dt / 0.16))
            if sa and 0 <= dt < 0.45:
                shake = max(shake, sa * h.amount * math.exp(-dt * 9))
            if ca and -0.07 <= dt < 0.13:
                chroma = max(chroma, ca * h.amount * (1 - abs(dt) / 0.13))
        return flash, shake, chroma

    def draw(self, surf: skia.Surface, ctx: Ctx, t: float, hits: list[Hit], grain=None, fi=0) -> np.ndarray:
        c = surf.getCanvas()
        c.save()
        c.scale(ctx.scale, ctx.scale)
        c.clear(ctx.c("ink"))
        flash, shake, chroma = self._post_amounts(t, hits)
        c.save()
        if shake > 0:
            s = shake * ctx.u
            c.translate(ctx.W / 2 + math.sin(t * 93) * s, ctx.H / 2 + math.cos(t * 71) * s * 0.7)
            c.scale(1.03, 1.03)
            c.translate(-ctx.W / 2, -ctx.H / 2)
        whip = 0.0
        for p in self.scenes:
            if not (p.start - 1e-6 <= t < p.end):
                continue
            lt, d = t - p.start, 0.34
            k = lt / d
            c.save()
            if p.transition in ("iris", "logo") and k < 1:
                self._iris(c, ctx, p, k, lambda: p.piece.draw(c, ctx, lt, p.end - p.start, t))
            elif p.transition in ("dissolve", "wipe") and lt < SOFT[p.transition]:
                self._blend(c, ctx, p, lt, t)
            else:
                p.piece.draw(c, ctx, lt, p.end - p.start, t)
            c.restore()
            if p.transition == "whip" and lt < 0.1:
                whip = max(whip, 1 - lt / 0.1)
            if p.transition == "glitch" and lt < 0.13:
                chroma = max(chroma, 10 * (1 - lt / 0.13))
        for p in self.overlays:
            if p.start - 1e-6 <= t < p.end:
                c.save()
                p.piece.draw(c, ctx, t - p.start, p.end - p.start, t)
                c.restore()
        c.restore()
        if flash > 0:
            c.drawRect(skia.Rect.MakeWH(ctx.W, ctx.H), skia.Paint(Color=col((255, 255, 255), flash)))
        dip = max([1 - abs(t - p.start) / SOFT["fade"] for p in self.scenes if p.transition == "fade"] + [0])
        if dip > 0:
            c.drawRect(skia.Rect.MakeWH(ctx.W, ctx.H), skia.Paint(Color=col((0, 0, 0), e_inout(dip))))
        end = self.duration
        fo = prog(t, end - 0.6, 0.6)
        if fo > 0:
            c.drawRect(skia.Rect.MakeWH(ctx.W, ctx.H), skia.Paint(Color=col((0, 0, 0), e_in(fo) if fo < 1 else 1)))
        c.restore()
        arr = surf.toarray(colorType=skia.kRGBA_8888_ColorType)[..., :3]
        ca = int(chroma * ctx.u * ctx.scale)
        if ca:
            arr = arr.copy()
            arr[..., 0] = np.roll(arr[..., 0], ca, 1)
            arr[..., 2] = np.roll(arr[..., 2], -ca, 1)
        if whip > 0:
            import cv2
            arr = cv2.blur(arr, (int(60 * whip * ctx.scale) + 1, 1))
        if grain is not None:
            arr = np.clip(arr.astype(np.int16) + grain[fi % len(grain)], 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    def _blend(self, c, ctx, p, lt, t):
        """The scene that ends where p starts, still running underneath, with p dissolving in or
        wiping across it (from the right, with a little parallax)."""
        prev = next((q for q in self.scenes if abs(q.end - p.start) < 1e-3 and q is not p), None)
        k = e_inout(clamp(lt / SOFT[p.transition]))
        if prev is not None:
            c.save()
            if p.transition == "wipe":
                c.translate(-k * ctx.W * 0.12, 0)
            prev.piece.draw(c, ctx, t - prev.start, prev.end - prev.start + SOFT[p.transition], t)
            c.restore()
        if p.transition == "dissolve":
            c.saveLayerAlpha(None, int(255 * k))
            p.piece.draw(c, ctx, lt, p.end - p.start, t)
            c.restore()
            return
        x = ctx.W * (1 - k)
        c.save()
        c.clipRect(skia.Rect.MakeXYWH(x, 0, ctx.W - x, ctx.H))
        c.translate((1 - k) * ctx.W * 0.25, 0)
        p.piece.draw(c, ctx, lt, p.end - p.start, t)
        c.restore()
        c.drawRect(skia.Rect.MakeXYWH(x - 60 * ctx.u, 0, 60 * ctx.u, ctx.H),
                   skia.Paint(Shader=skia.GradientShader.MakeLinear(
                       [skia.Point(x - 60 * ctx.u, 0), skia.Point(x, 0)], [col((0, 0, 0), 0), col((0, 0, 0), 0.35 * (1 - k))])))

    def _iris(self, c, ctx, p, k, draw):
        """Open the scene through a growing shape: a circle, or the brand mark (logo)."""
        s = lerp(0, max(ctx.W, ctx.H) * 1.6, e_inout(clamp(k)))
        img = logo_image(ctx) if p.transition == "logo" else None
        if img is None:
            path = skia.Path()
            path.addCircle(ctx.W / 2, ctx.H / 2, s / 2)
            c.save()
            c.clipPath(path, True)
            draw()
            c.restore()
            c.drawPath(path, skia.Paint(AntiAlias=True, Color=ctx.c("accent"), StrokeWidth=10 * ctx.u,
                                        Style=skia.Paint.kStroke_Style))
            return
        c.saveLayer()
        draw()
        sc = s / max(img.width(), img.height())
        w, h = img.width() * sc, img.height() * sc
        c.drawImageRect(img, skia.Rect.MakeXYWH(ctx.W / 2 - w / 2, ctx.H / 2 - h / 2, w, h), skia.SamplingOptions(),
                        skia.Paint(BlendMode=skia.BlendMode.kDstIn))
        c.restore()

    def sources(self) -> list[Source]:
        out = []
        for p in self.scenes + self.overlays:
            for s in p.piece.sources():
                if s not in out:
                    out.append(s)
        return out

    def _configure(self, scale: float, fps: int):
        for s in self.sources():
            s.close()
            aspect = _aspect(s.path, s.crop)
            long_side = max(self.W, self.H) * scale
            size = (long_side, long_side / aspect) if aspect >= 1 else (long_side * aspect, long_side)
            s.configure(fps, size)

    # --- rendering --------------------------------------------------------------------------------

    def render(self, out, preview=False, workers=None, crf=16) -> dict:
        """Render to out (mp4). preview = 480p on the short side at half the frame rate, fast encode.
        Returns timings and paths."""
        t_start = time.time()
        out = Path(out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        scale = 480 / min(self.W, self.H) if preview else 1.0
        fps = self.fps // 2 if preview else self.fps
        n = int(round(self.duration * fps))
        workers = workers or max(1, min(8, (os.cpu_count() or 4) - 2))
        chunk = max(int(fps * 1.5), math.ceil(n / workers))
        ranges = [(a, min(n, a + chunk)) for a in range(0, n, chunk)]
        tmp = Path(tempfile.mkdtemp(prefix="juice-", dir=out.parent))
        enc = encode.video_args({"bit_depth": 8, "crf": 28, "preset": "ultrafast"} if preview
                                else {"bit_depth": 10, "crf": crf, "preset": "medium"})
        jobs = [(k, a, b, scale, fps, str(tmp / f"part{k:03d}.mp4"), enc, preview) for k, (a, b) in enumerate(ranges)]
        for s in self.sources():
            s.close()
        # Workers are forked and read the video from this global, so nothing (skia images, open
        # readers) has to be pickled.
        global _JOB
        _JOB = self
        ctx = mp.get_context("fork")
        with ctx.Pool(len(jobs)) as pool:
            pending = pool.map_async(_render_chunk, jobs)
            wav = self.render_audio(tmp / "mix.wav", preview=preview)
            parts = pending.get()
        t_video = time.time() - t_start
        (tmp / "parts.txt").write_text("".join(f"file '{p}'\n" for p in parts))
        _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "parts.txt"), "-i", str(wav),
              "-map", "0:v", "-map", "1:a", *encode.copy_args({"bit_depth": 8 if preview else 10}),
              "-c:a", "aac", "-b:a", "128k" if preview else "256k",
              "-shortest", "-movflags", "+faststart", str(out)])
        shutil.rmtree(tmp, ignore_errors=True)
        report = {"output": str(out), "preview": preview, "size": [round(self.W * scale), round(self.H * scale)],
                  "fps": fps, "frames": n, "workers": len(jobs), "duration_s": round(self.duration, 3),
                  "render_s": round(time.time() - t_start, 1), "video_s": round(t_video, 1),
                  "audio": getattr(self, "audio_report", None)}
        return report

    def stills(self, times: list[float], out, scale=0.5, cols=4) -> Path:
        """Frames at the given output times on one contact sheet, labeled with their time."""
        import cv2
        out = Path(out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        self._configure(scale, self.fps)
        ctx = self._ctx(scale)
        surf = skia.Surface(round(self.W * scale), round(self.H * scale))
        hits = self._all_hits()
        frames = []
        for t in times:
            for s in self.sources():
                s.close()
            f = self.draw(surf, ctx, t, hits)
            f = cv2.putText(f.copy(), f"{t:.2f}s", (12, 34), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            frames.append(f)
        for s in self.sources():
            s.close()
        rows = [frames[i:i + cols] for i in range(0, len(frames), cols)]
        h, w = frames[0].shape[:2]
        sheet = np.zeros((len(rows) * (h + 8), cols * (w + 8), 3), np.uint8)
        for r, row in enumerate(rows):
            for k, f in enumerate(row):
                sheet[r * (h + 8):r * (h + 8) + h, k * (w + 8):k * (w + 8) + w] = f
        cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
        return out

    def render_audio(self, out: Path, preview=False) -> Path:
        from . import mix
        return mix.render(self, Path(out))


_JOB: Video | None = None


def _render_chunk(job):
    """One process: frames a..b of the video, encoded on their own (concatenated afterwards)."""
    k, a, b, scale, fps, path, enc, preview = job
    video = _JOB
    video._configure(scale, fps)
    W, H = round(video.W * scale) // 2 * 2, round(video.H * scale) // 2 * 2
    surf = skia.Surface(W, H)
    ctx = video._ctx(scale)
    hits = video._all_hits()
    grain = None if preview else [np.random.default_rng(i).normal(0, 4.0, (H, W, 1)).astype(np.int16) for i in range(6)]
    proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                             "-r", str(fps), "-i", "-", *enc,
                             "-r", str(fps), "-video_track_timescale", "15360", path], stdin=subprocess.PIPE)
    for fi in range(a, b):
        t = fi / fps
        proc.stdin.write(video.draw(surf, ctx, t, hits, grain, fi)[:H, :W].tobytes())
        for p in video.scenes + video.overlays:
            if p.end < t - 0.5:
                for s in p.piece.sources():
                    s.close()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"encoding chunk {k} failed")
    for s in video.sources():
        s.close()
    return path


@lru_cache(maxsize=128)
def _aspect(path: str, crop) -> float:
    if crop:
        return crop[2] / crop[3]
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:stream_side_data=rotation", "-of", "json", path],
                       capture_output=True, text=True)
    st = json.loads(r.stdout or "{}").get("streams", [{}])[0]
    w, h = st.get("width", 16), st.get("height", 9)
    rot = abs(int(next((d.get("rotation", 0) for d in st.get("side_data_list", []) if "rotation" in d), 0)))
    return h / w if rot in (90, 270) else w / h


def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-1200:]}")
