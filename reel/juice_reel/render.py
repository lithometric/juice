"""Render a spec to a finished video.

Pipeline (one video encode total):
  1. build the timeline: every overlay layer and sound effect on the output clock (timeline.py)
  2. per shot, in parallel: cut + speed + reframe/follow + grade + zoom + shake + blur + finish,
     with that shot's slice of the overlay track streamed in as raw RGBA frames -> clips/NN.mp4
  3. concat the shot clips losslessly (-c copy)
  4. mix music, sound effects and/or original audio, normalize loudness, mux -> output

Every shot's length is rounded to whole frames first and the audio is cut to the
same length, so picture and sound cannot drift apart across cuts.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import geom, sfx
from . import text as textmod
from . import timeline as tlmod
from .spec import Shot, Spec


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:6])}...\n{r.stderr[-1500:]}")


def _even(x: float) -> int:
    return int(round(x / 2) * 2)


def _crop_x(shot: Shot, live: float) -> str:
    """Crop x expression: a fixed center, or an eased pan between two centers over the shot."""
    fx = shot.frame_x
    if isinstance(fx, (list, tuple)):
        a, b = fx
        s = f"min(t/{max(live, 0.01)},1)"
        cx = f"({a}+({b}-{a})*({s})*({s})*(3-2*{s}))"
    else:
        cx = str(fx)
    return f"clip({cx}*iw-ow/2,0,iw-ow)"


def _pw_linear(var: str, ts: list[float], vs: list[float]) -> str:
    """Piecewise-linear ffmpeg expression of `var` through the points (ts, vs)."""
    expr = f"{vs[-1]:.2f}"
    for i in range(len(ts) - 2, -1, -1):
        t0, t1, v0, v1 = ts[i], ts[i + 1], vs[i], vs[i + 1]
        seg = f"({v0:.2f}+({v1 - v0:.2f})*({var}-{t0:.4f})/{max(t1 - t0, 1e-4):.4f})"
        expr = f"if(lt({var},{t1:.4f}),{seg},{expr})"
    return expr


def _zoom_expr(zoom, n: int) -> str:
    if isinstance(zoom, (list, tuple)):
        z0, z1 = zoom
        return f"({z0}+({z1}-{z0})*(1-pow(1-min(on/{n},1),3)))"
    return f"{zoom}"


def _finish(style: dict, theme: dict, footage: bool) -> list[str]:
    """Whole-frame look. Vignette sits under the graphics; grain and sharpening go over everything."""
    fin = {**theme.get("finish", {}), **style.get("finish", {})}
    u = min(style["width"], style["height"]) / 1080
    if footage:
        return [f"vignette=angle={0.35 + 0.6 * fin['vignette']:.3f}"] if fin.get("vignette") else []
    out = []
    if fin.get("sharpen"):
        out.append(f"unsharp=5:5:{fin['sharpen']:.2f}:5:5:0")
    if fin.get("grain"):
        out.append(f"noise=alls={max(1, round(fin['grain'] * u))}:allf=t")
    return out


def _video_filters(shot: Shot, spec: Spec, fr: geom.Framing) -> list[str]:
    style = spec.style
    w, h, fps = style["width"], style["height"], style["fps"]
    live = fr.L
    n = max(1, round(live * fps))
    vf = [f"setpts=(PTS-STARTPTS)/{shot.speed}", f"fps={fps}",
          f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos"]
    zoom = geom.zoom_range(shot)
    if fr.ts is not None:
        vf.append(f"crop={w}:{h}:x='{_pw_linear('t', fr.ts, fr.x0s)}':y='{fr.y0:.2f}'")
    else:
        vf.append(f"crop={w}:{h}:x='{_crop_x(shot, live)}':y='(ih-oh)/2'")
    if style.get("grade"):
        vf.append(style["grade"])
    if zoom != 1.0:
        if fr.ts is not None:
            tvar = f"(on/{fps})"
            cx = _pw_linear(tvar, fr.ts, [2 * v for v in fr.txs])
            cy = _pw_linear(tvar, fr.ts, [2 * v for v in fr.tys])
        else:
            cx, cy = "iw/2", "ih/2"
        # Upscale first so zooms don't jitter on integer crop steps; push/pull eases out.
        vf += [f"scale={w * 2}:{h * 2}:flags=lanczos",
               f"zoompan=z='{_zoom_expr(zoom, n)}':d=1:x='max(0,min(iw-iw/zoom,{cx}-iw/zoom/2))'"
               f":y='max(0,min(ih-ih/zoom,{cy}-ih/zoom/2))':s={w}x{h}:fps={fps}"]
    if shot.shake:
        at, dur, amp = shot.shake.get("at", 0), shot.shake.get("dur", 0.5), shot.shake.get("amp", 0.025)
        env = f"gte(t,{at})*lt(t,{at + dur})*exp(-(t-{at})*{6 / dur})"
        vf += [f"scale={_even(w * (1 + 3 * amp))}:{_even(h * (1 + 3 * amp))}",
               f"crop={w}:{h}:x='(iw-ow)/2+{amp}*ow*sin(t*61)*{env}':y='(ih-oh)/2+{amp}*oh*cos(t*47)*{env}'"]
    if shot.blur:
        vf.append(f"gblur=sigma={shot.blur * min(w, h) / 1080:.1f}")
    vf += _finish(style, spec.theme, footage=True)
    # Hold the last frame for a freeze, plus a few spare frames so -frames:v always gets its
    # full count even when the source runs a frame short at the cut.
    vf.append(f"tpad=stop_mode=clone:stop_duration={shot.freeze_s + 0.1}")
    return vf


def _atempo(speed: float) -> str:
    """atempo only takes 0.5-2.0 per stage; chain stages for anything outside that."""
    parts, s = [], speed
    while s < 0.5:
        parts.append("atempo=0.5")
        s /= 0.5
    while s > 2.0:
        parts.append("atempo=2.0")
        s /= 2.0
    parts.append(f"atempo={s}")
    return ",".join(parts)


def render_shot(i: int, shot: Shot, spec: Spec, clips: Path, tl: tlmod.Timeline) -> tuple[Path, Path | None, float]:
    style = spec.style
    fps, W, H = style["fps"], style["width"], style["height"]
    n = spec.shot_frames(shot)
    dur = n / fps
    t0 = tl.starts[i]
    src = str(spec.sources[shot.source])
    src_len = shot.end - shot.start
    out = clips / f"{i:02d}.mp4"
    fr = geom.Framing(shot, spec)

    cmd = ["ffmpeg", "-y", "-ss", f"{shot.start}", "-t", f"{src_len}", "-i", src]
    graph = [f"[0:v]{','.join(_video_filters(shot, spec, fr))}[v0]"]
    cur, k = "[v0]", 1
    if shot.text:
        png = textmod.render_png(shot.text, style, clips / f"{i:02d}_text.png")
        # Read the still at the video's frame rate with headroom; -frames:v sets the real length.
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{dur + 0.5}", "-i", str(png)]
        txt = "format=rgba"
        if style.get("text_fade_s"):
            txt += f",fade=t=in:st=0:d={style['text_fade_s']}:alpha=1"
        graph.append(f"[{k}:v]{txt}[t{k}];{cur}[t{k}]overlay=0:0:eof_action=repeat[v{k}]")
        cur, k = f"[v{k}]", k + 1
    layers = [L for L in tl.layers if L.start < t0 + dur and L.end > t0]
    if layers:
        # This shot's slice of the overlay track, streamed as raw RGBA (no image files).
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(fps), "-i", "pipe:0"]
        graph.append(f"[{k}:v]format=rgba[o{k}];{cur}[o{k}]overlay=0:0:eof_action=repeat[v{k}]")
        cur, k = f"[v{k}]", k + 1
    post = ",".join(_finish(style, spec.theme, footage=False) + ["format=yuv420p"])
    graph.append(f"{cur}{post}[v]")
    cmd += ["-filter_complex", ";".join(graph), "-map", "[v]",
            "-an", "-frames:v", str(n), "-c:v", "libx264", "-preset", style.get("preset", "medium"),
            "-crf", str(style["crf"]), "-r", str(fps), "-video_track_timescale", "15360", str(out)]
    if layers:
        ctx = tlmod.Ctx(spec, spec.theme, tl)
        log = clips / f"{i:02d}.log"
        with open(log, "wb") as errf:  # a file, not a pipe: a full stderr pipe would deadlock the writer
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errf)
            try:
                for f in range(n):
                    proc.stdin.write(tlmod.draw_frame(ctx, layers, t0 + f / fps).tobytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
            code = proc.wait()
        if code != 0:
            raise RuntimeError(f"ffmpeg failed on shot {i}:\n{log.read_text(errors='replace')[-1500:]}")
        log.unlink()
    else:
        run(cmd)

    wav = None
    if spec.original_audio:
        wav = clips / f"{i:02d}.wav"
        e = style["edge_fade_s"]
        live = src_len / shot.speed
        tempo = f"{_atempo(shot.speed)}," if shot.speed != 1 else ""
        # Fade at the cut edges (no pops), pad silence under a freeze, cut to the frame-rounded length.
        af = (f"{tempo}afade=t=in:st=0:d={e},afade=t=out:st={max(live - e, 0)}:d={e},"
              f"apad,atrim=0:{dur},asetpts=PTS-STARTPTS,aresample=48000")
        run(["ffmpeg", "-y", "-ss", f"{shot.start}", "-t", f"{src_len}", "-i", src, "-vn",
             "-af", af, "-ac", "2", "-c:a", "pcm_s16le", str(wav)])
    return out, wav, dur


def render(spec: Spec, keep_clips: bool = False, workers: int = 4) -> dict:
    style = spec.style
    edit = spec.edit_dir
    edit.mkdir(parents=True, exist_ok=True)
    tag = f"_{spec.format}" if spec.format else ""
    clips = edit / f"clips{tag}"
    shutil.rmtree(clips, ignore_errors=True)  # only the current edit's shots, never stale ones
    clips.mkdir()

    tl = tlmod.build(spec)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(lambda a: render_shot(a[0], a[1], spec, clips, tl), enumerate(spec.shots)))
    total = sum(d for _, _, d in parts)
    starts = tl.starts

    vlist = clips / "video.txt"
    vlist.write_text("".join(f"file '{p}'\n" for p, _, _ in parts))
    silent = clips / "video.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(vlist), "-c", "copy", str(silent)])

    inputs = ["-i", str(silent)]
    chains, labels = [], []

    def add_input(*args: str) -> int:
        inputs.extend(args)
        return sum(1 for a in inputs if a == "-i") - 1

    if spec.original_audio:
        alist = clips / "audio.txt"
        alist.write_text("".join(f"file '{w}'\n" for _, w, _ in parts))
        orig = clips / "audio.wav"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(alist), "-c", "copy", str(orig)])
        labels.append(f"[{add_input('-i', str(orig))}:a]")
    if spec.music:
        idx = add_input("-ss", f"{spec.music.offset}", "-i", str(spec.music.file))
        e = style["edge_fade_s"]
        fade_out = style.get("music_fade_out_s") or e
        gain = style["music_gain_under_speech"] if spec.original_audio else 1.0
        chains.append(f"[{idx}:a]atrim=0:{total},asetpts=PTS-STARTPTS,aresample=48000,"
                      f"afade=t=in:st=0:d={e},afade=t=out:st={total - fade_out}:d={fade_out},"
                      f"volume={gain}[mus]")
        labels.append("[mus]")
    events = [(max(0.0, at - sfx.ATTACK[s]), s) for at, s in tl.sfx]
    for j, (at, sound) in enumerate(events):
        idx = add_input("-i", str(sfx.render(sound, clips / f"sfx_{sound}.wav")))
        ms = int(at * 1000)
        chains.append(f"[{idx}:a]aresample=48000,adelay={ms}:all=1,volume={style.get('sfx_gain', 0.8)}[s{j}]")
        labels.append(f"[s{j}]")
    if not labels:
        labels.append(f"[{add_input('-f', 'lavfi', '-t', f'{total}', '-i', 'anullsrc=r=48000:cl=stereo')}:a]")

    if len(labels) == 1:
        mix = labels[0]
    else:
        mix = f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=first,"
    ln = style["loudness"]
    chains.append(f"{mix}atrim=0:{total},loudnorm=I={ln['I']}:TP={ln['TP']}:LRA={ln['LRA']},"
                  f"aresample=48000,aformat=channel_layouts=stereo[a]")
    run(["ffmpeg", "-y", *inputs, "-filter_complex", ";".join(chains),
         "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-t", f"{total}", "-movflags", "+faststart", str(spec.output)])

    for f in (vlist, silent, clips / "audio.txt", clips / "audio.wav"):
        f.unlink(missing_ok=True)
    if not keep_clips:
        shutil.rmtree(clips, ignore_errors=True)

    ranges = [{"source": s.source, "start": s.start, "end": s.end, "speed": s.speed, "at": round(t0, 3),
               "duration": round(d, 3), "text": s.text.lines if s.text else None,
               "fx": [f["type"] for f in s.fx]}
              for s, t0, (_, _, d) in zip(spec.shots, starts, parts)]
    report = {"output": str(spec.output), "format": spec.format, "size": [style["width"], style["height"]],
              "duration_s": round(total, 3), "shots": ranges, "cuts_at": [r["at"] for r in ranges[1:]],
              "layers": [{"type": L.comp, "from": round(L.start, 2), "to": round(L.end, 2)} for L in tl.layers],
              "sfx": [{"at": round(a, 3), "sound": s} for a, s in events]}
    (edit / f"edl{tag}.json").write_text(json.dumps(report, indent=2))
    return report
