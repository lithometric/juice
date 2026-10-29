"""juice ingest: a messy pile of downloads in, indexed and usable footage out, in one parallel pass.

Stalled Chrome downloads of Google Drive zips, moov-less screen recordings and loose takes all end
up as playable clips with transcripts, a contact sheet, and a manifest.json that's rewritten after
every clip (with "done": false until the end) so an agent can start ingest in the background and
read what's ready while it works on something else.

Inputs are untrusted: nothing in them is executed, archives only extract into new empty folders
(bsdtar refuses absolute and ".." paths), symlinks are never followed, and nothing is written
outside the output folder.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import recover

MEDIA_EXT = {".mov", ".mp4", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts", ".ts", ".3gp", ".flv",
             ".wmv", ".mpg", ".mpeg", ".mxf", ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".aif",
             ".aiff", ".caf"}
ISO_BOXES = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot", b"uuid"}
SKIP_DIRS = {"__MACOSX", ".git", ".Trash"}
THUMBS = 4
THUMB_W, THUMB_H = 320, 180


# --- what's in the pile -----------------------------------------------------------------------

def sniff(path: Path) -> str | None:
    """'archive' / 'media' by magic bytes (Chrome names a half-downloaded zip *.crdownload), else
    'media' by extension for anything ffprobe should look at, else None."""
    try:
        with open(path, "rb") as f:
            head = f.read(512)
    except OSError:
        return None
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08") or head[:2] == b"\x1f\x8b" \
            or head[:3] == b"BZh" or head[:6] in (b"\xfd7zXZ\x00", b"7z\xbc\xaf\x27\x1c") \
            or head[:4] == b"\x28\xb5\x2f\xfd" or head[:4] == b"Rar!" or head[257:262] == b"ustar":
        return "archive"
    if head[4:8] in ISO_BOXES or head[:4] in (b"\x1a\x45\xdf\xa3", b"OggS", b"fLaC", b"FORM", b"caff") \
            or (head[:4] == b"RIFF" and head[8:12] in (b"AVI ", b"WAVE")) or head[:3] in (b"ID3", b"FLV") \
            or (len(head) > 188 and head[0] == 0x47 and head[188] == 0x47):
        return "media"
    return "media" if path.suffix.lower() in MEDIA_EXT else None


def walk(root: Path, skip: Path | None = None):
    """Regular files under root; no symlinks (an archive can plant one pointing anywhere), no dot
    files, no __MACOSX resource forks, nothing under skip (the output folder)."""
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        d = Path(dirpath)
        if skip and (d == skip or skip in d.parents):
            dirs[:] = []
            continue
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not x.startswith("."))
        for name in sorted(files):
            p = d / name
            if not name.startswith(".") and not p.is_symlink() and p.is_file():
                yield p


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    return s[:80] or "file"


def unique(dir_: Path, stem: str, ext: str = "", taken: set | None = None) -> Path:
    i = 1
    while True:
        p = dir_ / (f"{stem}{ext}" if i == 1 else f"{stem}-{i}{ext}")
        if not p.exists() and (taken is None or p not in taken):
            if taken is not None:
                taken.add(p)
            return p
        i += 1


# --- archives ---------------------------------------------------------------------------------

def extract(archive: Path, raw: Path) -> dict:
    """Stream the archive through bsdtar into a new empty folder. Streaming reads local headers in
    order, so a zip cut off before its central directory still gives every complete member, plus
    the member it was cut in. A marker beside the folder makes re-runs skip finished extractions."""
    st = archive.stat()
    name = slug(archive.name.removesuffix(".crdownload"))
    marker = raw / f".{name}.extract.json"
    if marker.exists():
        m = json.loads(marker.read_text())
        if m.get("source") == str(archive) and m.get("size") == st.st_size and Path(m["dir"]).is_dir():
            return m
    raw.mkdir(parents=True, exist_ok=True)
    dest = unique(raw, name)
    dest.mkdir()  # new and empty: extraction never merges into anything
    with open(archive, "rb") as f:
        p = subprocess.run(["bsdtar", "-xvf", "-", "-C", str(dest), "--no-same-owner"], stdin=f,
                           capture_output=True, text=True, errors="replace")
    # -v lists "x <member>", with ": <error>" appended to the member the archive broke in.
    members, errors, cut = [], [], None
    for ln in p.stderr.splitlines():
        if not ln.startswith("x "):
            errors.append(ln)
            continue
        name, err = _member(dest, ln[2:])
        members.append(name)
        if err:
            errors.append(f"{name}: {err}")
            cut = cut or name
    members = [m for m in members if m]
    # A download that stopped shows up as an error on the last member ("Truncated ZIP file data", or
    # "ZIP decompression failed" when it was deflated).
    truncated = p.returncode != 0 and bool(members) and (cut == members[-1] or any(
        re.search(r"truncat|unexpected end|premature|EOF", ln, re.I) for ln in errors))
    m = {"source": str(archive), "size": st.st_size, "dir": str(dest), "members": len(members),
         "truncated": truncated, "partial_member": str(dest / (cut or members[-1])) if truncated and members else None,
         "error": "; ".join(errors)[-400:] if p.returncode else None}
    marker.write_text(json.dumps(m, indent=2))
    return m


def _member(dest: Path, text: str) -> tuple[str, str | None]:
    """Split "path: error" where the path may itself contain ": " (the extracted file tells which)."""
    if (dest / text).exists():
        return text, None
    parts = text.split(": ")
    for i in range(1, len(parts)):
        if (dest / ": ".join(parts[:i])).exists():
            return ": ".join(parts[:i]), ": ".join(parts[i:])
    return parts[0], ": ".join(parts[1:]) or None


# --- probing ----------------------------------------------------------------------------------

def _rate(s: str | None) -> float | None:
    try:
        n, d = (s or "").split("/")
        return round(float(n) / float(d), 3) if float(d) else None
    except ValueError:
        return None


def probe(path: Path) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                        str(path)], capture_output=True, text=True, errors="replace", timeout=120)
    if p.returncode != 0 or not p.stdout.strip():
        return {"ok": False, "moov_missing": "moov atom not found" in p.stderr,
                "error": (p.stderr.strip().splitlines() or ["unreadable"])[-1][:200]}
    data = json.loads(p.stdout)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(data.get("format", {}).get("duration") or (v or a or {}).get("duration") or 0)
    if not v and not a:
        return {"ok": False, "moov_missing": False, "error": "no audio or video streams"}
    if v and not a and v.get("codec_name") in ("png", "mjpeg", "bmp", "gif", "webp", "tiff") and dur < 0.5:
        return {"ok": False, "moov_missing": False, "image": True, "error": "still image"}
    rot = 0
    for sd in (v or {}).get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = rot or int((v or {}).get("tags", {}).get("rotate", 0))
    return {"ok": True, "duration": round(dur, 3), "width": v and v.get("width"), "height": v and v.get("height"),
            "fps": v and _rate(v.get("avg_frame_rate")), "codec": (v or a).get("codec_name"),
            "has_audio": a is not None, "rotation": rot, "stream": v,
            "encoder": data.get("format", {}).get("tags", {}).get("encoder")}


# --- per-clip work ----------------------------------------------------------------------------

def thumbs(media: Path, duration: float, out_dir: Path, key: str) -> list[str]:
    out = []
    for i in range(THUMBS):
        t = max(0.0, duration * (i + 0.5) / THUMBS - 0.05)
        p = out_dir / f"{key}_{i}.jpg"
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(media), "-frames:v", "1",
                            "-vf", f"scale={THUMB_W}:{THUMB_H}:force_original_aspect_ratio=decrease", str(p)],
                           capture_output=True, timeout=120)
        if r.returncode == 0 and p.exists():
            out.append(str(p))
    return out


def loudness(wav: Path) -> float:
    """Share of 50 ms windows louder than -40 dBFS (0 = silent track)."""
    with wave.open(str(wav)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    n = len(a) // 800
    if n == 0:
        return 0.0
    rms = np.sqrt((a[:n * 800].reshape(n, 800) ** 2).mean(axis=1) + 1e-12)
    return float((20 * np.log10(rms) > -40).mean())


def backend() -> str | None:
    from . import transcribe
    try:
        transcribe.api_key()
        return "elevenlabs"
    except SystemExit:
        return "whisper" if shutil.which("uvx") else None


def first_words(transcript: Path, n: int = 12) -> str:
    from . import transcribe
    return " ".join([w["text"] for w in transcribe.words(transcript) if w.get("type") == "word"][:n])


# --- outputs ----------------------------------------------------------------------------------

class Manifest:
    """manifest.json, rewritten atomically after every change so it can be read while ingest runs."""

    def __init__(self, path: Path, info: dict):
        self.path, self.info, self.clips, self.lock = path, info, {}, threading.Lock()

    def put(self, key: str, **fields):
        with self.lock:
            self.clips.setdefault(key, {}).update(fields)
            self._write()

    def note(self, key: str, msg: str):
        with self.lock:
            self.clips[key].setdefault("notes", []).append(msg)
            self._write()

    def finish(self, **info):
        with self.lock:
            self.info.update(info, done=True)
            self._write()

    def _write(self):
        clips = sorted(self.clips.values(), key=lambda c: c.get("original", ""))
        public = [{k: v for k, v in c.items() if not k.startswith("_")} for c in clips]
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({**self.info, "clips": public}, indent=2))
        os.replace(tmp, self.path)


def _mb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} KB"


def contact_sheet(clips: list[dict], out: Path) -> Path | None:
    """Every usable video clip as a row of thumbnails under its name, duration and size."""
    from .theme import font
    rows = [c for c in clips if c.get("_thumbs")]
    if not rows:
        return None
    pad, gap, label = 24, 8, 52
    row_h = label + THUMB_H + pad
    W = pad * 2 + THUMBS * THUMB_W + (THUMBS - 1) * gap
    sheet = Image.new("RGB", (W, pad + row_h * len(rows)), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    f_name, f_meta = font("Inter", 22, 700), font("Inter", 16, 500)
    colors = {"ok": (120, 220, 140), "recovered": (110, 180, 255), "partial": (255, 200, 90)}
    for i, c in enumerate(rows):
        y = pad + i * row_h
        d.text((pad, y), c["name"].replace("\u202f", " "), font=f_name, fill=(240, 240, 240))  # macOS "6.45.19\u202fPM"
        bits = [f"{c.get('duration') or 0:.1f}s", f"{c.get('width')}x{c.get('height')}",
                f"{c.get('fps') or 0:g} fps", _mb(c.get("size", 0)), c["status"]]
        if c.get("speech"):
            bits.append("speech")
        d.text((pad, y + 28), "  ·  ".join(bits), font=f_meta, fill=colors.get(c["status"], (200, 200, 200)))
        for k, t in enumerate(c["_thumbs"]):
            im = Image.open(t).convert("RGB")
            x = pad + k * (THUMB_W + gap)
            sheet.paste(im, (x + (THUMB_W - im.width) // 2, y + label + (THUMB_H - im.height) // 2))
    sheet.save(out, quality=85)
    return out


def report(out: Path, info: dict, clips: list[dict], archives: list[dict], skipped: list[str], wall: float) -> str:
    by = {s: [c for c in clips if c["status"] == s] for s in ("ok", "recovered", "partial", "unrecoverable")}
    talk = [c for c in clips if c.get("speech")]
    L = [f"# Ingest ({datetime.now().strftime('%Y-%m-%d %H:%M')})", "",
         f"{len(clips)} clips from {len(info['inputs'])} inputs in {wall:.0f}s: {len(by['ok'])} ok, "
         f"{len(by['recovered'])} recovered, {len(by['partial'])} partial, {len(by['unrecoverable'])} unrecoverable. "
         f"{len(talk)} with speech, transcribed with {info.get('transcriber') or 'nothing (--no-transcribe)'}.", ""]
    if archives:
        L += ["## Archives", ""]
        for a in archives:
            state = "**truncated**" if a.get("truncated") else ("error" if a.get("error") else "complete")
            L.append(f"- `{Path(a['source']).name}` -> `{os.path.relpath(a['dir'], out)}/`: {a['members']} "
                     f"files, {state}" + (f" (cut inside `{Path(a['partial_member']).name}`)"
                                          if a.get("partial_member") else "")
                     + (f"; {a['error']}" if a.get("error") and not a.get("truncated") else ""))
        L.append("")
    if by["recovered"] or by["partial"]:
        L += ["## Repaired", ""]
        for c in by["recovered"] + by["partial"]:
            L.append(f"- `{c['name']}` ({c['status']}) -> `{os.path.relpath(c['path'], out)}`: "
                     f"{c.get('width')}x{c.get('height')} {c.get('fps')} fps {c.get('duration') or 0:.1f}s. "
                     + "; ".join(c.get("notes", [])))
        L.append("")
    if by["unrecoverable"]:
        L += ["## Failed", ""]
        L += [f"- `{c['name']}`: {'; '.join(c.get('notes', []))}" for c in by["unrecoverable"]]
        L.append("")
    L += ["## Ready", "", "| clip | status | length | size | speech |", "|---|---|---|---|---|"]
    for c in clips:
        if c["status"] != "unrecoverable":
            said = f"\"{c['first_words']}…\"" if c.get("first_words") else ("yes" if c.get("speech") else "")
            L.append(f"| `{os.path.relpath(c['path'], out) if Path(c['path']).is_relative_to(out) else c['path']}` "
                     f"| {c['status']} | {c.get('duration') or 0:.1f}s | {c.get('width')}x{c.get('height')} | {said} |")
    if skipped:
        L += ["", f"Skipped {len(skipped)} non-media or duplicate files: "
              + ", ".join(f"`{s}`" for s in skipped[:20]) + (" …" if len(skipped) > 20 else "")]
    L += ["", "Contact sheet: `contact.jpg`. Everything per clip: `manifest.json`."]
    return "\n".join(L) + "\n"


# --- the run ----------------------------------------------------------------------------------

def ingest(inputs: list[Path], out: Path, workers: int = 8, transcribe: bool = True, budget: float = 60,
           log=print) -> dict:
    t0 = time.time()
    out = out.resolve()
    work = out / ".work"
    for d in (out / "raw", out / "recovered", out / "transcripts", work / "thumbs", work / "audio"):
        d.mkdir(parents=True, exist_ok=True)
    tb = backend() if transcribe else None
    if transcribe and not tb:
        log("no ELEVENLABS_API_KEY and no uvx for mlx-whisper: skipping transcripts")
    man = Manifest(out / "manifest.json", {"done": False, "started": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "out": str(out), "inputs": [str(p) for p in inputs], "transcriber": tb})
    pool = cf.ThreadPoolExecutor(workers)
    tpool = cf.ThreadPoolExecutor(2 if tb == "whisper" else 4)
    pending: dict[cf.Future, tuple[str, str | None]] = {}  # future -> (stage, clip key)
    archives, skipped, broken, seen = [], [], [], {}
    keys, stems, taken = set(), set(), set()
    partial_members: dict[str, str] = {}

    def go(stage: str, key: str | None, fn, *a, p=pool):
        pending[p.submit(fn, *a)] = (stage, key)

    def discovering() -> bool:  # donors are only known once every archive is out and every file probed
        return any(stage in ("extract", "probe") for stage, _ in pending.values())

    def new_clip(path: Path, origin: str | None) -> str:
        key = slug(path.stem)
        while key in keys:
            key = slug(f"{key}-{len(keys)}")
        keys.add(key)
        man.put(key, name=path.name, path=str(path), original=str(path), archive=origin, status="pending",
                notes=[], size=path.stat().st_size)
        return key

    def consider(path: Path, origin: str | None = None):
        kind = sniff(path)
        if kind == "archive" and origin is None:
            go("extract", None, lambda: ("extracted", extract(path, out / "raw"), path))
        elif kind == "media":
            dup = (path.name, path.stat().st_size)
            if dup in seen:
                skipped.append(f"{path.name} (same as {seen[dup]})")
                return
            seen[dup] = str(path)
            key = new_clip(path, origin)
            go("probe", key, lambda: ("probed", key, path, probe(path)))
        else:
            skipped.append(path.name)

    def finish_clip(key: str, media: Path, info: dict):
        """Thumbnails + speech check (+ transcript job) for a usable clip; runs on the pool."""
        th = thumbs(media, info["duration"], work / "thumbs", key) if info.get("width") else []
        speech, wav = False, None
        if info.get("has_audio"):
            wav = work / "audio" / f"{key}.wav"
            try:
                from .whisper import wav16k
                wav16k(media, wav)
                speech = loudness(wav) > 0.03
            except (subprocess.CalledProcessError, wave.Error, EOFError):
                man.note(key, "audio track unreadable")
        man.put(key, _thumbs=th, speech=speech)
        if not (speech and tb) and wav:
            wav.unlink(missing_ok=True)
        return ("finished", key, media, wav if speech and tb else None)

    def _transcribe(key: str, media: Path, wav: Path | None, tdir: Path):
        try:
            if tb == "elevenlabs":
                from . import transcribe as scribe
                p = scribe.transcribe(media, tdir)
            else:
                from . import whisper
                p = whisper.transcribe(media, tdir, wav=wav)
            return p, None
        except Exception as e:  # a failed transcript never fails the clip
            return None, f"transcription failed: {e}"
        finally:
            if wav:
                wav.unlink(missing_ok=True)

    def do_repair(key: str, src: Path, donors: list[dict]):
        dest = unique(out / "recovered", slug(src.stem), ".mp4", taken)
        cache = work / f"repair-{key}.json"
        if cache.exists():
            res = json.loads(cache.read_text())
            if res.get("source") == str(src) and res.get("ok") and Path(res["out"]).exists():
                return ("repaired", key, src, Path(res["out"]), res)
        res = recover.repair(src, dest, donors, budget=budget, workers=max(2, workers))
        res.update(source=str(src), out=str(dest))
        cache.write_text(json.dumps(res))
        return ("repaired", key, src, dest, res)

    def apply_probe(key: str, media: Path, info: dict, status: str):
        meta = {k: info.get(k) for k in ("duration", "width", "height", "fps", "codec", "has_audio", "rotation")}
        partial = partial_members.get(str(media)) or partial_members.get(man.clips[key]["original"])
        if partial and status == "ok":
            status = "partial"
        if partial:
            man.note(key, f"the {partial} download stopped inside this file: the recording ran longer")
        man.put(key, status=status, path=str(media), **meta, _stream=info.get("stream"))
        go("finish", key, finish_clip, key, media, info)

    # Direct inputs first: their probes and transcripts start while the archives are still extracting.
    for p in inputs:
        p = p.expanduser()
        if p.is_dir():
            for f in walk(p, skip=out):
                consider(f)
        elif p.is_file():
            consider(p)
        else:
            log(f"not found: {p}")

    repairs_started = False
    while pending or not repairs_started:
        if not repairs_started and not discovering():
            repairs_started = True
            donors = [{"path": c["path"], "stream": c["_stream"]} for c in man.clips.values()
                      if c.get("status") == "ok" and c.get("codec") == "h264" and c.get("_stream")]
            for key, src in broken:
                go("repair", key, do_repair, key, src, donors)
            continue
        done, _ = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
        for fut in done:
            stage, key = pending.pop(fut)
            try:
                r = fut.result()
            except Exception as e:  # keep going; the clip just records why
                log(f"{stage} failed: {type(e).__name__}: {e}")
                if key and stage in ("probe", "repair"):
                    man.note(key, f"{stage} failed: {type(e).__name__}: {e}")
                    man.put(key, status="unrecoverable")
                continue
            kind = r[0]
            if kind == "extracted":
                a, src = r[1], r[2]
                archives.append(a)
                log(f"extracted {src.name}: {a['members']} files" + (" (truncated)" if a["truncated"] else ""))
                if a.get("partial_member"):
                    partial_members[a["partial_member"]] = src.name
                for f in walk(Path(a["dir"])):
                    consider(f, origin=src.name)
            elif kind == "probed":
                key, media, info = r[1], r[2], r[3]
                if info["ok"]:
                    apply_probe(key, media, info, "ok")
                elif info.get("moov_missing"):
                    man.put(key, status="repairing")
                    broken.append((key, media))
                else:
                    if info.get("image"):
                        with man.lock:
                            man.clips.pop(key)
                        skipped.append(media.name)
                        continue
                    man.note(key, f"not readable: {info['error']}")
                    man.put(key, status="unrecoverable")
            elif kind == "repaired":
                key, src, dest, res = r[1:]
                if not res.get("ok"):
                    man.note(key, f"moov missing; {res.get('reason')}")
                    man.put(key, status="unrecoverable")
                    log(f"unrecoverable {src.name}: {res.get('reason')}")
                    continue
                info = probe(dest)
                if not info["ok"]:
                    man.note(key, f"rebuilt stream doesn't play: {info['error']}")
                    man.put(key, status="unrecoverable")
                    continue
                for n in res.get("notes", []):
                    man.note(key, n)
                log(f"recovered {src.name}: {res['width']}x{res['height']} @ {res['fps']:g} fps, {res['frames']} frames")
                apply_probe(key, dest, info, "partial" if res.get("stopped_early") else "recovered")
            elif kind == "finished" and r[3]:
                key, media, wav = r[1:]
                tdir = out / "transcripts"
                if media.stem in stems:  # transcripts are cached by stem; keep two "IMG_0001"s apart
                    tdir = tdir / key
                stems.add(media.stem)
                go("transcribe", key, lambda key=key, media=media, wav=wav, tdir=tdir:
                   ("transcribed", key, *_transcribe(key, media, wav, tdir)), p=tpool)
            elif kind == "transcribed":
                key, path, err = r[1:]
                if err:
                    man.note(key, err)
                else:
                    fw = first_words(path)
                    man.put(key, transcript=str(path), first_words=fw, speech=bool(fw))

    pool.shutdown()
    tpool.shutdown()
    clips = sorted(man.clips.values(), key=lambda c: c.get("original", ""))
    sheet = contact_sheet([c for c in clips if c["status"] != "unrecoverable"], out / "contact.jpg")
    wall = time.time() - t0
    (out / "INGEST.md").write_text(report(out, man.info, clips, archives, skipped, wall))
    man.finish(finished=datetime.now(timezone.utc).isoformat(timespec="seconds"), wall_s=round(wall, 1),
               archives=archives, skipped=skipped, contact_sheet=str(sheet) if sheet else None)
    return {"manifest": man.path, "clips": clips, "archives": archives, "wall_s": wall}


def summary(res: dict) -> str:
    clips = res["clips"]
    lines = [f"ingested {len(clips)} clips in {res['wall_s']:.0f}s -> {res['manifest'].parent}"]
    for c in clips:
        extra = c.get("first_words") and f'  "{c["first_words"]}…"' or ""
        why = "  " + "; ".join(c.get("notes", [])) if c["status"] == "unrecoverable" else ""
        lines.append(f"  {c['status']:<13} {c['name']:<48} {c.get('duration') or 0:7.1f}s "
                     f"{c.get('width') or '-'}x{c.get('height') or '-'}{extra}{why}")
    return "\n".join(lines)


def cmd_ingest(a):
    res = ingest([Path(p) for p in a.inputs], Path(a.out), workers=a.workers, transcribe=not a.no_transcribe)
    print(summary(res))


def register(sub):
    p = sub.add_parser("ingest", help="downloads/archives/folders -> repaired, transcribed, indexed footage")
    p.add_argument("inputs", nargs="+", help="files or folders (zips, stalled .crdownload files, videos)")
    p.add_argument("-o", "--out", default="footage")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    p.add_argument("--no-transcribe", action="store_true")
    p.set_defaults(fn=cmd_ingest)
