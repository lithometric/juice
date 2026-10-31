"""The one video encode juice uses, for cuts and motion videos alike.

Masters are 10-bit: HEVC Main 10 (libx265, yuv420p10le, bt709, hvc1 so Apple players open them), so
gradients, grain and dark footage don't band the way 8-bit H.264 does, and platforms transcode from
a clean source. Previews (480p, "look at it in seconds") stay 8-bit H.264 ultrafast; they're for checking
a cut, not for posting.

    enc = video_args(style)      # "-c:v ... -pix_fmt ..." for an encode
    cp = copy_args(style)        # "-c:v copy" (+ the hvc1 tag) when remuxing what video_args produced
    pix = pix_fmt(style)         # the pixel format to end a filter graph on

style keys: bit_depth (10 | 8), crf, preset.
"""
from __future__ import annotations


def bits(style: dict) -> int:
    return int(style.get("bit_depth", 10))


def pix_fmt(style: dict) -> str:
    return "yuv420p10le" if bits(style) == 10 else "yuv420p"


def video_args(style: dict) -> list[str]:
    crf, preset = str(style.get("crf", 16)), style.get("preset", "medium")
    if bits(style) == 10:
        return ["-c:v", "libx265", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p10le",
                "-tag:v", "hvc1", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-color_range", "tv", "-x265-params", "log-level=error"]
    return ["-c:v", "libx264", "-preset", preset, "-crf", crf, "-pix_fmt", "yuv420p"]


def copy_args(style: dict) -> list[str]:
    return ["-c:v", "copy"] + (["-tag:v", "hvc1"] if bits(style) == 10 else [])
