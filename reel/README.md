# juice-reel

Spec-driven video builder for any genre and length: short-form (TikTok, Reels, Shorts), long-form
stories, explainers, rage bait, fight edits. The whole edit lives in one JSON file, so an AI agent
(or a person) edits numbers instead of dragging clips, and every change re-renders in seconds.

A script or a human sets the **look** (a theme) and the **story beats** (shots + events); juice
handles the rest: framing, tracking, motion graphics, HUDs, captions, sound effects, music sync,
loudness, platform safe zones, and verification.

```
reel.json  ──►  juice-reel render  ──►  edit/final.mp4  ──►  juice-reel check
(shots, text, music)   (one video encode)        (1080×1920)       (lengths, loudness, cuts, text zones)
```

This is the working prototype of the Juice edit model: the `reel.json` timeline format is what the
Rust engine and its MCP tools will read and write.

## Install

Needs `ffmpeg`, `yt-dlp` (for music) and [uv](https://docs.astral.sh/uv/).

```bash
brew install ffmpeg yt-dlp uv
cd reel && uv sync
echo 'ELEVENLABS_API_KEY=...' > .env   # Speech to Text permission only; git-ignored
```

Run from your project folder with `uv run --project /path/to/juice/reel juice-reel <command>`.

## A reel

```json
{
  "sources": { "main": "IMG_1957.MOV" },
  "output": "edit/final.mp4",
  "shots": [
    { "source": "main", "start": 23.12, "end": 26.00,
      "text": { "lines": ["bro this girl in tech", "wants to talk to you"], "x": 0.50, "y": 0.70 } },
    { "source": "main", "start": 55.64, "end": 58.60,
      "text": { "lines": ["can we pull her", "SEO analytics?"], "x": 0.68, "y": 0.17 } },
    { "source": "main", "start": 63.80, "end": 66.10, "punch_in": 0.18 },
    { "source": "main", "start": 74.55, "end": 75.45 }
  ],
  "music": { "file": "music/Ramalama.mp3", "offset": 86.62 }
}
```

| Field | Meaning |
|---|---|
| `shots[].start/end` | Source range in seconds. Lengths are rounded to whole frames. |
| `shots[].text` | `lines` (hand-set breaks), `x`/`y` as fractions of the frame, `align` `center`/`right`/`left` |
| `shots[].punch_in` | Slow eased zoom over the shot (0.18 = 18% tighter at the end) |
| `shots[].freeze_s` | Hold the last frame |
| `shots[].speed` | `0.5` = slow motion (smooth from 60fps sources), `2` = speed ramp |
| `shots[].frame_x` | Crop center when the output is narrower than the source (16:9 → 9:16); `[from, to]` pans with easing |
| `shots[].shake` | `{"at", "dur", "amp"}` decaying camera shake |
| `shots[].fx` | Components on this shot (`at` relative to the shot); see **Components**. Any fx can carry `"sfx": "whoosh" \| "impact" \| "boom" \| "pop"` |
| `shots[].replay` | A hook / flash-forward: shown, but source events don't fire in it |
| `shots[].blur` | Background blur under title/end cards |
| `shots[].zoom` | `1.6` static close-up, `[1.0, 1.3]` push in, `[1.4, 1.0]` pull out (eased) |
| `shots[].follow` | Keep a tracked target centered through the crop and zoom (needs top-level `"tracks": "tracks.json"`); `follow_y` aims below the head in head-heights |
| `theme` | The look: `"bold"`, `"clean"`, `"loud"`, a file, or `{"base": "bold", "colors": {...}, "fonts": {...}}` |
| `overlays` | Components that span cuts, by output time (`from`/`to`) or shot (`from_shot`/`to_shot`): meters HUD, captions, tags, ticker, watermark, progress |
| `events` | Moments in the **source** (`"t": 27.08`) with `fx`, `sfx` and/or `meter` changes; juice fires them wherever that moment lands in the edit |
| `meters` | Named values for HUD bars: `{"hp": 100, "claude": {"value": 100, "drift": -0.3}}` |
| `transcripts` | `{"main": "transcripts/clip.json"}`, word-level, for the `captions` component |
| `formats` | Render one edit to several sizes, e.g. `{"vertical": {"style": {"width": 1080, "height": 1920}}, "horizontal": {"style": {"width": 1920, "height": 1080}}}`; shots can override fields per format with `per_format` |
| `music.offset` | Where in the song the reel starts |
| `original_audio` | Keep the sources' audio (music is ducked under it) |
| `style.platform` | `all` (default: clear of every app's UI), `tiktok`, `reels` or `shorts`; sets the text safe zones `check` enforces |
| `style` | Per-reel overrides of `juice_reel/style.default.json`; a `style.json` next to the spec applies to every reel in that folder |

## Themes and components

Themes (`juice_reel/themes/`): **bold** (condensed display, mono labels, slanted chips, hard shadows),
**clean** (Inter, soft shadows: stories, documentary), **loud** (heavy outlined type, yellow/red: rage
bait). Fonts are Google Fonts fetched once to `~/.cache/juice/fonts`. Preview a look before rendering:
`juice-reel gallery --theme loud --bg frame.jpg`.

Components (`juice_reel/components.py`), all themed and genre-neutral: `flash`, `slam` (giant word →
snaps to size), `versus` (+ typed premise), `title` (kicker / big / sub), `lower_third`, `chip`,
`caption`, `captions` (word-by-word from the transcript, active word highlighted), `meters` (two-sided
bars + counter, driven by events), `progress`, `callout` (circle + leader + label on a tracked thing),
`tags` (name + stat pinned over tracked heads, constant size through zooms), `ticker`, `watermark`,
`letterbox`, `endcard`. Slow-motion shots automatically get letterbox bars and a `SLOW-MO 0.5x` chip.

Overlays are drawn per frame and streamed into ffmpeg (no image files); shots render in parallel; the
theme's finish (vignette, sharpening, film grain against banding) is applied in the same single encode.

## Commands

```bash
juice-reel phrases IMG_1957.MOV                 # what's said, when (transcribes once, cached)
juice-reel words IMG_1957.MOV --from 54 --to 59 # word timings for exact cut points
juice-reel filmstrip IMG_1957.MOV 52 60 -n 12   # frames + waveform in one PNG
juice-reel frames source.mp4 26.8 27.1 27.4      # labeled frames at exact times (find the frame a hit lands)
juice-reel track source.mp4 targets.json --from 6 --to 41 -o tracks.json   # follow heads (keyframed CSRT) + debug sheet
juice-reel tag source.mp4 tracks.json -o tagged.mp4                        # name tags pinned above each head
juice-reel music search "Roisin Murphy Ramalama"
juice-reel music get <url> --name Ramalama
juice-reel music lyrics music/Ramalama.mp3
juice-reel music offset 7.46 2.88               # offset that lands a lyric 40ms after a cut
juice-reel gallery --theme bold --bg frame.jpg    # every component in a theme on one sheet
juice-reel render reel.json
juice-reel check reel.json                      # exits 1 if anything is off (lengths, loudness, cuts, text in platform UI)
```

`CLAUDE.md` holds the editing rules (cut timing, text, music) learned from real edits; agents should
follow it.

## Tests

```bash
uv run --extra dev pytest -q
```
