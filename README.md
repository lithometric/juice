# juice

An AI-driven video toolkit. Two kinds of video, **portrait** (1080×1920) and **landscape** (1920×1080), each
**short** form (stops a scroll) or **long** form (holds attention), all on-brand from a `juice.json` brand kit.

- **Short-form cuts** (talking heads, reactions, gameplay, stories): the whole edit is one JSON file, so an
  agent edits numbers instead of dragging clips. juice handles framing, tracking, captions, HUDs, sound
  effects, music sync, loudness, safe zones and verification.
- **Motion videos** (trailers, launches, explainers): a short Python script places brand-driven scenes
  (logo reveal, product panels, kinetic type, feature grids, end cards) and voice-over on one clock, over an
  original music bed with hits on the cuts.
- **Ingest**: a pile of downloads becomes usable, transcribed, indexed footage.
- **An agent harness**: any model (through OpenRouter) or Claude (through MCP) edits with these tools.

```
juice ingest ~/Downloads/*  ──►  footage/ (unpacked, repaired, transcribed, indexed)
reel.json  ──►  juice render [--preview]  ──►  edit/final.mp4  ──►  juice check
trailer.py ──►  juice motion [--preview | --stills]  ──►  out/trailer_landscape.mp4
juice-agent "cut a 30s portrait reel, bold look"   ──►  a model does all of the above
```

## Layout

```
juice/              the engine (Python package, `juice` command)
  shortform/        cuts: spec.py (reel.json), render.py, check.py, timeline, components, themes, hooks, retention, tracking
  motion/           motion videos: video.py (the Video API), scenes.py (loud look), clean.py (clean look), mix.py
  media/            ingest, recover, transcribe, whisper, isolate (voice), inspect (filmstrips), music, beds
  brand.py          juice.json brand kits        cli.py  the command line
harness/            the agent: tools.py (every command as a tool), agent.py (the loop), openrouter.py, prompt.py,
                    cli.py (`juice-agent`), mcp_server.py (`juice-mcp`)
examples/           openbeam-trailer/ (a motion video, both looks), seo-analytics/ (a cut)
tests/
CLAUDE.md           the editing guide: how to cut, hook, caption, score and mix. Read by people, Claude Code and the harness.
```

The `reel.json` format and the motion script are the edit model the Rust engine and its MCP tools will use.
Brand kits come from [`juice-press`](../juice-press) (`npx juice-press yourbrand.com ./your-repo`).

## Install

Needs `ffmpeg`, `yt-dlp` (for music) and [uv](https://docs.astral.sh/uv/) (its `uvx` also runs the local
whisper and voice-isolation models, downloaded once on first use).

```bash
brew install ffmpeg yt-dlp uv
uv sync --extra dev --extra mcp
echo 'ELEVENLABS_API_KEY=...' >> .env    # optional: Speech to Text; without it mlx-whisper runs locally
echo 'OPENROUTER_API_KEY=...' >> .env    # for juice-agent
```

Run from your project folder with `uv run --project /path/to/juice juice <command>` (or `juice-agent`, `juice-mcp`).

## The agent

```bash
juice-agent "cut a 30s portrait reel from footage/, bold look, hook on the funniest line"
juice-agent -p ~/edits/launch -m google/gemini-3.5-flash "a 25s landscape launch trailer in the clean look"
juice-agent --resume "tighter: lose the second reaction"       # same session, same context
juice-agent --once --max-cost 2 "..."                          # scripted: answer and exit, stop past $2
juice-agent --tools                                            # the tool list
```

The harness is a loop over the tool table in `harness/tools.py`: the model gets the brief, calls tools
(`footage`, `phrases`, `words`, `filmstrip`, `frames`, `hooks`, `write_file`, `render`, `check`, `motion`, ...),
and every picture a tool makes (filmstrips, contact sheets, the verifier's cut sheet) is shown to it in the
next message, since it cannot watch video. The verifier's deterministic feedback (`check` must come back
with no problems) is what lets any model with tool calling and image input finish a cut. The system prompt
is `CLAUDE.md` plus the spec reference below, so the guide is one file for people and every agent.

The model endpoint is OpenRouter's chat API, so the model is a string: `anthropic/claude-opus-5.5`
(default), `anthropic/claude-sonnet-5.5`, `google/gemini-3.5-flash`, `openai/gpt-6.1-sol`,
`qwen/qwen3.8-max-0902`, ... Reasoning blocks come back with each message and are passed back, so a
model's thinking stays attached to the tool calls it made. Sessions are logged to `<project>/agent/`.

**With Claude directly:** `juice-mcp --project <folder>` serves the same tools over MCP, so Claude Code, Claude
Desktop or claude.ai edit with juice with no loop of ours in between:

```bash
claude mcp add juice -- uv run --project /path/to/juice juice-mcp --project ~/edits/launch
```

Claude Code also works with the repo checked out and no server at all: `CLAUDE.md` tells it how to use the
`juice` command.

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
| `shots[].fx` | Components on this shot (`at` relative to the shot); see **Themes and components**. Any fx can carry `"sfx": "whoosh" \| "impact" \| "boom" \| "pop"` |
| `shots[].replay` | A hook / flash-forward: shown, but source events don't fire in it |
| `shots[].blur` | Background blur under title/end cards |
| `shots[].zoom` | `1.6` static close-up, `[1.0, 1.3]` push in, `[1.4, 1.0]` pull out (eased) |
| `shots[].follow` | Keep a tracked target centered through the crop and zoom (needs top-level `"tracks": "tracks.json"`); `follow_y` aims below the head in head-heights |
| `theme` | The look: `"bold"`, `"clean"`, `"loud"`, a file, or `{"base": "bold", "colors": {...}, "fonts": {...}}` |
| `overlays` | Components that span cuts, by output time (`from`/`to`) or shot (`from_shot`/`to_shot`): meters HUD, captions, tags, ticker, watermark, progress |
| `events` | Moments in the **source** (`"t": 27.08`) with `fx`, `sfx` and/or `meter` changes; juice fires them wherever that moment lands in the edit |
| `meters` | Named values for HUD bars: `{"hp": 100, "claude": {"value": 100, "drift": -0.3}}` |
| `transcripts` | `{"main": "transcripts/clip.json"}`, word-level, for the `captions` component |
| `orientation` | `portrait` (default, 1080×1920) or `landscape` (1920×1080) |
| `length` | `short` (default: triple hook in 1.5 s) or `long` (longer hook window); see `lengths` in `style.default.json` |
| `brand` | A `juice.json` brand kit (path, URL or domain); `juice.json` next to the spec is picked up automatically. Its colors and fonts become the theme |
| `formats` | Render one edit in both orientations: `{"portrait": {}, "landscape": {"style": {...}}}`; shots can override fields per orientation with `per_format` |
| `music.offset` | Where in the song the reel starts |
| `original_audio` | Keep the sources' audio (music is ducked under it) |

## Themes and components

Themes (`juice/shortform/themes/`): **bold** (condensed display, heavy sans labels, slanted chips, hard
shadows), **clean** (Inter, soft shadows: stories, documentary), **loud** (heavy outlined type, yellow/red:
rage bait). Fonts are Google Fonts fetched once to `~/.cache/juice/fonts`. Monospace faces are never used:
any family that looks like one (Mono, Code, Courier, Menlo, Consolas…) renders in Inter. Preview a look
before rendering: `juice gallery --theme loud --bg frame.jpg`.

Components (`juice/shortform/components.py`), all themed and genre-neutral: `flash`, `slam` (giant word →
snaps to size), `versus` (+ typed premise), `title` (kicker / big / sub), `lower_third`, `chip`,
`caption`, `captions` (word-by-word from the transcript: plain white, thin black stroke; highlights are opt-in), `meters` (two-sided
bars + counter, driven by events), `progress`, `callout` (circle + leader + label on a tracked thing),
`tags` (name + stat pinned over tracked heads, constant size through zooms), `ticker`, `watermark`,
`letterbox`, `endcard`. Slow-motion shots automatically get letterbox bars and a `SLOW-MO 0.5x` chip.

Overlays are drawn per frame and streamed into ffmpeg (no image files); shots render in parallel; the
theme's finish (vignette, sharpening, film grain against banding) is applied in the same single encode.

## Commands

```bash
juice phrases IMG_1957.MOV                 # what's said, when (transcribes once, cached)
juice words IMG_1957.MOV --from 54 --to 59 # word timings for exact cut points
juice filmstrip IMG_1957.MOV 52 60 -n 12   # frames + waveform in one PNG
juice frames source.mp4 26.8 27.1 27.4      # labeled frames at exact times (find the frame a hit lands)
juice track source.mp4 targets.json --from 6 --to 41 -o tracks.json   # follow heads (keyframed CSRT) + debug sheet
juice tag source.mp4 tracks.json -o tagged.mp4                        # name tags pinned above each head
juice music search "Roisin Murphy Ramalama"
juice music get <url> --name Ramalama
juice music lyrics music/Ramalama.mp3
juice music offset 7.46 2.88               # offset that lands a lyric 40ms after a cut
juice hooks IMG_*.MOV -n 10 -o hooks.jpg     # rank hook candidates across clips + contact sheet
juice gallery --theme bold --bg frame.jpg    # every component in a theme on one sheet
juice retention reel.json                  # predicted retention score, 3s/end retention, the moments that lose
                                                # viewers and why, + edit/verify/retention.png (no render needed)
juice ingest ~/Downloads/RESEARCH "~/Downloads/Unconfirmed 1234.crdownload" -o footage
                                           # unpack (even cut-off Drive zips), repair moov-less recordings,
                                           # transcribe, contact sheet + manifest.json, in parallel
juice render reel.json --preview           # 480p, half frame rate: a check in seconds
juice motion trailer.py --stills           # a frame from every scene on one sheet
juice motion trailer.py --preview          # the whole motion video at 480p in seconds
juice motion trailer.py --orientation portrait
juice beds                                 # the music bed styles, with a 30 s preview of each
juice render reel.json
juice check reel.json                      # exits 1 if anything is off: lengths, loudness, cuts, text in platform UI,
                                                # reused or unused footage, and the triple hook (spoken + text + visual)
```

`CLAUDE.md` holds the editing rules (cut timing, text, music) learned from real edits; agents follow it.

## Motion videos

```python
from juice.motion import Video, scenes as s

v = Video(length="short", brand="juice.json")          # orientation from --orientation (default landscape)
take = "takes/a.mov"
v.voice(take, 0.0, 1.6, at=0.0)                        # the speaker's line, placed on the output clock
line = v.voice(take, 1.7, 5.9, at=6.0)                 # keeps talking over other scenes (L-cut)
v.scene(s.Footage(v.source(take, 0, 1.6)), 0, 1.6, energy=0)
v.overlay(s.Headline("Now in beta.", highlight={"beta.": "accent"}), 0.3, 1.6)
v.scene(s.LogoReveal(badge="beta", tagline="What it is, in five words"), 1.6, 3.6, transition="flash")
v.scene(s.Panels(["screens/app.png", v.source("rec.mp4", 2, 4)],
                 callouts=[(1, 0.2, 0.5, "The thing it does")]), 3.6, 6.0, transition="logo")
v.overlay(s.Slam(["WORLD", "MODELS"]), line.out(3.3), 8.0)   # lands on the spoken word
v.scene(s.EndCard(line="Sign up at"), 8.0, 11.0, transition="flash", energy=0)
v.music("hype")                                        # hype | tech | calm | keynote | future, arranged to the energies
```

Two looks. **Loud** (`juice.motion.scenes`, hype edits): scenes `Footage`, `Stage`, `LogoReveal`, `Panels`,
`Linked`, `FeatureGrid` (+ `Card`), `Notification`, `EndCard`; overlays `Slam`, `Headline`, `Words`, `Chip`,
`Callout`, `Tracker`, `Letterbox`, `Label`, `Bug`, `UrlBar`; transitions `cut`, `flash`, `iris`, `logo`
(opens through the brand mark), `whip`, `glitch`, each with its own hit sound; `v.hit(t, "boom" | "slam" |
"land" | "riser" | "type" | ...)` adds punctuation. **Clean** (`juice.motion.clean`, launches and
enterprise): scenes `Frame` (product in a flat window beside its copy), `Split` (two windows on a labeled
link), `Bento` (feature grid, a cell per beat), `Lockup` (wordmark + link); overlays `Title` (masked line
reveals), `Terms` (terms that replace each other as they're spoken), `Caption`, `Notice`; soft transitions
`fade`, `dissolve`, `wipe`; quiet hits `sub`, `tick`, `air`, `swell`; and the `keynote` and `future` beds.

The voice is isolated before it's mixed (DeepFilterNet3, run locally through `uvx` and cached: room noise,
chatter and traffic removed, the words untouched; `isolate=False` per line keeps the room). The music is
set by loudness (BS.1770): as loud as the voice between lines, `duck` dB under them, leveled so no section
jumps out; `v.music(style, level=-3)` sits it 3 dB under the voice. The render reports `voice_lufs` and
`music_lufs`.

Everything is laid out for both orientations. A full render splits the frames across
CPU cores (the 27 s OpenBeam trailer: 24 s at 1080p, 6 s as a preview); see
`examples/openbeam-trailer/trailer.py` (clean) and `trailer_hype.py` (loud).

## Brand kits

`juice.json` (the Juice Brand Kit Protocol, see `juice-press/PROTOCOL.md`) holds a brand's name, line,
link, colors, fonts and logo files. Colors map to theme roles (`primary` → accent, `secondary` →
accent2, `background` → ink, `text` → paper), fonts to display/body. Monospace fonts are never used.

## Retention score

`juice retention` is a survival model of the edit: R(t) = exp(-∫h), with a front-loaded Weibull
baseline (calibrated to published short-form benchmarks: ~70% at 3s, ~45% at 30s for a neutral edit)
and per-moment hazard multipliers for the hook (text by 0.5s, a question, speech by 0.4s, motion),
word-level captions (Facebook +12% view time), visual change (Kim et al. 2014: 61% of interaction
peaks sit on visual transitions), dead air, fillers, restarts, and speaking rate (Guo, Kim & Rubin
2014). The score is the predicted average percentage viewed (Wu, Rizoiu & Xie 2018). It reports the
stretches that lose the most viewers with the reason, so the fix is concrete: cut the pause, add a
push-in, caption the speech. See `juice/shortform/retention.py` for the sources and the constants.

## Tests

```bash
uv run pytest -q
```
