# juice-reel: how to edit with it

For any short-form vertical video: TikTok, Instagram Reels, YouTube Shorts, Snapchat Spotlight, and so on.
Pick the target with `"style": {"platform": "tiktok" | "reels" | "shorts" | "all"}`; `all` (the default)
keeps text clear of every app's UI so one export can be posted everywhere.

The edit is `reel.json`. Every change the creator asks for is an edit to that file followed by
`juice-reel render reel.json && juice-reel check reel.json`. Never hand-run ffmpeg for a cut.

## What the human/script gives you vs. what you handle

- **They give:** the footage, the look ("bold game", "calm doc", "loud rage bait", brand colors/fonts,
  or a reference video), and the story (script, beats, or just "make it funny / make it hype").
- **You handle everything else:** pick or extend a theme, find the beats (transcript + frames), track
  subjects, cut, frame, choose components, place events, music + sync, sfx, captions, safe zones, render,
  check, and iterate on their notes.

## Look → theme

1. Map the brief onto the nearest built-in theme (`bold`, `clean`, `loud`) and override only what differs:
   `"theme": {"base": "bold", "colors": {"accent": [217,119,87]}, "fonts": {"display": {"family": "Bebas Neue"}}}`.
   Brand colors become named roles (e.g. `"gpt": [16,163,127]`) that components reference by name.
2. From a reference video: frames → read its fonts (condensed? mono labels?), palette, label shape (slanted
   chips?), shadow style, and its "story layer" (premise line, HUD, callouts, labeled slow-mo, end card).
3. `juice-reel gallery --theme <it> --bg <a real frame>` and look before rendering anything.

## Story layer (what makes an edit feel produced, any genre)

- **Stakes in one line** at the top (`versus.premise`, a `title.sub`, or a `chip`): "FIRST ONE ON THE GROUND
  LOSES", "I tried this for 30 days", "he said WHAT".
- **Something changes on screen every ~1s** even without a cut: a meter moves, a chip lands, a caption word
  pops, a callout draws. Combine with fast cutting for short-form; rely on it more for long-form.
- **Annotate what the viewer might miss** with `callout` on a tracked thing; **label effects** (auto
  `SLOW-MO 0.5x`); **pay off the stakes** (`slam` K.O. + sub "X IS DOWN", `title` WINNER / STILL STANDING).
- **Close with an end card** (brand, line, link, handle) over blurred footage when it's for a brand.
- Use **events in source time** for anything tied to the footage (hits, reveals, punchlines), so re-cuts
  never break the sync. Mark hooks/flash-forwards `"replay": true`.
- Genre sketches: *fight/game* = versus + tags(stat) + meters + slam + callout + title + endcard;
  *story* (clean) = lower_third + captions + title chapters + progress; *rage bait* (loud) = captions
  (max_words 2-3) + slam on the outrage line + chip reactions + fast push-ins.

## Workflow

1. **Inventory.** `ffprobe` the source, `juice-reel phrases <video>` for what's said and when, then
   `juice-reel filmstrip <video> <start> <end>` across the whole clip (4 windows) to see who is where.
2. **Propose** the shot list in plain English (shot → text → music) and wait for a yes.
3. **Write `reel.json`**, render, check. Look at `edit/verify/cuts.jpg` (the frame either side of every
   cut) and `edit/verify/overview.png` before showing anything. `check` exits non-zero on problems.
4. **Iterate.** One line of feedback = one change to the spec. Re-render; never re-transcribe.
5. Log decisions in `edit/project.md` (one short section per round).

## Cutting rules (creator's taste, learned on real edits)

- **Start a talking shot right as the person visibly starts talking**: `start = first_word.start - 0.08`.
  Get the word from `juice-reel words <video> --from A --to B`. Gesturing before speech is dead air.
- **Hard cuts between people.** Do not keep or speed up the camera being passed between people (whip pans);
  cut the handoff out entirely.
- **Walk-away endings** start with the back already turned and end as they start to turn back.
- A reaction shot can take a slow punch-in (`punch_in: 0.18`); freezes are optional and short.
- Every shot is rounded to whole frames; cut times in reports are frame-exact.

## Text rules

- (The SEO reel's caption style) Plain white, Helvetica Neue Bold 54px. **No outline, no fade in/out, no quotes, no emojis.**
  This was one creator's taste for one reel; for new work the theme decides.
- Hand-set line breaks so a phrase never splits awkwardly (`"lines": ["can we pull her", "SEO analytics?"]`).
- Short. Cut every word that doesn't serve the joke.
- **Vary placement per shot**, and because there's no outline, put each block over a **dark** part of that
  shot (shirt, curtain), checked across the whole shot since the camera drifts. `check` flags text in the
  target platform's UI areas (top bar, bottom caption area, side margins, right-side buttons).
- Don't label punchlines that the face already sells; a reaction shot usually needs no text.
- Keep the creator's personal context in mind: nothing about a person's looks unless they ask.

## Music rules

- Trending sounds: search the current month's trend lists for the platform(s) it's going to, then `juice-reel music search` /
  `music get` (yt-dlp; if an upload 403s, try another upload of the same song).
- Find the clean, recognizable part of the song with `juice-reel music lyrics <track>`. Avoid explicit
  verses entirely, not just at the edges.
- **Land the vocal entrance on a cut**: `juice-reel music offset <lyric_s> <cut_s>`.
- **The music must not run past the last shot or start a new hit right before the end**: end in the gap
  after a lyric line, and look at the waveform at the end of `overview.png` for a cut-off transient.
- **No music fade-out** (only the 30ms anti-click edge). Loudness -14 LUFS, true peak ≤ -1 dB.
- Sung vocals sometimes transcribe as a single `[singing]` event; then align by the waveform.
- You can't hear: report the numbers and let the creator judge the feel. Remind them that a sound baked into
  the file doesn't count toward a trend; to ride one, add the official sound in each app when posting.

## Action / gameplay edits (no dialogue)

- Find beats by looking, not listening: `juice-reel frames <video> t1 t2 ...` around each moment until you
  have the exact frame of the hit, the fall, the reveal.
- Structure that works: flash-forward hook (≤1.5s) → VS/intro card → speed-ramped build → slow-mo hit →
  stamp (K.O.) → payoff → winner title with a short freeze.
- **Put the music drop on the biggest hit**: find the drop by loudness (low-passed RMS jump), then
  `offset = drop_time - hit_output_time`. Back the hit with flash + shake + boom on the same frame.
- Vertical from 16:9: set `frame_x` per shot; when two subjects don't fit in a 9:16 crop, pan between them
  (`[from, to]`) rather than framing the empty space between them.
- Render both formats from one spec (`formats`) and check each; motion graphics adapt their layout.
- **Fast-cut ("ADHD") pacing:** shots of 0.45–1.35s, every cut on the beat grid (60/BPM), no two
  consecutive shots with the same framing: alternate wide / close-up / push-in / pull-out / speed ramp /
  slow-mo, and alternate subjects. Keep the payoff moments (hit, K.O., winner) a little longer.
- **Labels that follow people/objects:** keyframe the target's head box every ~1s through anything fast
  (ragdoll, camera whips) using gridded full-res frames, `juice-reel track`, inspect the debug sheet, add
  keys where it drifts, `hide` ranges where it's occluded or off-screen. Then `juice-reel tag` and use the
  tagged copy as the shot source (it keeps the original timestamps). Use the clean source under title cards.
- Close-ups that `follow` a target: aim at `follow_y` ≈ 1.2 so the head (and its tag) stays in frame.
- yt-dlp may get bot-blocked by YouTube after several downloads; SoundCloud (`scsearch5:<query>`) works.

## Files

- `reel.json` (the edit), `style.json` (optional, the creator's style overrides), `music/`, `transcripts/`
- Output: `edit/final.mp4`, overwritten in place. `edit/clips/` and `edit/verify/` hold only the
  current render's files and are rebuilt every run. Don't create new copies of the video per edit.
- Secrets: `ELEVENLABS_API_KEY` in the environment or a git-ignored `.env`. Never commit it.
