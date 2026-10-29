# juice: how to edit with it

There are two kinds of video, and each is short or long form:
- **portrait** (1080×1920: TikTok, Reels, Shorts) or **landscape** (1920×1080: YouTube, X, sites, decks)
- **short** (built to stop a scroll: triple hook in 1.5 s) or **long** (built to hold attention)

Portrait text stays clear of every app's UI, so one export posts everywhere.

Two ways to build one; pick by what the video is:
- **Cut** (`reel.json` + `juice render`): footage-led edits: talking heads, reactions, interviews, gameplay,
  stories. Every change the creator asks for is an edit to that file followed by
  `juice render reel.json && juice check reel.json`. Never hand-run ffmpeg for a cut.
- **Motion** (`juice.motion` script + `juice motion`): graphics-led videos: trailers, launches, product
  explainers, where voice-over runs under full-screen motion graphics and screen recordings. Never write a
  one-off compositor again; if a piece is missing, add it to `juice/motion/scenes.py`.

## Start every job (in parallel)

1. **Footage:** start `juice ingest <everything they gave you> -o footage` **in the background** right away.
   It unpacks archives (including stalled Chrome `Unconfirmed *.crdownload` Drive zips), repairs recordings
   whose moov is missing, transcribes speech and writes `footage/manifest.json` (updated as it goes, `done`
   flips at the end), `footage/contact.jpg` and `footage/INGEST.md`. Read the manifest while it works.
2. **Brand:** look for `juice.json` (project folder, then `https://<domain>/.well-known/juice.json`). None?
   `npx juice-press <site> <repo>` gathers one (colors, fonts, logo, line, link); then look at the site and
   fix what it got wrong (PROTOCOL step C) and `npx juice-press validate`. Everything after this is on-brand
   automatically: cut themes and motion scenes both read the kit.
3. **Story:** while those run, write the plan (beats, lines, scenes) from the transcripts as they land.

## Motion videos

`examples/openbeam-trailer/trailer.py` is the reference: a 27 s launch trailer in ~100 lines, in the clean
look (`trailer_hype.py` is the same story in the loud look).
- `Video(length="short", brand="juice.json")`; orientation comes from `juice motion --orientation`, so one
  script renders both. Lay every scene out by fraction of the frame, never hard-coded pixels.
- Voice: `v.voice(take, start, end, at)` places a line anywhere on the clock (the speaker keeps talking over
  other scenes); `line.out(src_t)` gives the output time of a word, so slams and captions land on it.
- Scenes fill the frame (`Footage`, `LogoReveal`, `Panels`, `Linked`, `FeatureGrid`, `Notification`,
  `EndCard`, `Stage`); overlays sit on top (`Slam`, `Headline`, `Words`, `Chip`, `Callout`, `Tracker`,
  `Letterbox`, `Label`, `Bug`, `UrlBar`). Each scene's `transition` (cut, flash, iris, logo, whip, glitch)
  brings its own sound; `v.hit()` adds booms, slams, landings, risers, typing.
- Two looks; pick one per video and don't mix their pieces. **Clean** (`juice.motion.clean`: launches,
  enterprise, anything that should feel expensive): `Title`, `Terms`, `Caption`, `Notice` over `Footage`,
  `Frame`, `Split`, `Bento`, `Lockup`; soft transitions (cut, fade, dissolve, wipe), quiet hits (sub, tick,
  air, swell), the `future` or `keynote` bed. Fast cuts stay, but they land on the beat (`v.beat(k)`) or on the spoken
  word, never in between; type rises through a mask with one ease, one margin, sentence case, the accent
  on one word at a time, nothing springs, flashes or shakes. **Loud** (`scenes`: hype edits): the rest.
- Music: `v.music("hype" | "tech" | "calm" | "keynote" | "future")` arranges an original bed (nothing to license) to the scenes'
  `energy` (0 drone, 1 pad + bass, 2 groove, 3 full), downbeat on the first cut, risers into every lift,
  set by loudness, not a fader: between lines it plays as loud as the voice (`level=0` dB), under a line it
  ducks `duck=12` dB, and it's leveled so no section jumps out. The render's `audio` report gives
  `voice_lufs` and `music_lufs`; they should be within ~1.5 dB. `juice beds` previews each style.
- Voice is isolated by default (DeepFilterNet3 via uvx, cached per take: chatter, traffic and room tone
  gone, the speech untouched). `v.voice(..., isolate=False)` keeps the room when it's part of the story.
- Loop: `juice motion x.py --stills` (one frame per scene, ~4 s) → fix → `--preview` (480p, ~6 s) →
  full render (~1 s of render per second of video). Look at the stills every round.
- After the render: check loudness (-14 LUFS), and transcribe the output once to confirm every line reads.

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
2. From a reference video: frames → read its fonts (condensed? heavy sans labels? never monospace, even if the reference uses it), palette, label shape (slanted
   chips?), shadow style, and its "story layer" (premise line, HUD, callouts, labeled slow-mo, end card).
3. `juice gallery --theme <it> --bg <a real frame>` and look before rendering anything.
4. With a `juice.json` brand kit, skip 1-2: its colors and fonts are the theme (`"brand"` in the spec, or
   just `juice.json` next to it), on top of `clean` unless the spec names another base.

## Triple hook (every video)

In the first ~1.5s three hooks land together, on the same idea. Viewers decide in under 3 seconds.
- **Spoken:** the first words start by ~0.4s and are the line itself (no "okay so…", no "um").
- **Text:** one short line on screen by 0.5s that frames it, readable in one glance (≤ 10 words).
  Extra labels/context come in **after** the hook (e.g. on the first cut), never stacked on top of it.
- **Visual:** frame one is the subject: a face mid-reaction, motion, something happening. Not a wide empty
  frame, not a slow start.

How to get there:
1. `juice hooks <all clips>` ranks every line across the clips (stands alone, short, laugh/reaction
   right after, energy, face on screen; interviewer lines and questions are pushed down) and writes a
   contact sheet. It's a shortlist: pick by reading the line, because the funniest/most wrong-sounding
   wording is a judgment call the score can't make.
2. Build the opening around the pick: trim to ~0.1s before its first word, put the one-line text hook on it.
3. `juice check` runs the triple-hook check (spoken start + filler, text words + timing, frame detail /
   motion / faces) and fails if any of the three is missing. Tune per video with
   `"style": {"hook": {"speech_by": 0.4, "text_by": 0.5, "max_words": 10, "require_spoken": "auto"}}`;
   music-only edits skip the spoken check automatically.

## Retention score (every video)

`juice retention reel.json` predicts the share of viewers still watching at every moment (a survival
model: front-loaded Weibull baseline × per-moment hazard multipliers, sources in `retention.py`) and
prints a 0-100 score (predicted average % viewed), 3s / end retention, the moments that lose the most
viewers and why (no visual change, dead air, filler, a repeat, slow pace, a weak hook), and writes
`edit/verify/retention.png`. `check` includes the summary. Use it as the edit loop, not a grade:
1. Score the spec before rendering (it needs no render). Fix the top peaks: cut the pause/repeat, add a
   push-in, chip or slam where nothing changes for > 2s, open on a question, caption all speech.
2. Re-score, then render. Aim for a positive `lift_pct` (this edit vs the same cut with neutral
   multipliers) and no peak over ~1% lost. Absolute numbers are calibrated defaults, the ranking is
   what to trust; when the creator has real retention curves, recalibrate `BASE` / `WEIGHTS`.
3. Levers, strongest first: word-level `captions` (whole video), a text hook by 0.5s that asks a question,
   speech by 0.4s, something new on screen every 1-3s (cut / `zoom` push-in / `chip` / `slam` on the key
   word via a source-time event), no silence ≥ 0.5s, no restarts, shorter total length.

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
  *story* (clean) = captions + question `title` hook + numbered `chip` chapters (`badge`) + slams on key
  words + progress; *rage bait* (loud) = captions (max_words 2-3) + slam on the outrage line + chip
  reactions + fast push-ins.
- **First-person vlogs get no lower third**: the creator is talking to camera, a name tag reads as a
  third-person interview. Name tags are for other people on screen.
- Graphics animate with springs and leave on purpose (`title`, `chip`, `captions` pop with overshoot and
  slide out); plain fades read as cheap.

## Workflow

1. **Inventory.** `ffprobe` the source, `juice phrases <video>` for what's said and when, then
   `juice filmstrip <video> <start> <end>` across the whole clip (4 windows) to see who is where.
   For dialogue footage, `juice hooks <clips>` for the hook shortlist.
2. **Propose** the shot list in plain English (shot → text → music) and wait for a yes.
3. **Write `reel.json`**, `juice render reel.json --preview` while iterating (480p, seconds), then render
   and check. Look at `edit/verify/cuts.jpg` (the frame either side of every
   cut) and `edit/verify/overview.png` before showing anything. `check` exits non-zero on problems.
4. **Iterate.** One line of feedback = one change to the spec. Re-render; never re-transcribe.
5. Log decisions in `edit/project.md` (one short section per round).

## Cutting rules (creator's taste, learned on real edits)

- **Never use the same footage twice.** No replays of an answer, no reusing a reaction, no freeze-frame
  of a moment that also plays. Every second of source appears once. `check` fails on overlapping ranges of
  the same clip; mark a deliberate flash-forward `"replay": true` only when the creator asks for one.
- **Use every clip the creator provides.** Find a place for each one (an intro line, a reaction, a button
  at the end). `check` fails on unused sources. Only leave a clip out if they say so.

- **Start a talking shot right as the person visibly starts talking**: `start = first_word.start - 0.08`.
  Get the word from `juice words <video> --from A --to B`. Gesturing before speech is dead air.
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
  apps' UI areas in portrait (top bar, bottom caption area, side margins, right-side buttons) and the
  title-safe margin in landscape.
- Don't label punchlines that the face already sells; a reaction shot usually needs no text.
- Keep the creator's personal context in mind: nothing about a person's looks unless they ask.

## Music rules

- Trending sounds: search the current month's trend lists for the platform(s) it's going to, then `juice music search` /
  `music get` (yt-dlp; if an upload 403s, try another upload of the same song).
- Find the clean, recognizable part of the song with `juice music lyrics <track>`. Avoid explicit
  verses entirely, not just at the edges.
- **Land the vocal entrance on a cut**: `juice music offset <lyric_s> <cut_s>`.
- **The music must not run past the last shot or start a new hit right before the end**: end in the gap
  after a lyric line, and look at the waveform at the end of `overview.png` for a cut-off transient.
- **No music fade-out** (only the 30ms anti-click edge). Loudness -14 LUFS, true peak ≤ -1 dB.
- Sung vocals sometimes transcribe as a single `[singing]` event; then align by the waveform.
- You can't hear: report the numbers and let the creator judge the feel. Remind them that a sound baked into
  the file doesn't count toward a trend; to ride one, add the official sound in each app when posting.

## Trend recipe: "switched questions"

People are filmed answering an innocent question (food, restaurants, travel); the edit claims a
different, unhinged question was asked, so the innocent answer sounds wrong.
- **No interviewer questions in the audio at all**, not the real question, not follow-ups ("A month?",
  "30 minutes max?", "Say a number."), not a spoken premise ("So the question is..."). People only answer.
  Interviewer reactions that aren't questions are fine ("Wow, you're generous"). Also cut answers that
  repeat the real question back ("how many times do I have to go there to call it a regular spot?").
- If the host introduces themselves on camera, put them right after the hook as the premise beat; if no
  footage of them explaining exists, set it up with one chip ("everyone thought we were asking about
  food...") and say so to the creator.
- **Caption the unhinged question once**, at the top, as a plain chip at the start of that person's
  section (an `overlay` chip with `from_shot`/`to_shot`); one question per person or per group of
  answers; a pop on each new question. No "what we really asked" reveal, no replay of the answer.
- Pick each fake question from the answer's wording so it lands hardest ("30 minute is the max I could go"
  → "how long can you last in bed?"; "my name is ___, what's your name?" → "what do you say right after a
  hookup?"). Clips where the unhinged question was genuinely asked go under that question, grouped.
- Open on the most wrong-sounding answer; end on the best reaction ("Bro, can we not do this?").
- Original audio only, no music; plain white chips (clean theme), no other graphics.

## Action / gameplay edits (no dialogue)

- Find beats by looking, not listening: `juice frames <video> t1 t2 ...` around each moment until you
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
  (ragdoll, camera whips) using gridded full-res frames, `juice track`, inspect the debug sheet, add
  keys where it drifts, `hide` ranges where it's occluded or off-screen. Then `juice tag` and use the
  tagged copy as the shot source (it keeps the original timestamps). Use the clean source under title cards.
- Close-ups that `follow` a target: aim at `follow_y` ≈ 1.2 so the head (and its tag) stays in frame.
- yt-dlp may get bot-blocked by YouTube after several downloads; SoundCloud (`scsearch5:<query>`) works.

## Files

- `reel.json` (the edit) or a motion script, `juice.json` (the brand kit), `style.json` (optional, the
  creator's style overrides), `footage/` (from `juice ingest`), `music/`, `transcripts/`
- Output: `edit/final.mp4`, overwritten in place. `edit/clips/` and `edit/verify/` hold only the
  current render's files and are rebuilt every run. Don't create new copies of the video per edit.
- Secrets: `ELEVENLABS_API_KEY` in the environment or a git-ignored `.env`. Never commit it.
