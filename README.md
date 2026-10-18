# juice
Guava Juice is a browser based video editor that attaches to MCP for any ai model to come in and accurately and quickly edit uploaded content, built in Rust.

## In this repo

- [`reel/`](reel/): **juice-reel**, the working prototype for short-form video (TikTok, Reels, Shorts, ...). Each video is one JSON spec
  (shots, text, music); `juice-reel render` builds it in seconds and `juice-reel check` verifies it.
  Its spec format is the edit model the Rust engine and MCP tools will use.
