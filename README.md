# juice
Guava Juice is a browser based video editor that attaches to MCP for any ai model to come in and accurately and quickly edit uploaded content, built in Rust.

## In this repo

- [`reel/`](reel/): **juice**, the working prototype (Python): portrait or landscape video, short or long
  form. Cuts are one JSON spec (`juice render` / `juice check`); motion videos (trailers, launches) are a
  short `juice.motion` script (`juice motion`); `juice ingest` turns a pile of downloads into usable,
  transcribed footage. Its spec format is the edit model the Rust engine and MCP tools will use.
- Brand kits: [`juice-press`](../juice-press) (separate repo) gathers a brand's colors, fonts and logo
  into `juice.json`, which juice reads.
