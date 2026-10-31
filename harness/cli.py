"""juice-agent: give a model a brief and a project folder; it edits the video with juice's tools.

  juice-agent "cut a 30s portrait reel from footage/, bold look"       # in the current folder
  juice-agent -p ~/edits/launch -m google/gemini-3.5-flash "..."       # another model, another folder
  juice-agent --resume "make the hook land on 'nobody asked'"          # carry on the last session
  juice-agent --once "..."                                             # no follow-up prompt (scripts, CI)
  juice-agent --tools                                                  # list the tools

After the agent answers it waits for your notes on the same session; an empty line or `q` ends it.
Needs OPENROUTER_API_KEY (environment or .env). Model ids are OpenRouter's (anthropic/claude-opus-5.5,
google/gemini-3.5-flash, openai/gpt-6.1-sol, qwen/qwen3.8-max-0902, ...): any model with tool calling and
image input works.
"""
from __future__ import annotations

import argparse
import sys

from . import tools
from .agent import Agent
from .openrouter import DEFAULT_MODEL, ModelError, OpenRouter
from .workspace import Workspace


def main(argv=None):
    ap = argparse.ArgumentParser(prog="juice-agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("brief", nargs="?", help="what to make (omit to be asked)")
    ap.add_argument("-p", "--project", default=".", help="project folder (default: current)")
    ap.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"OpenRouter model id (default {DEFAULT_MODEL})")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="reasoning effort, where the model supports it")
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--max-cost", type=float, help="stop past this many dollars (OpenRouter's usage.cost)")
    ap.add_argument("--resume", action="store_true", help="continue the project's last session")
    ap.add_argument("--once", action="store_true", help="answer the brief and exit; no follow-up prompt")
    ap.add_argument("--quiet", action="store_true", help="no tool trace on stderr")
    ap.add_argument("--tools", action="store_true", help="list the tools and exit")
    a = ap.parse_args(argv)

    if a.tools:
        for t in tools.TOOLS:
            print(f"{t.name:14s} {t.description.splitlines()[0]}")
        return
    try:
        model = OpenRouter(a.model, effort=a.effort)
    except ModelError as e:
        sys.exit(str(e))
    ws = Workspace(a.project)
    agent = Agent(ws, model, max_turns=a.max_turns, max_cost=a.max_cost, trace=(lambda s: None) if a.quiet else None)
    if a.resume and not agent.resume():
        print("nothing to resume in", ws.root / "agent", file=sys.stderr)
    print(f"juice-agent  {a.model}  project {ws.root}", file=sys.stderr)

    brief = a.brief or _ask("brief> ")
    while brief:
        answer = agent.run(brief)
        print("\n" + answer.strip() + "\n")
        print(f"[{agent.stats.line()}]", file=sys.stderr)
        if a.once:
            break
        brief = _ask("you> ")


def _ask(label: str) -> str:
    try:
        s = input(label).strip()
    except (EOFError, KeyboardInterrupt):
        return ""
    return "" if s.lower() in ("q", "quit", "exit") else s


if __name__ == "__main__":
    main()
