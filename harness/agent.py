"""The loop: brief -> the model calls tools -> results go back (pictures attached) -> ... -> an answer.

    agent = Agent(Workspace("~/edits/launch"), OpenRouter(model="anthropic/claude-opus-5.5"))
    answer = agent.run("cut a 30s portrait reel, bold look")
    answer = agent.run("tighter: lose the second reaction")      # same session, the model remembers

Pictures: an OpenAI-style tool message is text only, so after the tool results of a turn the loop adds one
user message carrying every image those tools produced, labeled with the call that made it.

The session (messages, cost, tool calls) is logged to <project>/agent/<stamp>.jsonl as it goes and the
messages are saved to <project>/agent/session.json so `--resume` can carry on with the same context.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import prompt, tools
from .openrouter import ModelError, OpenRouter
from .workspace import Workspace, image_data_url

IMAGE_PREFIX = "data:image/"


@dataclass
class Stats:
    turns: int = 0
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    tool_s: float = 0.0
    model_s: float = 0.0

    def add(self, usage: dict):
        self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
        self.completion_tokens += usage.get("completion_tokens", 0) or 0
        self.cost += usage.get("cost", 0) or 0

    def line(self) -> str:
        return (f"{self.turns} turns, {self.calls} tool calls, {self.prompt_tokens + self.completion_tokens:,} tokens, "
                f"${self.cost:.3f}, model {self.model_s:.0f}s, tools {self.tool_s:.0f}s")


class Agent:
    def __init__(self, ws: Workspace, model: OpenRouter, max_turns: int = 80, max_cost: float | None = None,
                 trace=None):
        self.ws, self.model = ws, model
        self.max_turns, self.max_cost = max_turns, max_cost
        self.trace = trace if trace is not None else (lambda s: print(s, file=sys.stderr, flush=True))
        self.messages: list[dict] = [{"role": "system", "content": prompt.system(ws.root)}]
        self.stats = Stats()
        self.tool_list = tools.openai_tools()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_path = ws.root / "agent" / f"{stamp}.jsonl"
        self.session_path = ws.root / "agent" / "session.json"

    # --- the loop ----------------------------------------------------------------------------

    def run(self, brief: str) -> str:
        self.messages.append({"role": "user", "content": brief})
        self._log({"user": brief})
        for _ in range(self.max_turns):
            if self.max_cost is not None and self.stats.cost > self.max_cost:
                return self._stop(f"stopped: spent ${self.stats.cost:.2f}, over the ${self.max_cost:.2f} budget")
            t0 = time.time()
            try:
                msg, usage = self.model.chat(self.messages, self.tool_list)
            except ModelError as e:
                return self._stop(f"model error: {e}")
            self.stats.model_s += time.time() - t0
            self.stats.turns += 1
            self.stats.add(usage)
            self.messages.append(_clean(msg))
            self._log({"assistant": _clean(msg), "usage": usage})
            if msg.get("content"):
                self.trace(f"\n{msg['content'].strip()}\n")
            calls = msg.get("tool_calls") or []
            if not calls:
                self._save()
                return msg.get("content") or ""
            images: list[tuple[str, Path]] = []
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    args, res = {}, tools.Result(f"error: arguments were not valid JSON ({e})")
                else:
                    self.trace(f"-> {name}({_short(args)})")
                    t1 = time.time()
                    res = tools.run(self.ws, name, args)
                    self.stats.tool_s += time.time() - t1
                self.stats.calls += 1
                self.trace("   " + _head(res.text))
                self.messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": res.text})
                self._log({"tool": name, "args": args, "result": res.text[:4000], "images": [str(p) for p in res.images]})
                images += [(f"{name}({_short(args, 60)})", p) for p in res.images]
            if images:
                self.messages.append(_image_message(images))
            self._save()
        return self._stop(f"stopped after {self.max_turns} turns")

    def _stop(self, why: str) -> str:
        self.trace(why)
        self._save()
        return why

    # --- bookkeeping -------------------------------------------------------------------------

    def _log(self, rec: dict):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"t": time.time(), **rec}, default=str) + "\n")

    def _save(self):
        """Messages with every image replaced by its path, so a session file stays small and can be resumed."""
        slim = []
        for m in self.messages:
            if isinstance(m.get("content"), list):
                parts = []
                for part in m["content"]:
                    if part.get("type") == "image_url":
                        parts.append({"type": "image_path", "path": part.get("path")})
                    else:
                        parts.append(part)
                slim.append({**m, "content": parts})
            else:
                slim.append(m)
        self.session_path.write_text(json.dumps({"model": self.model.model, "messages": slim, "stats": vars(self.stats)},
                                                indent=1, default=str))

    def resume(self) -> bool:
        if not self.session_path.exists():
            return False
        d = json.loads(self.session_path.read_text())
        msgs = []
        for m in d["messages"]:
            if isinstance(m.get("content"), list):
                parts = []
                for part in m["content"]:
                    if part.get("type") == "image_path":
                        p = Path(part["path"])
                        if p.exists():
                            parts.append({"type": "image_url", "image_url": {"url": image_data_url(p)}, "path": str(p)})
                        else:
                            parts.append({"type": "text", "text": f"(image {p.name} no longer on disk)"})
                    else:
                        parts.append(part)
                msgs.append({**m, "content": parts})
            else:
                msgs.append(m)
        msgs[0] = self.messages[0]  # the current system prompt
        self.messages = msgs
        self.stats = Stats(**{k: v for k, v in d.get("stats", {}).items() if k in Stats.__dataclass_fields__})
        return True


def _image_message(images: list[tuple[str, Path]]) -> dict:
    parts = [{"type": "text", "text": "Images produced by the tool calls above. Look at each before deciding what to do next."}]
    for label, p in images:
        parts.append({"type": "text", "text": f"{label} -> {p.name}:"})
        parts.append({"type": "image_url", "image_url": {"url": image_data_url(p)}, "path": str(p)})
    return {"role": "user", "content": parts}


def _clean(msg: dict) -> dict:
    """The assistant message as the API wants it back: role, content, tool_calls, reasoning details; no nulls."""
    keep = {k: v for k, v in msg.items() if k in ("role", "content", "tool_calls", "reasoning_details", "reasoning") and v is not None}
    keep.setdefault("role", "assistant")
    if "content" not in keep:
        keep["content"] = ""
    return keep


def _short(args: dict, n: int = 140) -> str:
    s = ", ".join(f"{k}={_val(v)}" for k, v in args.items())
    return s if len(s) <= n else s[: n - 1] + "…"


def _val(v) -> str:
    s = json.dumps(v) if not isinstance(v, str) else v
    return s if len(s) <= 80 else s[:79] + "…"


def _head(text: str, n: int = 160) -> str:
    first = text.strip().splitlines()[0] if text.strip() else "(empty)"
    return first if len(first) <= n else first[: n - 1] + "…"
