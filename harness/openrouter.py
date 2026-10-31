"""The model endpoint: OpenRouter's chat completions (OpenAI-style, any model id, tool calling, images).

    client = OpenRouter(model="anthropic/claude-opus-5.5")
    msg, usage = client.chat(messages, tools)

Reasoning models (Claude, Gemini, GPT) return `reasoning_details` with the assistant message; the loop
passes the message back verbatim so their thinking stays attached to the tool calls it produced.
"""
from __future__ import annotations

import os
import time

import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-opus-5.5"


class ModelError(RuntimeError):
    pass


class OpenRouter:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, effort: str | None = None,
                 max_tokens: int = 16000, timeout: float = 900):
        self.model = model
        self.key = api_key or os.environ.get("OPENROUTER_API_KEY") or _dotenv("OPENROUTER_API_KEY")
        if not self.key:
            raise ModelError("OPENROUTER_API_KEY is not set (environment or .env)")
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[dict, dict]:
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens,
                "usage": {"include": True}}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "HTTP-Referer": "https://github.com/lithometric/juice", "X-Title": "juice"}
        delay = 2.0
        for attempt in range(5):
            r = requests.post(URL, json=body, headers=headers, timeout=self.timeout)
            if r.status_code in (408, 409, 429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code != 200:
                raise ModelError(f"{r.status_code}: {r.text[:800]}")
            d = r.json()
            if "error" in d:  # OpenRouter reports some upstream failures inside a 200
                raise ModelError(str(d["error"])[:800])
            choice = d["choices"][0]
            msg = choice["message"]
            msg["finish_reason"] = choice.get("finish_reason")
            return msg, d.get("usage") or {}
        raise ModelError("gave up after retries")


def _dotenv(name: str) -> str | None:
    try:
        for ln in open(".env"):
            k, _, v = ln.strip().partition("=")
            if k == name:
                return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return None
