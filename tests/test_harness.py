"""The harness without a paid model: a fake OpenRouter drives the loop, and the MCP server is talked to
over stdio by the mcp client."""
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from harness import agent as agent_mod, openrouter, tools
from harness.workspace import Workspace

REPO = Path(__file__).resolve().parent.parent


class FakeRouter(BaseHTTPRequestHandler):
    """Scripted replies: each POST pops the next one; the requests are kept for the test to read."""
    script: list[dict] = []
    seen: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeRouter.seen.append(body)
        msg = FakeRouter.script.pop(0)
        reply = {"choices": [{"message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.001}}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def router(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeRouter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(openrouter, "URL", f"http://127.0.0.1:{srv.server_port}/chat")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    FakeRouter.script, FakeRouter.seen = [], []
    yield FakeRouter
    srv.shutdown()


def call(id_, name, **args):
    return {"id": id_, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_loop_runs_tools_shows_images_and_resumes(router, tmp_path):
    ws = Workspace(tmp_path / "proj")
    shutil.copytree(REPO / "examples" / "openbeam-trailer", ws.root / "trailer", ignore=shutil.ignore_patterns("out"))
    router.script = [
        {"role": "assistant", "content": "Looking at the beds and the clean look.",
         "tool_calls": [call("a", "beds"), call("b", "gallery", theme="clean")],
         "reasoning_details": [{"type": "reasoning.text", "text": "plan"}]},
        {"role": "assistant", "content": "Done: the clean look it is."},
    ]
    a = agent_mod.Agent(ws, openrouter.OpenRouter("fake/model"), trace=lambda s: None)
    answer = a.run("pick a look")
    assert answer == "Done: the clean look it is."

    first, second = router.seen
    assert first["messages"][0]["role"] == "system" and "juice" in first["messages"][0]["content"]
    assert first["tools"][0]["type"] == "function" and {t["function"]["name"] for t in first["tools"]} >= {"render", "check", "motion"}
    msgs = second["messages"]
    assistant = msgs[2]
    assert assistant["tool_calls"][0]["id"] == "a" and assistant["reasoning_details"]  # passed back verbatim
    assert msgs[3]["role"] == "tool" and msgs[3]["tool_call_id"] == "a" and "hype" in msgs[3]["content"]
    assert msgs[4]["role"] == "tool" and msgs[4]["tool_call_id"] == "b" and "gallery_clean.png" in msgs[4]["content"]
    pictures = msgs[5]
    assert pictures["role"] == "user"
    assert [p["type"] for p in pictures["content"]] == ["text", "text", "image_url"]
    assert pictures["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert a.stats.calls == 2 and a.stats.turns == 2 and a.stats.cost == pytest.approx(0.002)

    # the session is saved without image bytes and comes back with them
    saved = json.loads((ws.root / "agent" / "session.json").read_text())
    assert saved["messages"][5]["content"][2]["type"] == "image_path"
    b = agent_mod.Agent(ws, openrouter.OpenRouter("fake/model"), trace=lambda s: None)
    assert b.resume()
    assert b.messages[5]["content"][2]["image_url"]["url"].startswith("data:image/jpeg")
    assert b.stats.calls == 2
    log = [json.loads(ln) for ln in (a.log_path).read_text().splitlines()]
    assert [next(k for k in r if k != "t") for r in log] == ["user", "assistant", "tool", "tool", "assistant"]


def test_bad_tool_calls_come_back_as_errors(router, tmp_path):
    ws = Workspace(tmp_path / "proj")
    router.script = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "frames", "arguments": "{not json"}},
            call("y", "no_such_tool"),
            call("z", "write_file", path="../outside.txt", content="hi"),
            call("w", "filmstrip", media="missing.mov", start=0, end=1)]},
        {"role": "assistant", "content": "ok"},
    ]
    a = agent_mod.Agent(ws, openrouter.OpenRouter("fake/model"), trace=lambda s: None)
    assert a.run("go") == "ok"
    results = [m["content"] for m in router.seen[1]["messages"] if m["role"] == "tool"]
    assert "not valid JSON" in results[0]
    assert "no tool named" in results[1]
    assert "outside the project" in results[2]
    assert "no such file" in results[3]
    assert not (tmp_path / "outside.txt").exists()


def test_budget_and_turn_limits(router, tmp_path):
    ws = Workspace(tmp_path / "proj")
    router.script = [{"role": "assistant", "content": "", "tool_calls": [call("a", "beds")]}] * 3
    a = agent_mod.Agent(ws, openrouter.OpenRouter("fake/model"), max_turns=2, trace=lambda s: None)
    assert a.run("go").startswith("stopped after 2 turns")
    router.script = [{"role": "assistant", "content": "", "tool_calls": [call("a", "beds")]}] * 3
    b = agent_mod.Agent(ws, openrouter.OpenRouter("fake/model"), max_cost=0.0015, trace=lambda s: None)
    assert "budget" in b.run("go")


def test_every_tool_has_a_valid_schema_and_a_doc():
    for t in tools.TOOLS:
        assert t.parameters["type"] == "object" and t.parameters["additionalProperties"] is False
        for r in t.parameters["required"]:
            assert r in t.parameters["properties"], (t.name, r)
        assert len(t.description) > 20
    names = [t.name for t in tools.TOOLS]
    assert len(names) == len(set(names))


def test_docs_tool_reads_the_repo_docs(tmp_path):
    ws = Workspace(tmp_path / "proj")
    assert "shots[].start/end" in tools.run(ws, "docs", {"topic": "spec"}).text
    assert "Triple hook" in tools.run(ws, "docs", {"topic": "guide"}).text
    motion = tools.run(ws, "docs", {"topic": "motion"}).text
    assert "LogoReveal(" in motion and "Bento(" in motion
    assert "captions" in tools.run(ws, "docs", {"topic": "components"}).text


def test_mcp_server_lists_and_calls_tools(tmp_path):
    import anyio
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    ws = tmp_path / "proj"
    ws.mkdir()
    params = StdioServerParameters(command=sys.executable, args=["-m", "harness.mcp_server", "--project", str(ws)],
                                   cwd=str(REPO))

    async def go():
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                listed = await s.list_tools()
                names = {t.name for t in listed.tools}
                assert {"render", "check", "motion", "filmstrip", "docs"} <= names
                beds = await s.call_tool("beds", {})
                assert "hype" in beds.content[0].text
                gal = await s.call_tool("gallery", {"theme": "bold"})
                kinds = [c.type for c in gal.content]
                assert kinds == ["text", "image"] and gal.content[1].mime_type == "image/jpeg"

    anyio.run(go)
