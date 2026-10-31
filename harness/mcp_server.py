"""juice-mcp: the same tool table over MCP (stdio), so Claude Code, Claude Desktop or any MCP client
edits with juice directly, no harness of ours in the loop.

  juice-mcp --project ~/edits/launch

Claude Code:   claude mcp add juice -- uv run --project /path/to/juice juice-mcp --project ~/edits/launch
Claude Desktop (claude_desktop_config.json):
  {"mcpServers": {"juice": {"command": "uv", "args": ["run", "--project", "/path/to/juice", "juice-mcp",
                                                        "--project", "/Users/you/edits/launch"]}}}

Needs the `mcp` extra (mcp >= 2): uv sync --extra mcp. Images come back as image content, so the model sees them.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from . import tools
from .workspace import Workspace, image_bytes


def build(ws: Workspace):
    try:
        from mcp.server import Server
        from mcp.types import CallToolResult, ImageContent, ListToolsResult, TextContent, Tool
    except ImportError:
        sys.exit("the mcp package is missing: uv sync --extra mcp")
    import base64

    async def list_tools(ctx, params) -> ListToolsResult:
        return ListToolsResult(tools=[Tool(name=t.name, description=t.description, inputSchema=t.parameters)
                                      for t in tools.TOOLS])

    async def call_tool(ctx, params) -> CallToolResult:
        res = await asyncio.to_thread(tools.run, ws, params.name, dict(params.arguments or {}))
        content = [TextContent(type="text", text=res.text)]
        for p in res.images:
            data, mime = image_bytes(p)
            content.append(ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType=mime))
        return CallToolResult(content=content, isError=res.text.startswith("error:"))

    def schema(name: str):
        try:
            return tools.by_name(name).parameters
        except KeyError:
            return None

    return Server("juice", instructions="Video editing with juice: cuts from reel.json, motion videos from a script. "
                  "Call docs(\"guide\") first for how to edit.", on_list_tools=list_tools, on_call_tool=call_tool,
                  get_tool_input_schema=schema)


async def serve(ws: Workspace):
    from mcp.server.stdio import stdio_server
    server = build(ws)
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="juice-mcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--project", default=".", help="project folder the tools work in")
    a = ap.parse_args(argv)
    asyncio.run(serve(Workspace(a.project)))


if __name__ == "__main__":
    main()
