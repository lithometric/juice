"""The juice agent harness: any model edits video through juice's tools.

    juice-agent "cut a 30s portrait reel from footage/, bold look, hook on the funniest line"
    juice-agent --model google/gemini-3.5-flash --project ~/edits/launch "..."
    juice-mcp --project ~/edits/launch        # the same tools over MCP, for Claude Code / Desktop

tools.py     the tool table: every juice command as a function with a JSON schema, returning text + images
openrouter.py   the model endpoint (OpenAI-style chat completions with tool calling, any model id)
agent.py     the loop: brief -> tool calls -> results (images attached) -> ... -> answer, logged per session
prompt.py    the system prompt, built from the editing guide (CLAUDE.md) and the spec reference (README.md)
cli.py       `juice-agent`: run a brief, then keep giving notes in the same session
mcp_server.py  `juice-mcp`: the tool table over MCP stdio (needs the `mcp` extra)
"""
