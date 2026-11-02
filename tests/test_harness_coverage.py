"""The harness keeps up with juice: every `juice` command is a tool the agent can call, so a new command
can't ship without the agent being able to use it."""
import argparse

from harness import tools
from juice import cli

# commands whose tools are named differently, or split into several
TOOL_FOR = {"music": ["music_search", "music_get", "music_lyrics", "music_offset"]}


def commands() -> set[str]:
    seen = set()
    real = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        for a in self._actions:
            if isinstance(a, argparse._SubParsersAction):
                seen.update(a.choices)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = grab
    try:
        cli.main(["--help"])
    except SystemExit:
        pass
    finally:
        argparse.ArgumentParser.parse_args = real
    return seen


def test_every_juice_command_is_a_tool():
    names = {t.name for t in tools.TOOLS}
    cmds = commands()
    assert {"render", "check", "motion", "ingest"} <= cmds          # the parser was really read
    missing = {c for c in cmds if not set(TOOL_FOR.get(c, [c])) <= names}
    assert not missing, f"juice commands with no harness tool (add them to harness/tools.py): {sorted(missing)}"
