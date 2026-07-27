"""Trigger a one-off A2A round trip between the Claude and Gemini peer nodes.

Usage:
    python trigger.py claude->gemini "your question"
    python trigger.py gemini->claude "your question"

Both nodes must already be running (`python -m claude_node`, `python -m gemini_node`).
"""

import argparse
import asyncio
import sys

from common.config import CLAUDE_AGENT_URL, GEMINI_AGENT_URL, RELAY_PREFIX
from common.peer_client import PeerCallError, ask_peer

DIRECTIONS = {
    "claude->gemini": CLAUDE_AGENT_URL,
    "gemini->claude": GEMINI_AGENT_URL,
}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=sorted(DIRECTIONS))
    parser.add_argument("question")
    args = parser.parse_args()

    source_url = DIRECTIONS[args.direction]
    print(f"[trigger] {args.direction}: sending to {source_url} -> relay -> peer")
    print(f"[trigger] question: {args.question}")

    try:
        answer = await ask_peer(source_url, RELAY_PREFIX + args.question)
    except PeerCallError as exc:
        print(f"[trigger] FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"[trigger] answer: {answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
