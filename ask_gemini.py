"""ask.ps1 replacement: ask an OpenRouter model a question over A2A.

Starts the Gemini A2A node, sends one question through the real A2A protocol
(agent card discovery + message/send), prints the answer, and shuts the node
back down — so it behaves like a single stateless command, same as ask.ps1.

Usage:
    python ask_gemini.py "your question" [--model MODEL] [--system "system prompt"]
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from common.config import GEMINI_AGENT_URL  # noqa: E402
from common.peer_client import PeerCallError, ask_peer  # noqa: E402


def _wait_until_ready(agent_card_url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(agent_card_url, timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.3)
    raise RuntimeError(f"Gemini node did not become ready in time") from last_error


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--model", default=None, help="OpenRouter model id (defaults to google/gemini-3.5-flash-lite)")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    args = parser.parse_args()

    process = subprocess.Popen(
        [sys.executable, "-m", "gemini_node"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until_ready(GEMINI_AGENT_URL + ".well-known/agent.json")
        metadata = {}
        if args.model:
            metadata["model"] = args.model
        if args.system:
            metadata["system"] = args.system
        try:
            answer = await ask_peer(GEMINI_AGENT_URL, args.prompt, metadata=metadata)
        except PeerCallError as exc:
            print(f"A2A call failed: {exc}", file=sys.stderr)
            return 1
        print(answer)
        return 0
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
