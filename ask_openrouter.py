"""ask.ps1 replacement: ask an OpenRouter model a question over A2A.

Starts the OpenRouter A2A node, sends one question through the real A2A protocol
(agent card discovery + message/send), prints the answer, and shuts the node
back down — so it behaves like a single stateless command, same as ask.ps1.

Usage:
    python ask_openrouter.py "your question" [--model MODEL] [--role ROLE | --system "system prompt"]
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

from common.config import OPENROUTER_AGENT_URL  # noqa: E402
from common.peer_client import PeerCallError, ask_peer  # noqa: E402
from common.roles import ROLES, RoleResolutionError, resolve_system_prompt  # noqa: E402


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
    raise RuntimeError(f"OpenRouter node did not become ready in time") from last_error


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="Inline prompt text (omit if using --prompt-file)")
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Read the prompt from this file instead — avoids shell quoting/line-break issues with long prompts",
    )
    parser.add_argument("--model", default=None, help="OpenRouter model id (defaults to OPENROUTER_MODEL, currently openai/gpt-5.6-luna)")
    parser.add_argument(
        "--role",
        default=None,
        help=f"Named system-prompt role (defaults to the adversarial reviewer). Valid roles: {', '.join(sorted(ROLES))}. Mutually exclusive with --system.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Raw system prompt, overriding any role. Mutually exclusive with --role.",
    )
    args = parser.parse_args()

    try:
        system_prompt = resolve_system_prompt(args.system, args.role)
    except RoleResolutionError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.prompt_file:
        prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    elif args.prompt:
        prompt_text = args.prompt
    else:
        parser.error("either prompt or --prompt-file is required")

    process = subprocess.Popen(
        [sys.executable, "-m", "openrouter_node"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_until_ready(OPENROUTER_AGENT_URL + ".well-known/agent.json")
        metadata = {}
        if args.model:
            metadata["model"] = args.model
        if system_prompt:
            metadata["system"] = system_prompt
        try:
            answer = await ask_peer(OPENROUTER_AGENT_URL, prompt_text, metadata=metadata)
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
