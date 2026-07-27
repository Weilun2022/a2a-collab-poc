import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"


def _wait_until_ready(agent_card_url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(agent_card_url, timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Server at {agent_card_url} did not become ready in time") from last_error


def _start_node(module: str, agent_card_url: str) -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC_DIR)},
    )
    try:
        _wait_until_ready(agent_card_url)
    except Exception:
        process.terminate()
        process.wait(timeout=5)
        raise
    return process


@pytest.fixture(scope="session")
def gemini_server():
    from common.config import GEMINI_AGENT_URL

    process = _start_node("gemini_node", GEMINI_AGENT_URL + ".well-known/agent.json")
    yield GEMINI_AGENT_URL
    process.terminate()
    process.wait(timeout=5)


@pytest.fixture(scope="session")
def claude_server():
    from common.config import CLAUDE_AGENT_URL

    process = _start_node("claude_node", CLAUDE_AGENT_URL + ".well-known/agent.json")
    yield CLAUDE_AGENT_URL
    process.terminate()
    process.wait(timeout=5)
