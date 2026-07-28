"""Start/stop helpers for running a node (claude_node/gemini_node) as a subprocess.

Shared by the debate coordinator (production use) and the test fixtures
(tests/conftest.py) so the readiness-wait/teardown logic isn't duplicated.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"


def wait_until_ready(agent_card_url: str, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(agent_card_url, timeout=2)
            if response.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_error = exc
        time.sleep(0.3)
    raise RuntimeError(f"Server at {agent_card_url} did not become ready in time") from last_error


def start_node(module: str, agent_card_url: str, *, ready_timeout: float = 20.0) -> subprocess.Popen:
    """Starts `module` (e.g. "gemini_node") as a subprocess and waits for it to be ready.

    On readiness failure, the process is terminated before the exception propagates
    -- callers never have to clean up a half-started node themselves.
    """
    process = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_ready(agent_card_url, timeout=ready_timeout)
    except Exception:
        stop_node(process)
        raise
    return process


def stop_node(process: subprocess.Popen, *, timeout: float = 5.0) -> None:
    """Terminates `process`, falling back to a hard kill if it won't exit in time."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
