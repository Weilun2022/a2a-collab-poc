"""Ticket #6: run several rounds of each relay direction and report stability.

Starts both nodes itself, runs N rounds of claude->gemini and gemini->claude,
records success/failure per round, and writes a conclusion to
docs/poc-results.md.

Usage: python scripts/stability_check.py [rounds-per-direction]
"""

import asyncio
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import httpx  # noqa: E402

from common.config import CLAUDE_AGENT_URL, GEMINI_AGENT_URL, RELAY_PREFIX  # noqa: E402
from common.peer_client import PeerCallError, ask_peer  # noqa: E402

DIRECTIONS = {
    "claude->gemini": CLAUDE_AGENT_URL,
    "gemini->claude": GEMINI_AGENT_URL,
}


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
        time.sleep(0.5)
    raise RuntimeError(f"Server at {agent_card_url} did not become ready in time") from last_error


def _start_node(module: str, agent_card_url: str) -> subprocess.Popen:
    process = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(SRC_DIR)},
    )
    _wait_until_ready(agent_card_url)
    return process


async def run_rounds(direction: str, source_url: str, rounds: int) -> list[dict]:
    results = []
    for i in range(1, rounds + 1):
        question = f"Reply with exactly the word: pong{i}"
        started = time.monotonic()
        try:
            answer = await ask_peer(source_url, RELAY_PREFIX + question)
            elapsed = time.monotonic() - started
            ok = f"pong{i}".lower() in answer.lower()
            results.append(
                {"round": i, "success": ok, "elapsed_s": round(elapsed, 2), "answer": answer.strip(), "error": None}
            )
        except PeerCallError as exc:
            elapsed = time.monotonic() - started
            results.append({"round": i, "success": False, "elapsed_s": round(elapsed, 2), "answer": None, "error": str(exc)})
        print(f"[{direction}] round {i}/{rounds}: {results[-1]}")
    return results


def write_report(all_results: dict[str, list[dict]], rounds: int) -> Path:
    lines = [
        "# A2A POC Stability Check Results",
        "",
        f"Run at: {datetime.now(timezone.utc).isoformat()}",
        f"Rounds per direction: {rounds}",
        "",
    ]
    overall_ok = True
    for direction, results in all_results.items():
        successes = sum(1 for r in results if r["success"])
        overall_ok = overall_ok and (successes == len(results))
        lines.append(f"## {direction}: {successes}/{len(results)} succeeded")
        lines.append("")
        lines.append("| Round | Success | Elapsed (s) | Answer / Error |")
        lines.append("|---|---|---|---|")
        for r in results:
            detail = r["answer"] if r["success"] else r["error"]
            lines.append(f"| {r['round']} | {r['success']} | {r['elapsed_s']} | {detail} |")
        lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    if overall_ok:
        lines.append(
            "All rounds succeeded in both directions. The A2A protocol reliably carries "
            "requests and responses between the Claude Agent SDK node and the OpenRouter "
            "(Gemini) node on this machine. Stable enough to consider replacing `ask.ps1`."
        )
    else:
        lines.append(
            "At least one round failed — see the failure details above before relying on "
            "this for day-to-day use. Investigate the failure mode (timeout, connection, "
            "malformed response) before replacing `ask.ps1`."
        )

    report_path = REPO_ROOT / "docs" / "poc-results.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


async def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    gemini_process = _start_node("gemini_node", GEMINI_AGENT_URL + ".well-known/agent.json")
    claude_process = _start_node("claude_node", CLAUDE_AGENT_URL + ".well-known/agent.json")

    try:
        all_results = {}
        for direction, source_url in DIRECTIONS.items():
            all_results[direction] = await run_rounds(direction, source_url, rounds)
    finally:
        gemini_process.terminate()
        claude_process.terminate()
        gemini_process.wait(timeout=5)
        claude_process.wait(timeout=5)

    report_path = write_report(all_results, rounds)
    print(f"\nReport written to {report_path}")

    total_failures = sum(1 for results in all_results.values() for r in results if not r["success"])
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
