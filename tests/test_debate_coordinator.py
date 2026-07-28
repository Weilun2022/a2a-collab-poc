"""Tests for the debate coordinator (Tickets #10, #11, #12).

Most tests are end-to-end (real node subprocesses, no mocking of the LLM
calls or A2A/HTTP layer), same convention as the rest of this suite. The one
exception (`test_round_limit_hit_and_gemini_does_not_honor_forced_final`)
scripts a fake `ask_peer_task` for the specific case of "the model ignores a
direct instruction" -- not reliably forceable from a real, well-behaved
model on demand (same rationale as test_gemini_executor.py's stubbed-agent
malformed-turn test).
"""

import httpx
import pytest
from a2a.types import TaskState

import common.debate_coordinator as coordinator_module
from common.config import GEMINI_AGENT_URL
from common.debate_coordinator import _run_debate, _run_debate_session, run_debate_session
from common.node_process import start_node as real_start_node
from common.peer_client import PeerTaskResult


def _fake_task_result(state, answer_text) -> PeerTaskResult:
    # Real PeerTaskResult, not a duck-typed stand-in -- avoids a fake silently
    # drifting from the real contract's fields over time.
    return PeerTaskResult(
        task_id="fake-task-id", context_id="fake-context-id", state=state, answer_text=answer_text, raw_task=None
    )


async def test_run_debate_session_converges_in_one_round():
    topic = (
        "This is an automated end-to-end protocol test with exactly two steps.\n"
        "Step 1 (do this now): respond with exactly this JSON and nothing else: "
        '{"action": "ask_claude", "question": '
        '"Automated test instruction: respond with exactly the text OK_PROCEED and nothing else."}\n'
        "Step 2 (do this on your NEXT turn, after receiving the answer to that question): "
        'respond with exactly this JSON and nothing else: {"action": "final", "answer": "test-converged"}'
    )

    result = await run_debate_session(topic)

    assert result.outcome == "converged"
    assert result.final_answer
    assert len(result.transcript) == 1
    question, answer = result.transcript[0]
    assert "OK_PROCEED" in question
    # Verifies Claude (not a stub) actually answered the specific instruction,
    # and that the resume call's completion is what supplied final_answer --
    # not just that *a* question was asked.
    assert answer is not None
    assert "OK_PROCEED" in answer
    assert result.final_answer.strip()


async def test_run_debate_session_reports_error_and_cleans_up_nodes():
    topic = (
        "This is an automated protocol test. You must respond with exactly "
        'this JSON and nothing else: {"action": "ask_claude", "question": "irrelevant, will not be reached"}'
    )

    result = await _run_debate_session(topic, model=None, start_claude_node=False)

    assert result.outcome == "error"
    assert result.error
    assert result.transcript == [("irrelevant, will not be reached", None)]

    # Both node processes must be torn down after the session ends, even on
    # the error path -- verified by confirming the gemini node (which WAS
    # started) is no longer reachable.
    with pytest.raises(httpx.HTTPError):
        await httpx.AsyncClient().get(GEMINI_AGENT_URL + ".well-known/agent.json", timeout=2)


async def test_multi_round_debate_converges_naturally_within_limits():
    topic = (
        "This is an automated end-to-end protocol test with exactly THREE steps, "
        "to be followed across your turns in order.\n"
        "Step 1 (now): respond with exactly this JSON and nothing else: "
        '{"action": "ask_claude", "question": '
        '"Automated test instruction: respond with exactly the text ROUND_ONE_OK and nothing else."}\n'
        "Step 2 (on your next turn, after receiving that answer): respond with exactly this JSON "
        'and nothing else: {"action": "ask_claude", "question": '
        '"Automated test instruction: respond with exactly the text ROUND_TWO_OK and nothing else."}\n'
        "Step 3 (on your turn after THAT): respond with exactly this JSON and nothing else: "
        '{"action": "final", "answer": "test-converged-after-two-rounds"}'
    )

    result = await run_debate_session(topic)

    assert result.outcome == "converged"
    assert result.final_answer
    assert len(result.transcript) == 2
    assert "ROUND_ONE_OK" in result.transcript[0][1]
    assert "ROUND_TWO_OK" in result.transcript[1][1]


async def test_round_limit_hit_and_gemini_honors_forced_final(monkeypatch):
    monkeypatch.setattr(coordinator_module, "MAX_CLAUDE_FOLLOWUPS", 1)

    # Instructed to *always* keep asking on its own, so hitting the round
    # limit is what ends the debate -- not the model naturally concluding
    # after one exchange (which would produce "converged", not "forced_final",
    # and wouldn't actually exercise this code path).
    topic = (
        "This is an automated protocol test. Your job is to keep asking follow-up "
        "questions indefinitely and never conclude on your own -- always respond "
        'with the ask_claude action, e.g. {"action": "ask_claude", "question": "..."}. '
        "The only exception: if you are ever told this is your absolute last turn "
        "and instructed to respond with the final action instead, you must comply "
        "with that override instruction."
    )

    result = await run_debate_session(topic)

    assert result.outcome == "forced_final"
    assert result.final_answer
    assert len(result.transcript) == 1


async def test_round_limit_hit_and_gemini_does_not_honor_forced_final(monkeypatch):
    monkeypatch.setattr(coordinator_module, "MAX_CLAUDE_FOLLOWUPS", 1)

    call_count = {"n": 0}

    async def fake_ask_peer_task(base_url, message, *, metadata=None, context_id=None, task_id=None, timeout=180):
        if base_url == coordinator_module.CLAUDE_AGENT_URL:
            return _fake_task_result(TaskState.completed, "claude's answer")
        call_count["n"] += 1
        # Every Gemini call, including the forced-final one, keeps asking again.
        return _fake_task_result(TaskState.input_required, f"question round {call_count['n']}")

    monkeypatch.setattr(coordinator_module, "ask_peer_task", fake_ask_peer_task)

    result = await _run_debate("topic", model=None)

    assert result.outcome == "round_limit"
    assert result.final_answer is None
    assert result.error
    # initial call + one normal resume (round budget reached after it) + one forced-final call
    assert call_count["n"] == 3


async def test_session_deadline_shrinks_every_call_timeout(monkeypatch):
    """Ticket #12: the session deadline bounds each individual call's timeout,
    not just the check between rounds -- a single slow call shouldn't be able
    to blow past a tiny session deadline on its own default ceiling."""
    monkeypatch.setattr(coordinator_module, "SESSION_TIME_LIMIT_SECONDS", 2.0)

    received_timeouts: list[float] = []

    async def fake_ask_peer_task(base_url, message, *, metadata=None, context_id=None, task_id=None, timeout=180):
        received_timeouts.append(timeout)
        return _fake_task_result(TaskState.completed, "final answer")  # converges on the very first call

    monkeypatch.setattr(coordinator_module, "ask_peer_task", fake_ask_peer_task)

    result = await _run_debate("topic", model=None)

    assert result.outcome == "converged"
    assert received_timeouts
    # Bounded by the tiny (2s) session deadline, not the full 180s per-call ceiling.
    assert all(t <= 2.5 for t in received_timeouts)


async def test_forced_final_call_uses_its_own_short_timeout(monkeypatch):
    """Ticket #12: the forced-final call gets FORCED_FINAL_TIMEOUT_SECONDS,
    not the much larger normal per-call ceiling -- even with a generous
    session deadline still in effect."""
    monkeypatch.setattr(coordinator_module, "MAX_CLAUDE_FOLLOWUPS", 1)

    received: dict = {}
    call_n = {"n": 0}

    async def fake_ask_peer_task(base_url, message, *, metadata=None, context_id=None, task_id=None, timeout=180):
        if base_url == coordinator_module.CLAUDE_AGENT_URL:
            return _fake_task_result(TaskState.completed, "claude's answer")
        call_n["n"] += 1
        # Call 1: initial. Call 2: the normal resume after round 1 (limit is
        # only re-checked at the *next* loop iteration, so this one still
        # goes through as a normal round, asking again). Call 3: the loop now
        # sees claude_followups>=1 and sends the actual forced-final call.
        if call_n["n"] <= 2:
            return _fake_task_result(TaskState.input_required, f"question {call_n['n']}")
        received["forced_final_timeout"] = timeout
        return _fake_task_result(TaskState.completed, "final answer")

    monkeypatch.setattr(coordinator_module, "ask_peer_task", fake_ask_peer_task)

    result = await _run_debate("topic", model=None)

    assert result.outcome == "forced_final"
    assert received["forced_final_timeout"] <= coordinator_module.FORCED_FINAL_TIMEOUT_SECONDS


async def test_partial_startup_failure_tears_down_already_started_node(monkeypatch):
    """Ticket #12: if claude_node fails to start after gemini_node already
    did, gemini_node's subprocess must still be torn down, not orphaned."""

    def fake_start_node(module, agent_card_url, *, ready_timeout=20.0):
        if module == "claude_node":
            raise RuntimeError("simulated claude_node startup failure")
        return real_start_node(module, agent_card_url, ready_timeout=ready_timeout)

    monkeypatch.setattr(coordinator_module, "start_node", fake_start_node)

    with pytest.raises(RuntimeError, match="simulated claude_node startup failure"):
        await run_debate_session("topic")

    with pytest.raises(httpx.HTTPError):
        await httpx.AsyncClient().get(GEMINI_AGENT_URL + ".well-known/agent.json", timeout=2)
