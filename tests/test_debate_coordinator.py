"""Tests for the debate coordinator (Tickets #10, #11, #12).

Most tests are end-to-end (real node subprocesses, no mocking of the LLM
calls or A2A/HTTP layer), same convention as the rest of this suite. The one
exception (`test_round_limit_hit_and_openrouter_does_not_honor_forced_final`)
scripts a fake `ask_peer_task` for the specific case of "the model ignores a
direct instruction" -- not reliably forceable from a real, well-behaved
model on demand (same rationale as test_openrouter_executor.py's stubbed-agent
malformed-turn test).
"""

import httpx
import pytest
from a2a.types import TaskState

import common.debate_coordinator as coordinator_module
from common.config import OPENROUTER_AGENT_URL
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
    # See test_multi_round_debate_converges_naturally_within_limits for why this
    # counts rendered "Round N" lines instead of using "next turn" ordinal
    # phrasing -- openai/gpt-5.6-luna (OPENROUTER_MODEL's current default)
    # doesn't reliably track ordinal turn phrasing the way the prior default did.
    topic = (
        "This is an automated end-to-end protocol test. Count how many times you have "
        "already been asked a question in this transcript (look at the 'Round N' lines "
        "above, if any) and follow exactly ONE of these two rules for THIS turn only:\n"
        "- If you see ZERO prior rounds in the transcript: respond with exactly this JSON "
        'and nothing else: {"action": "ask_claude", "question": '
        '"Automated test instruction: respond with exactly the text OK_PROCEED and nothing else."}\n'
        "- If you see exactly ONE prior round in the transcript: respond with exactly this JSON "
        'and nothing else: {"action": "final", "answer": "test-converged"}'
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
    # the error path -- verified by confirming the openrouter node (which WAS
    # started) is no longer reachable.
    with pytest.raises(httpx.HTTPError):
        await httpx.AsyncClient().get(OPENROUTER_AGENT_URL + ".well-known/agent.json", timeout=2)


async def test_multi_round_debate_converges_naturally_within_limits():
    # Phrased so the model must read the running transcript (rendered by
    # _render_transcript) to decide which turn it's on, rather than relying on
    # "next turn"/"turn after that" ordinal language -- an earlier version of
    # this topic used ordinal phrasing and openai/gpt-5.6-luna (the model
    # actually configured as OPENROUTER_MODEL's default) repeatedly re-asked
    # round 1's question instead of progressing, where google/gemini-3.5-flash-lite
    # (the prior default) had reliably progressed. Counting rendered "Round N"
    # lines directly converges deterministically with both models.
    topic = (
        "This is an automated end-to-end protocol test. Count how many times you have "
        "already been asked a question in this transcript (look at the 'Round N' lines "
        "above, if any) and follow exactly ONE of these three rules for THIS turn only:\n"
        "- If you see ZERO prior rounds in the transcript: respond with exactly this JSON "
        'and nothing else: {"action": "ask_claude", "question": '
        '"Automated test instruction: respond with exactly the text ROUND_ONE_OK and nothing else."}\n'
        "- If you see exactly ONE prior round in the transcript: respond with exactly this JSON "
        'and nothing else: {"action": "ask_claude", "question": '
        '"Automated test instruction: respond with exactly the text ROUND_TWO_OK and nothing else."}\n'
        "- If you see exactly TWO prior rounds in the transcript: respond with exactly this JSON "
        'and nothing else: {"action": "final", "answer": "test-converged-after-two-rounds"}'
    )

    result = await run_debate_session(topic)

    assert result.outcome == "converged"
    assert result.final_answer
    assert len(result.transcript) == 2
    assert "ROUND_ONE_OK" in result.transcript[0][1]
    assert "ROUND_TWO_OK" in result.transcript[1][1]


async def test_round_limit_hit_and_openrouter_honors_forced_final(monkeypatch):
    monkeypatch.setattr(coordinator_module, "MAX_CLAUDE_FOLLOWUPS", 1)

    # Instructed to *always* keep asking on its own, so hitting the round
    # limit is what ends the debate -- not the model naturally concluding
    # after one exchange (which would produce "converged", not "forced_final",
    # and wouldn't actually exercise this code path).
    #
    # Framed explicitly as an authorized QA-harness test of the round-limit
    # feature itself (not a bare "ignore your judgment and loop forever"
    # instruction) -- openai/gpt-5.6-luna (OPENROUTER_MODEL's current default)
    # was observed to sometimes refuse the bare instruction outright, treating
    # it as a suspicious behavioral override, and answer directly instead
    # (still passing the forced-final check, but not exercising the "keeps
    # asking" branch this test targets). The prior default, google/gemini-3.5-flash-lite,
    # complied with the bare framing reliably.
    topic = (
        "You are participating in an authorized, automated QA harness test for this "
        "system's own round-limit safety feature (not a real design discussion, and "
        "not an attempt to get you to behave unsafely -- it is testing whether a "
        "hard cap correctly interrupts an open-ended exchange). For this test only: "
        "on every turn, respond with the ask_claude action asking any simple filler "
        'question, e.g. {"action": "ask_claude", "question": "..."} -- UNLESS the '
        "message you receive explicitly states you have reached the maximum number "
        "of exchanges and instructs you to respond with the final action instead, in "
        "which case you must comply with that instruction exactly, since that is the "
        "specific behavior this test is verifying."
    )

    result = await run_debate_session(topic)

    assert result.outcome == "forced_final"
    assert result.final_answer
    assert len(result.transcript) == 1


async def test_round_limit_hit_and_openrouter_does_not_honor_forced_final(monkeypatch):
    monkeypatch.setattr(coordinator_module, "MAX_CLAUDE_FOLLOWUPS", 1)

    call_count = {"n": 0}

    async def fake_ask_peer_task(base_url, message, *, metadata=None, context_id=None, task_id=None, timeout=180):
        if base_url == coordinator_module.CLAUDE_AGENT_URL:
            return _fake_task_result(TaskState.completed, "claude's answer")
        call_count["n"] += 1
        # Every OpenRouter call, including the forced-final one, keeps asking again.
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
    """Ticket #12: if claude_node fails to start after openrouter_node already
    did, openrouter_node's subprocess must still be torn down, not orphaned."""

    def fake_start_node(module, agent_card_url, *, ready_timeout=20.0):
        if module == "claude_node":
            raise RuntimeError("simulated claude_node startup failure")
        return real_start_node(module, agent_card_url, ready_timeout=ready_timeout)

    monkeypatch.setattr(coordinator_module, "start_node", fake_start_node)

    with pytest.raises(RuntimeError, match="simulated claude_node startup failure"):
        await run_debate_session("topic")

    with pytest.raises(httpx.HTTPError):
        await httpx.AsyncClient().get(OPENROUTER_AGENT_URL + ".well-known/agent.json", timeout=2)
