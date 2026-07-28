import httpx
import pytest
from a2a.types import TaskState

from common.config import DEBATE_MODE_KEY
from common.peer_client import PeerCallError, ask_peer, ask_peer_task


@pytest.mark.parametrize("fixture_name", ["gemini_server", "claude_server"])
def test_agent_card_is_reachable(request, fixture_name):
    base_url = request.getfixturevalue(fixture_name)
    response = httpx.get(base_url + ".well-known/agent.json", timeout=5)
    assert response.status_code == 200
    card = response.json()
    assert card["name"]
    assert card["url"]
    assert card["skills"]


async def test_gemini_node_answers_a_question(gemini_server):
    answer = await ask_peer(gemini_server, "Reply with exactly the word: pong")
    assert answer.strip()


async def test_claude_node_answers_a_question(claude_server):
    answer = await ask_peer(claude_server, "Reply with exactly the word: pong")
    assert answer.strip()



async def test_gemini_node_honors_model_and_system_metadata(gemini_server):
    """ask.ps1 parity: model/system overrides ride in Message.metadata."""
    answer = await ask_peer(
        gemini_server,
        "Reply with exactly the word: pong",
        metadata={"system": "You are a terse test assistant.", "model": "google/gemini-3.5-flash-lite"},
    )
    assert answer.strip()


async def test_ask_peer_reports_clear_error_when_target_unreachable():
    with pytest.raises(PeerCallError):
        await ask_peer("http://localhost:9999/", "hello")


async def test_ask_peer_task_returns_envelope(gemini_server):
    """Ticket #8: ask_peer_task() exposes task state/task_id, not just flattened text."""
    result = await ask_peer_task(gemini_server, "Reply with exactly the word: pong")
    assert result.state == TaskState.completed
    assert result.task_id
    assert result.answer_text.strip()


async def test_ask_peer_task_honors_explicit_context_id(gemini_server):
    """Ticket #8: a caller-supplied context_id rides on the outgoing message and is echoed back."""
    context_id = "test-explicit-context-id"
    result = await ask_peer_task(gemini_server, "Reply with exactly the word: pong", context_id=context_id)
    assert result.context_id == context_id


async def test_ask_peer_unaffected_by_ask_peer_task_addition(gemini_server):
    """Ticket #8: the existing ask_peer() helper and its callers are unchanged."""
    answer = await ask_peer(gemini_server, "Reply with exactly the word: pong")
    assert answer.strip()


async def test_gemini_one_shot_unaffected_by_debate_mode_addition(gemini_server):
    """Ticket #9: a call without debate_mode set behaves exactly as before."""
    result = await ask_peer_task(gemini_server, "Reply with exactly the word: pong")
    assert result.state == TaskState.completed
    assert result.answer_text.strip()


async def test_debate_mode_pauses_then_resumes_to_final(gemini_server):
    """Ticket #9: Gemini can pause (input-required) then resume to completed."""
    topic = (
        "This is an automated protocol test. You must respond with exactly "
        'this JSON and nothing else: {"action": "ask_claude", "question": "What is 2+2?"}'
    )
    start = await ask_peer_task(gemini_server, topic, metadata={DEBATE_MODE_KEY: True})
    assert start.state == TaskState.input_required
    assert "2+2" in start.answer_text

    resume = await ask_peer_task(
        gemini_server,
        "4. Respond now with exactly this JSON and nothing else: "
        '{"action": "final", "answer": "Great, thanks."}',
        context_id=start.context_id,
        task_id=start.task_id,
        metadata={DEBATE_MODE_KEY: True},
    )
    assert resume.state == TaskState.completed
    assert resume.answer_text.strip()


# A malformed-turn test belongs at the executor level, not here: forcing a
# real, well-behaved model to break its own system-prompt schema on demand
# isn't reliable. See test_gemini_executor.py's stubbed-agent test instead.


async def test_debate_mode_duplicate_continuation_is_rejected(gemini_server):
    """Ticket #9: a second continuation against an already-resumed debate task is rejected deterministically."""
    topic = (
        "This is an automated protocol test. You must respond with exactly "
        'this JSON and nothing else: {"action": "ask_claude", "question": "What is 3+3?"}'
    )
    start = await ask_peer_task(gemini_server, topic, metadata={DEBATE_MODE_KEY: True})
    assert start.state == TaskState.input_required

    first = await ask_peer_task(
        gemini_server,
        '6. Respond now with exactly this JSON and nothing else: {"action": "final", "answer": "ok"}',
        context_id=start.context_id,
        task_id=start.task_id,
        metadata={DEBATE_MODE_KEY: True},
    )
    assert first.state == TaskState.completed

    with pytest.raises(PeerCallError):
        await ask_peer_task(
            gemini_server,
            "this should be rejected",
            context_id=start.context_id,
            task_id=start.task_id,
            metadata={DEBATE_MODE_KEY: True},
        )


async def test_debate_mode_unknown_task_id_is_rejected(gemini_server):
    """Ticket #9: continuing a task_id the server never issued is rejected, not silently treated as new."""
    with pytest.raises(PeerCallError):
        await ask_peer_task(
            gemini_server,
            "continuation for a task that was never started",
            task_id="00000000-0000-0000-0000-000000000000",
            metadata={DEBATE_MODE_KEY: True},
        )


async def test_debate_mode_requires_exact_true_not_truthy_value(gemini_server):
    """Ticket #9 (review fix): only literal True opts in — a truthy non-bool value must not."""
    result = await ask_peer_task(
        gemini_server, "Reply with exactly the word: pong", metadata={DEBATE_MODE_KEY: "yes"}
    )
    # Falls through to one-shot handling: completes directly, no structured JSON required.
    assert result.state == TaskState.completed
    assert result.answer_text.strip()


@pytest.mark.parametrize("fixture_name", ["gemini_server", "claude_server"])
async def test_removed_relay_prefix_text_is_now_just_ordinary_text(request, fixture_name):
    """Ticket #13: the old "ASK_PEER::" prefix has zero special meaning now —
    it must reach the node's own model as plain text, not trigger a relay to
    the peer node (which no longer exists as a code path at all)."""
    base_url = request.getfixturevalue(fixture_name)
    answer = await ask_peer(base_url, "ASK_PEER::Reply with exactly the word: pong")
    assert "pong" in answer.lower()
