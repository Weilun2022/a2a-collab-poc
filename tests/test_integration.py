import httpx
import pytest
from a2a.types import TaskState

from common.config import RELAY_PREFIX
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


async def test_claude_relays_question_to_gemini(claude_server, gemini_server):
    """Ticket #4: Claude node acts as an A2A client and calls the Gemini node."""
    answer = await ask_peer(claude_server, RELAY_PREFIX + "Reply with exactly the word: pong")
    assert answer.strip()


async def test_gemini_relays_question_to_claude(claude_server, gemini_server):
    """Ticket #5: Gemini node acts as an A2A client and calls the Claude node."""
    answer = await ask_peer(gemini_server, RELAY_PREFIX + "Reply with exactly the word: pong")
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
