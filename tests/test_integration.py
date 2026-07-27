import httpx
import pytest

from common.peer_client import ask_peer


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
