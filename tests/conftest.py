import pytest

from common.config import CLAUDE_AGENT_URL, OPENROUTER_AGENT_URL
from common.node_process import start_node, stop_node


@pytest.fixture(scope="session")
def openrouter_server():
    process = start_node("openrouter_node", OPENROUTER_AGENT_URL + ".well-known/agent.json")
    yield OPENROUTER_AGENT_URL
    stop_node(process)


@pytest.fixture(scope="session")
def claude_server():
    process = start_node("claude_node", CLAUDE_AGENT_URL + ".well-known/agent.json")
    yield CLAUDE_AGENT_URL
    stop_node(process)
