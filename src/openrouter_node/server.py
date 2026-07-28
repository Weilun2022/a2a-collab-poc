from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from common.config import OPENROUTER_AGENT_URL
from openrouter_node.executor import OpenRouterAgentExecutor


def build_agent_card() -> AgentCard:
    skill = AgentSkill(
        id="ask_question",
        name="Ask a Question",
        description="Answers a free-text question or gives an opinion on a topic.",
        tags=["question answering", "second opinion"],
        examples=["What's a good approach to caching this API response?"],
    )
    return AgentCard(
        name="openrouter_agent",
        description="Answers questions using an OpenRouter-hosted model (configurable per call).",
        url=OPENROUTER_AGENT_URL,
        version="1.0.0",
        defaultInputModes=["text", "text/plain"],
        defaultOutputModes=["text", "text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )


def build_app() -> A2AStarletteApplication:
    request_handler = DefaultRequestHandler(
        agent_executor=OpenRouterAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=build_agent_card(), http_handler=request_handler)
