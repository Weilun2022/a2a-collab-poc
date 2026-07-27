from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import InternalError
from a2a.utils import completed_task, new_text_artifact
from a2a.utils.errors import ServerError

from claude_node.agent import ClaudeAgent
from common.config import GEMINI_AGENT_URL, RELAY_PREFIX
from common.peer_client import PeerCallError, ask_peer


class ClaudeAgentExecutor(AgentExecutor):
    """Answers incoming A2A messages using Claude (via Claude Agent SDK).

    A message prefixed with RELAY_PREFIX is instead relayed to the Gemini peer
    node as an outbound A2A call, and that peer's answer is returned as-is —
    this is what lets this node act as an A2A client, not just a server.
    """

    def __init__(self):
        self.agent = ClaudeAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        try:
            if query.startswith(RELAY_PREFIX):
                answer = await ask_peer(GEMINI_AGENT_URL, query[len(RELAY_PREFIX) :])
            else:
                answer = await self.agent.ask(query)
        except PeerCallError as exc:
            raise ServerError(error=InternalError(message=str(exc))) from exc
        except Exception as exc:
            raise ServerError(error=InternalError(message=str(exc))) from exc

        artifact = new_text_artifact(name=f"claude_{context.task_id}", text=answer)
        await event_queue.enqueue_event(
            completed_task(
                context.task_id,
                context.context_id,
                [artifact],
                [context.message],
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=InternalError(message="cancel is not supported"))
