from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import InternalError
from a2a.utils import completed_task, new_text_artifact
from a2a.utils.errors import ServerError

from claude_node.agent import ClaudeAgent


class ClaudeAgentExecutor(AgentExecutor):
    """Answers incoming A2A messages using Claude (via Claude Agent SDK)."""

    def __init__(self):
        self.agent = ClaudeAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        try:
            answer = await self.agent.ask(query)
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
