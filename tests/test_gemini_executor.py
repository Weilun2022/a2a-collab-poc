"""Direct executor-level tests for gemini_node (Ticket #9).

These construct a real RequestContext/EventQueue (the actual A2A machinery
used in production) but substitute a stub agent, because forcing a real LLM
to produce malformed debate-mode output on demand isn't reliable -- a
well-behaved model complies with the system prompt's schema even when the
user turn tries to provoke noncompliance. What's under test here is the
executor's own reaction to a malformed decision, not the model's compliance
(that's covered by the real end-to-end happy-path tests in
test_integration.py, and by the pure parser tests in test_debate.py).
"""

import uuid

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, MessageSendParams, Part, Role, TaskState, TextPart

from common.config import DEBATE_MODE_KEY
from gemini_node.debate import MalformedDebateTurn
from gemini_node.executor import GeminiAgentExecutor


class _StubAgentAlwaysMalformed:
    async def debate_turn(self, transcript: str, *, model: str | None = None):
        raise MalformedDebateTurn("stubbed malformed turn for test")


def _build_context(topic: str, *, debate_mode: bool = True) -> RequestContext:
    message = Message(
        role=Role.user,
        message_id=str(uuid.uuid4()),
        parts=[Part(root=TextPart(text=topic))],
        metadata={DEBATE_MODE_KEY: debate_mode},
    )
    return RequestContext(request=MessageSendParams(message=message))


async def test_malformed_debate_turn_fails_task_deterministically():
    executor = GeminiAgentExecutor()
    executor.agent = _StubAgentAlwaysMalformed()

    context = _build_context("this will be answered by a stub, not a real model")
    queue = EventQueue()

    await executor.execute(context, queue)

    working_event = await queue.dequeue_event(no_wait=True)
    assert working_event.status.state == TaskState.working

    failed_event = await queue.dequeue_event(no_wait=True)
    assert failed_event.status.state == TaskState.failed
    assert failed_event.final is True
    assert "malformed" in failed_event.status.message.parts[0].root.text.lower()

    # The failed session must not be left retrievable for a later continuation.
    assert context.task_id not in executor._debate_sessions
