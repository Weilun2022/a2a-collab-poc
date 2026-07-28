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

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, MessageSendParams, Part, Role, Task, TaskState, TaskStatus, TextPart
from a2a.utils.errors import ServerError

from common.config import DEBATE_MODE_KEY
from gemini_node.debate import MalformedDebateTurn
from gemini_node.executor import MAX_TRANSCRIPT_CHARS, GeminiAgentExecutor, _DebateSession, _render_transcript


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


def _build_continuation_context(task_id: str, context_id: str, answer_text: str) -> RequestContext:
    message = Message(
        role=Role.user,
        message_id=str(uuid.uuid4()),
        parts=[Part(root=TextPart(text=answer_text))],
        metadata={DEBATE_MODE_KEY: True},
    )
    # A still-non-terminal (input-required) task from the store's perspective
    # -- the a2a-sdk framework itself only rejects continuations against an
    # UNKNOWN or already-TERMINAL task_id; it does not know this task was
    # already claimed by our own in-process session bookkeeping.
    fake_task = Task(id=task_id, contextId=context_id, status=TaskStatus(state=TaskState.input_required))
    return RequestContext(
        request=MessageSendParams(message=message), task_id=task_id, context_id=context_id, task=fake_task
    )


async def test_duplicate_continuation_while_still_paused_is_rejected():
    """Ticket #9 (review fix): a genuine duplicate/racing continuation for a
    task that's still input-required (already claimed by an in-flight
    continuation, awaiting_continuation already False) must be rejected --
    this is the one edge case the a2a-sdk framework itself doesn't catch
    (it only rejects unknown/already-terminal task_ids, not "claimed but
    still non-terminal"). A prior test named for this scenario actually only
    exercised continuation-after-terminal-completion; real concurrent HTTP
    racing isn't reliably reproducible, so this drives the executor directly."""
    executor = GeminiAgentExecutor()
    task_id = "seeded-task-id"
    context_id = "seeded-context-id"
    executor._debate_sessions[task_id] = _DebateSession(
        topic="topic", pending_question="already answered by the winning continuation", awaiting_continuation=False
    )

    context = _build_continuation_context(task_id, context_id, "a second, duplicate answer")
    queue = EventQueue()

    with pytest.raises(ServerError):
        await executor.execute(context, queue)

    # The legitimate (already-claimed) session must be left untouched by the
    # rejected duplicate -- not popped, not mutated back to awaiting again.
    assert executor._debate_sessions[task_id].awaiting_continuation is False


def test_render_transcript_unbounded_case_includes_everything():
    session = _DebateSession(topic="short topic", exchanges=[("q1", "a1"), ("q2", "a2")])
    rendered = _render_transcript(session)
    assert "short topic" in rendered
    assert "q1" in rendered and "a1" in rendered
    assert "q2" in rendered and "a2" in rendered
    assert "omitted" not in rendered


def test_render_transcript_truncates_oldest_rounds_when_over_budget():
    # Each exchange is long enough that many rounds together blow past the bound.
    long_q = "Q" * 500
    long_a = "A" * 500
    exchanges = [(f"{long_q}-{i}", f"{long_a}-{i}") for i in range(20)]
    session = _DebateSession(topic="topic", exchanges=exchanges)

    rendered = _render_transcript(session)

    assert len(rendered) <= MAX_TRANSCRIPT_CHARS
    assert "topic" in rendered
    assert "[earlier rounds omitted for length]" in rendered
    # The oldest round must be dropped, the most recent round must survive.
    assert f"{long_q}-0" not in rendered
    assert f"{long_q}-19" in rendered
