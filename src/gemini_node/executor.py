from dataclasses import dataclass, field

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InternalError, Part, TextPart
from a2a.utils import completed_task, new_text_artifact
from a2a.utils.errors import ServerError

from common.config import DEBATE_MODE_KEY
from gemini_node.agent import OpenRouterAgent
from gemini_node.debate import MalformedDebateTurn


@dataclass
class _DebateSession:
    """In-process bookkeeping for one debate-mode task across its pause/resume calls.

    Lives only in this executor instance's memory for the process's lifetime —
    consistent with this being a run-to-completion dev tool, not a persistent
    service (see ticket #9/#10 scope: no durable/crash-recoverable session state).
    """

    topic: str
    exchanges: list[tuple[str, str]] = field(default_factory=list)
    pending_question: str | None = None
    awaiting_continuation: bool = False


MAX_TRANSCRIPT_CHARS = 6000
_TRUNCATION_NOTICE = "[earlier rounds omitted for length]"


def _render_transcript(session: _DebateSession) -> str:
    """Renders topic + exchange history, bounded to MAX_TRANSCRIPT_CHARS.

    Ticket #11: transcript grows every round (it's replayed into every model
    call in full), so without a bound it can eventually blow past the model's
    context. When it would exceed the bound, the OLDEST rounds are dropped
    first — the topic and the most recent rounds are always kept verbatim,
    since those matter most for deciding the next turn.

    Known limitation: this bounds exchange growth, not the topic itself — if
    the topic text alone already exceeds MAX_TRANSCRIPT_CHARS, the returned
    text can exceed the cap (with zero exchange rounds kept). This tool
    controls what topics get passed in, so an oversized topic isn't expected
    in practice; truncating the topic itself would need its own policy this
    ticket doesn't attempt.
    """
    topic_line = f"Topic: {session.topic}"
    exchange_lines: list[str] = []
    for i, (question, answer) in enumerate(session.exchanges, start=1):
        exchange_lines.append(f"Round {i} - You asked Claude: {question}")
        exchange_lines.append(f"Round {i} - Claude answered: {answer}")

    full = "\n".join([topic_line, *exchange_lines])
    if len(full) <= MAX_TRANSCRIPT_CHARS:
        return full

    budget = MAX_TRANSCRIPT_CHARS - len(topic_line) - len(_TRUNCATION_NOTICE) - 2
    kept: list[str] = []
    i = len(exchange_lines)
    while i > 0 and budget > 0:
        pair = exchange_lines[i - 2 : i]
        pair_text = "\n".join(pair)
        if len(pair_text) + 1 > budget:
            break
        kept = pair + kept
        budget -= len(pair_text) + 1
        i -= 2

    return "\n".join([topic_line, _TRUNCATION_NOTICE, *kept])


def _text_message(updater: TaskUpdater, text: str):
    return updater.new_agent_message([Part(root=TextPart(text=text))])


class GeminiAgentExecutor(AgentExecutor):
    """Answers incoming A2A messages using the OpenRouter-hosted Gemini model.

    When Message.metadata carries a truthy DEBATE_MODE_KEY, the executor instead
    runs a structured debate turn (see gemini_node.debate): the model's decision
    resolves to either `ask_claude` (task pauses at input-required, carrying the
    question) or `final` (task completes normally). Debate mode is opt-in — a
    call without it behaves exactly as before.
    """

    def __init__(self):
        self.agent = OpenRouterAgent()
        self._debate_sessions: dict[str, _DebateSession] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        metadata = context.message.metadata or {} if context.message else {}
        is_continuation = context.current_task is not None and context.task_id in self._debate_sessions

        # Strict `is True` (not truthiness): metadata is caller-controlled input,
        # and a stray "false"/1/[] should not silently opt a call into debate
        # mode. Continuation is also detected from our own session bookkeeping,
        # not solely from the flag being resent — a continuation call that
        # forgot to resend debate_mode=True must still be recognized as one,
        # not misrouted into one-shot handling.
        if metadata.get(DEBATE_MODE_KEY) is not True and not is_continuation:
            await self._execute_one_shot(context, event_queue, metadata)
            return

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        if is_continuation:
            await self._continue_debate(context, updater, metadata)
        else:
            await self._start_debate(context, updater, metadata)

    async def _execute_one_shot(self, context: RequestContext, event_queue: EventQueue, metadata: dict) -> None:
        query = context.get_user_input()
        try:
            answer = await self.agent.ask(query, model=metadata.get("model"), system=metadata.get("system"))
        except Exception as exc:
            raise ServerError(error=InternalError(message=str(exc))) from exc

        artifact = new_text_artifact(name=f"gemini_{context.task_id}", text=answer)
        await event_queue.enqueue_event(
            completed_task(
                context.task_id,
                context.context_id,
                [artifact],
                [context.message],
            )
        )

    async def _start_debate(self, context: RequestContext, updater: TaskUpdater, metadata: dict) -> None:
        topic = context.get_user_input()
        session = _DebateSession(topic=topic)
        self._debate_sessions[context.task_id] = session

        await updater.start_work()
        await self._advance_debate(context.task_id, session, updater, model=metadata.get("model"))

    async def _continue_debate(self, context: RequestContext, updater: TaskUpdater, metadata: dict) -> None:
        session = self._debate_sessions.get(context.task_id)
        if session is None or not session.awaiting_continuation:
            # Framework-level checks (in DefaultRequestHandler) already reject
            # continuations against an unknown or already-terminal task_id
            # before this executor ever runs. What lands here is the one edge
            # case that isn't: a second continuation for a task that's still
            # non-terminal (input-required) but that we've already resumed.
            #
            # This errors the RPC call itself rather than calling
            # updater.failed() -- that would write a terminal status onto the
            # *shared* persisted task, which could clobber a legitimately
            # in-flight winning continuation for the same task_id. Rejecting
            # here only fails this one redundant/racing request.
            raise ServerError(
                error=InternalError(
                    message="Duplicate or unexpected continuation for this debate task — "
                    "it is not currently awaiting an answer."
                )
            )

        # In this single-process, single-event-loop server, claiming the flag
        # here (before any `await`) is atomic with respect to other coroutines
        # on the same loop — there's no `await` between the read above and this
        # write, so nothing else can interleave. This does NOT hold across
        # threads, multiple worker processes, or multiple server instances;
        # this tool assumes exactly one.
        session.awaiting_continuation = False
        answer_text = context.get_user_input()
        session.exchanges.append((session.pending_question, answer_text))
        session.pending_question = None

        await updater.start_work()
        await self._advance_debate(context.task_id, session, updater, model=metadata.get("model"))

    async def _advance_debate(
        self, task_id: str, session: _DebateSession, updater: TaskUpdater, *, model: str | None
    ) -> None:
        transcript = _render_transcript(session)
        try:
            decision = await self.agent.debate_turn(transcript, model=model)
        except MalformedDebateTurn as exc:
            await updater.failed(_text_message(updater, f"Malformed debate turn: {exc}"))
            self._debate_sessions.pop(task_id, None)
            return
        except Exception as exc:
            # Bounded: an arbitrary exception (e.g. a huge HTTP error body)
            # must not turn into an unbounded diagnostic message.
            diagnostic = str(exc)[:500]
            await updater.failed(_text_message(updater, f"Debate turn failed: {diagnostic}"))
            self._debate_sessions.pop(task_id, None)
            return

        if decision.action == "ask_claude":
            session.pending_question = decision.question
            session.awaiting_continuation = True
            # final=True: input-required isn't in the SDK's "interruptable" set
            # (only auth-required is), so the event consumer doesn't know to
            # stop waiting once this executor's coroutine returns unless this
            # event is explicitly marked final — otherwise it blocks until the
            # now-closed queue raises QueueEmpty instead of returning cleanly.
            await updater.requires_input(_text_message(updater, decision.question), final=True)
        else:
            await updater.add_artifact([Part(root=TextPart(text=decision.answer))], name=f"gemini_{task_id}")
            await updater.complete()
            self._debate_sessions.pop(task_id, None)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=InternalError(message="cancel is not supported"))
