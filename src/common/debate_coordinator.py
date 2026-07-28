"""Multi-round debate coordinator (Tickets #10, #11, #12).

Runs a Gemini-pause -> Claude-answer -> Gemini-resume loop, up to a hard
round/time cap, with a forced-final turn when the cap is hit. The session-wide
deadline bounds every individual call (not just the loop-top check), the
forced-final call has its own dedicated timeout, and cleanup never masks a
primary error.
"""

import time
from dataclasses import dataclass, field

from a2a.types import TaskState

from common.config import CLAUDE_AGENT_URL, DEBATE_MODE_KEY, GEMINI_AGENT_URL
from common.node_process import safe_stop_node, start_node
from common.peer_client import PeerCallError, ask_peer_task
from gemini_node.debate import MAX_RAW_RESPONSE_CHARS

_ANSWER_TRUNCATION_NOTE = "...[truncated]"
_ERROR_TRUNCATION_CHARS = 500

MAX_CLAUDE_FOLLOWUPS = 3
# Counts every outbound call to either node (Gemini decisions AND Claude
# answers) -- a full round costs 2. Checked once per loop iteration (before
# starting a round), not before each individual call within a round, so a
# round already in progress can land 1 call over this nominal cap before the
# next check fires.
MAX_TOTAL_MODEL_CALLS = 6
SESSION_TIME_LIMIT_SECONDS = 300.0

# Ceiling for a single ask_peer_task call, further shrunk to whatever time
# remains before SESSION_TIME_LIMIT_SECONDS -- see `_call_timeout()`. This is
# what makes the session deadline bind on an in-flight call, not just at the
# top of the loop between calls.
PER_CALL_TIMEOUT_SECONDS = 180.0
# The forced-final call gets its own short, separate ceiling: by the time
# it's sent, the session is already meant to be wrapping up, so it shouldn't
# get the same generous budget as a normal round.
FORCED_FINAL_TIMEOUT_SECONDS = 30.0
_MIN_CALL_TIMEOUT_SECONDS = 1.0

FORCED_FINAL_PROMPT = (
    "You have reached the maximum number of exchanges allowed for this debate. "
    "You must respond now with exactly one JSON object of the 'final' shape: "
    '{"action": "final", "answer": "<your final answer, using everything discussed so far>"}. '
    "Do not ask another question — this is your last turn."
)


@dataclass
class DebateResult:
    """Structured outcome of a debate session -- always check `outcome` first.

    `outcome` is one of:
    - "converged": Gemini reached a `final` decision on its own, within budget.
    - "forced_final": a round/time limit was hit, Gemini was told to wrap up,
      and it complied -- there's an answer, but it was cut off, not organic.
    - "round_limit" / "time_limit": a limit was hit and Gemini did NOT comply
      with the forced-final turn (asked again, malformed, or otherwise) --
      no usable final_answer.
    - "error": an operational failure (network/protocol), not a debate outcome.

    `transcript` entries are `(question, answer)`; `answer` is `None` when the
    question was asked but no answer was obtained (e.g. Claude's call failed).
    """

    topic: str
    final_answer: str | None
    transcript: list[tuple[str, str | None]] = field(default_factory=list)
    outcome: str = "error"
    error: str | None = None


def _converged(topic: str, answer: str, transcript: list[tuple[str, str | None]]) -> DebateResult:
    return DebateResult(topic=topic, final_answer=answer, transcript=transcript, outcome="converged")


def _failed(
    topic: str, error: str, transcript: list[tuple[str, str | None]] | None = None
) -> DebateResult:
    return DebateResult(topic=topic, final_answer=None, transcript=transcript or [], outcome="error", error=error)


def _claude_prompt(topic: str, question: str) -> str:
    return (
        f"Design topic under discussion: {topic}\n\n"
        f"A reviewer has a follow-up question about it:\n{question}\n\n"
        "Please answer the question."
    )


def _bounded(text: str) -> str:
    if len(text) <= MAX_RAW_RESPONSE_CHARS:
        return text
    return text[:MAX_RAW_RESPONSE_CHARS] + _ANSWER_TRUNCATION_NOTE


def _bounded_error(exc: Exception) -> str:
    # PeerCallError messages can echo endpoint/request details from the
    # underlying transport -- bound them the same way debate.py bounds model
    # output, so a DebateResult.error never carries unbounded content.
    text = str(exc)
    if len(text) <= _ERROR_TRUNCATION_CHARS:
        return text
    return text[:_ERROR_TRUNCATION_CHARS] + _ANSWER_TRUNCATION_NOTE


def _call_timeout(deadline: float, ceiling: float) -> float:
    """Shrinks `ceiling` to whatever time remains before `deadline`.

    Floored at _MIN_CALL_TIMEOUT_SECONDS: once the deadline has already
    passed, callers are expected to check that themselves and treat it as
    time_exceeded (skip the call) rather than issuing a near-zero-timeout
    call, which would just fail immediately and look like a transport error.
    """
    remaining = deadline - time.monotonic()
    return max(_MIN_CALL_TIMEOUT_SECONDS, min(ceiling, remaining))


async def run_debate_session(topic: str, *, model: str | None = None) -> DebateResult:
    """Runs a debate on `topic`, starting and tearing down both nodes.

    Both node subprocesses are started for the session and are guaranteed to
    be terminated on every path -- normal completion or any error -- via
    try/finally covering both. Cleanup itself never raises (see
    `node_process.safe_stop_node`), so it can never mask whatever result or
    exception was already in flight.
    """
    return await _run_debate_session(topic, model=model, start_claude_node=True)


async def _run_debate_session(topic: str, *, model: str | None, start_claude_node: bool) -> DebateResult:
    """Internal entry point; `start_claude_node=False` is a test-only hook
    (see tests/test_debate_coordinator.py) to simulate the Claude node being
    unavailable, for exercising the error/cleanup path without needing to
    break the real node. Kept out of the public `run_debate_session` signature
    so production callers never see a test seam."""
    gemini_process = start_node("gemini_node", GEMINI_AGENT_URL + ".well-known/agent.json")
    try:
        claude_process = (
            start_node("claude_node", CLAUDE_AGENT_URL + ".well-known/agent.json")
            if start_claude_node
            else None
        )
        try:
            return await _run_debate(topic, model=model)
        finally:
            if claude_process is not None:
                safe_stop_node(claude_process)
    finally:
        safe_stop_node(gemini_process)


async def _run_debate(topic: str, *, model: str | None) -> DebateResult:
    metadata: dict = {DEBATE_MODE_KEY: True}
    if model:
        metadata["model"] = model

    deadline = time.monotonic() + SESSION_TIME_LIMIT_SECONDS
    transcript: list[tuple[str, str | None]] = []
    total_calls = 0
    claude_followups = 0

    try:
        current = await ask_peer_task(
            GEMINI_AGENT_URL, topic, metadata=metadata, timeout=_call_timeout(deadline, PER_CALL_TIMEOUT_SECONDS)
        )
    except PeerCallError as exc:
        return _failed(topic, _bounded_error(exc))
    total_calls += 1

    if current.state == TaskState.completed:
        return _converged(topic, current.answer_text, transcript)
    if current.state != TaskState.input_required:
        return _failed(topic, f"Gemini returned unexpected state {current.state!r} starting the debate.")

    forced_final_sent = False

    while True:
        time_exceeded = time.monotonic() >= deadline
        round_exceeded = claude_followups >= MAX_CLAUDE_FOLLOWUPS
        calls_exceeded = total_calls >= MAX_TOTAL_MODEL_CALLS

        if not forced_final_sent and (time_exceeded or round_exceeded or calls_exceeded):
            limit_kind = "time_limit" if time_exceeded else "round_limit"
            forced_final_sent = True
            try:
                current = await ask_peer_task(
                    GEMINI_AGENT_URL,
                    FORCED_FINAL_PROMPT,
                    context_id=current.context_id,
                    task_id=current.task_id,
                    metadata=metadata,
                    timeout=_call_timeout(deadline, FORCED_FINAL_TIMEOUT_SECONDS),
                )
            except PeerCallError as exc:
                return DebateResult(
                    topic=topic,
                    final_answer=None,
                    transcript=transcript,
                    outcome=limit_kind,
                    error=_bounded_error(exc),
                )
            total_calls += 1

            # A `completed` state alone isn't enough -- an empty/missing
            # answer is still noncompliance, not a usable final answer.
            if current.state == TaskState.completed and current.answer_text.strip():
                return DebateResult(
                    topic=topic, final_answer=current.answer_text, transcript=transcript, outcome="forced_final"
                )
            return DebateResult(
                topic=topic,
                final_answer=None,
                transcript=transcript,
                outcome=limit_kind,
                error=f"Gemini did not honor the forced-final turn (ended in state {current.state!r}).",
            )

        question = current.answer_text
        try:
            claude_result = await ask_peer_task(
                CLAUDE_AGENT_URL,
                _claude_prompt(topic, question),
                timeout=_call_timeout(deadline, PER_CALL_TIMEOUT_SECONDS),
            )
        except PeerCallError as exc:
            transcript.append((question, None))
            return _failed(topic, _bounded_error(exc), transcript)
        total_calls += 1

        if claude_result.state != TaskState.completed:
            transcript.append((question, claude_result.answer_text))
            return _failed(
                topic,
                f"Claude node returned unexpected state {claude_result.state!r} answering the follow-up.",
                transcript,
            )

        transcript.append((question, claude_result.answer_text))
        claude_followups += 1

        try:
            current = await ask_peer_task(
                GEMINI_AGENT_URL,
                _bounded(claude_result.answer_text),
                context_id=current.context_id,
                task_id=current.task_id,
                metadata=metadata,
                timeout=_call_timeout(deadline, PER_CALL_TIMEOUT_SECONDS),
            )
        except PeerCallError as exc:
            return _failed(topic, _bounded_error(exc), transcript)
        total_calls += 1

        if current.state == TaskState.completed:
            return _converged(topic, current.answer_text, transcript)
        if current.state != TaskState.input_required:
            return _failed(
                topic, f"Gemini returned unexpected state {current.state!r} resuming the debate.", transcript
            )
        # else: loop again -- limits are re-checked at the top before another round.
