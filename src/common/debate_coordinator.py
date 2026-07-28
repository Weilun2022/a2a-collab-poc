"""Minimal single-round debate coordinator (Ticket #10).

Runs exactly one Gemini-pause -> Claude-answer -> Gemini-resume round. Fixed
one-round shape is deliberate: it cannot loop unboundedly by construction.
Multi-round policy, transcript growth limits, round/time caps, and
forced-final handling are ticket #11's scope, not this one.
"""

from dataclasses import dataclass, field

from a2a.types import TaskState

from common.config import CLAUDE_AGENT_URL, DEBATE_MODE_KEY, GEMINI_AGENT_URL
from common.node_process import start_node, stop_node
from common.peer_client import PeerCallError, ask_peer_task
from gemini_node.debate import MAX_RAW_RESPONSE_CHARS

_ANSWER_TRUNCATION_NOTE = "...[truncated]"


@dataclass
class DebateResult:
    """Structured outcome of a debate session -- always check `outcome` first.

    `transcript` entries are `(question, answer)`; `answer` is `None` when the
    question was asked but no answer was obtained (e.g. Claude's call failed).
    """

    topic: str
    final_answer: str | None
    transcript: list[tuple[str, str | None]] = field(default_factory=list)
    outcome: str = "error"  # "converged" or "error"
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


async def run_debate_session(topic: str, *, model: str | None = None) -> DebateResult:
    """Runs one debate round on `topic`, starting and tearing down both nodes.

    Both node subprocesses are started for the session and are guaranteed to
    be terminated on every path -- normal completion or any error -- via
    try/finally covering both.
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
            return await _run_round(topic, model=model)
        finally:
            if claude_process is not None:
                stop_node(claude_process)
    finally:
        stop_node(gemini_process)


async def _run_round(topic: str, *, model: str | None) -> DebateResult:
    metadata: dict = {DEBATE_MODE_KEY: True}
    if model:
        metadata["model"] = model

    try:
        start = await ask_peer_task(GEMINI_AGENT_URL, topic, metadata=metadata)
    except PeerCallError as exc:
        return _failed(topic, str(exc))

    if start.state == TaskState.completed:
        return _converged(topic, start.answer_text, transcript=[])

    if start.state != TaskState.input_required:
        return _failed(topic, f"Gemini returned unexpected state {start.state!r} starting the debate.")

    question = start.answer_text
    try:
        claude_result = await ask_peer_task(CLAUDE_AGENT_URL, _claude_prompt(topic, question))
    except PeerCallError as exc:
        return _failed(topic, str(exc), transcript=[(question, None)])

    if claude_result.state != TaskState.completed:
        return _failed(
            topic,
            f"Claude node returned unexpected state {claude_result.state!r} answering the follow-up.",
            transcript=[(question, claude_result.answer_text)],
        )

    transcript: list[tuple[str, str | None]] = [(question, claude_result.answer_text)]
    try:
        resume = await ask_peer_task(
            GEMINI_AGENT_URL,
            _bounded(claude_result.answer_text),
            context_id=start.context_id,
            task_id=start.task_id,
            metadata=metadata,
        )
    except PeerCallError as exc:
        return _failed(topic, str(exc), transcript=transcript)

    if resume.state == TaskState.completed:
        return _converged(topic, resume.answer_text, transcript=transcript)

    return _failed(
        topic,
        f"Gemini did not converge after one round (ended in state {resume.state!r}); "
        "multi-round continuation is out of scope for this minimal coordinator (see ticket #11).",
        transcript=transcript,
    )
