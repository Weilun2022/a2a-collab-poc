"""Structured turn protocol for the OpenRouter node's debate mode (Ticket #9).

Debate mode requires every model turn to resolve to one of two validated
shapes instead of free-text prose — this is what lets the executor drive
`input-required` pause/resume deterministically instead of guessing at intent
from natural language.
"""

import json
from dataclasses import dataclass

DEBATE_SYSTEM_PROMPT = (
    "You are participating in a structured multi-turn design debate with "
    "another AI (Claude). Respond with ONLY a single JSON object — no prose "
    "before or after it, no markdown code fences. The JSON must have exactly "
    "one of these two shapes:\n"
    '{"action": "ask_claude", "question": "<your follow-up question for Claude>"}\n'
    '{"action": "final", "answer": "<your final answer, once you have no more questions>"}\n'
    "Do not include anything in your response other than this one JSON object."
)


class MalformedDebateTurn(ValueError):
    """Raised when a model's debate-mode output doesn't parse into a valid decision.

    Callers must treat `str(exc)` as bounded/sanitized: it never contains more
    than a truncated prefix of the model's raw output, so a malformed response
    can't blow up a diagnostic message or a downstream log with unbounded or
    sensitive content.
    """


MAX_RAW_RESPONSE_CHARS = 4000
_TRUNCATION_PREVIEW_CHARS = 200


@dataclass(frozen=True)
class DebateDecision:
    action: str  # "ask_claude" or "final"
    question: str | None = None
    answer: str | None = None


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = text.strip("`")
    if text.startswith("json"):
        text = text[len("json") :]
    return text.strip()


def parse_debate_decision(raw_text: str) -> DebateDecision:
    """Parses and validates a model's raw debate-mode output.

    Raises `MalformedDebateTurn` for anything that doesn't cleanly resolve to
    a valid `ask_claude`/`final` decision — callers must treat this as a
    protocol violation, not attempt to guess at the model's intent.
    """
    if len(raw_text) > MAX_RAW_RESPONSE_CHARS:
        preview = raw_text[:_TRUNCATION_PREVIEW_CHARS]
        raise MalformedDebateTurn(
            f"response exceeds {MAX_RAW_RESPONSE_CHARS} chars "
            f"(got {len(raw_text)}); rejected before parsing, preview: {preview!r}..."
        )

    text = _strip_code_fence(raw_text.strip())

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:_TRUNCATION_PREVIEW_CHARS]
        raise MalformedDebateTurn(f"response is not valid JSON: {exc}; preview: {preview!r}") from exc

    if not isinstance(data, dict):
        raise MalformedDebateTurn("response JSON must be an object")

    action = data.get("action")
    if action == "ask_claude":
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise MalformedDebateTurn("'ask_claude' action requires a non-empty 'question' string")
        return DebateDecision(action="ask_claude", question=question)

    if action == "final":
        answer = data.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise MalformedDebateTurn("'final' action requires a non-empty 'answer' string")
        return DebateDecision(action="final", answer=answer)

    raise MalformedDebateTurn(f"unknown or missing action {action!r}, expected 'ask_claude' or 'final'")
