from pathlib import Path

import openrouter_node.debate as debate_module
import common.debate_coordinator as debate_coordinator_module
from openrouter_node.debate import DEBATE_SYSTEM_PROMPT

# Ticket #20 is scoped to one-shot mode only. These tests guard against the
# new roles registry accidentally getting wired into debate mode later,
# which would be an undocumented behavior change out of scope for that ticket.


def test_debate_module_source_does_not_reference_roles_registry():
    source = Path(debate_module.__file__).read_text(encoding="utf-8")
    assert "roles" not in source


def test_debate_coordinator_source_does_not_reference_roles_registry():
    source = Path(debate_coordinator_module.__file__).read_text(encoding="utf-8")
    assert "roles" not in source


def test_debate_system_prompt_unchanged():
    assert DEBATE_SYSTEM_PROMPT == (
        "You are participating in a structured multi-turn design debate with "
        "another AI (Claude). Respond with ONLY a single JSON object — no prose "
        "before or after it, no markdown code fences. The JSON must have exactly "
        "one of these two shapes:\n"
        '{"action": "ask_claude", "question": "<your follow-up question for Claude>"}\n'
        '{"action": "final", "answer": "<your final answer, once you have no more questions>"}\n'
        "Do not include anything in your response other than this one JSON object."
    )
