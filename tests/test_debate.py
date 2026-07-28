import pytest

from openrouter_node.debate import DebateDecision, MalformedDebateTurn, parse_debate_decision


def test_parses_ask_claude_decision():
    decision = parse_debate_decision('{"action": "ask_claude", "question": "What about concurrency?"}')
    assert decision == DebateDecision(action="ask_claude", question="What about concurrency?")


def test_parses_final_decision():
    decision = parse_debate_decision('{"action": "final", "answer": "Looks good."}')
    assert decision == DebateDecision(action="final", answer="Looks good.")


def test_tolerates_markdown_code_fence():
    decision = parse_debate_decision('```json\n{"action": "final", "answer": "ok"}\n```')
    assert decision == DebateDecision(action="final", answer="ok")


def test_rejects_non_json():
    with pytest.raises(MalformedDebateTurn):
        parse_debate_decision("sure, that sounds fine to me!")


def test_rejects_unknown_action():
    with pytest.raises(MalformedDebateTurn):
        parse_debate_decision('{"action": "maybe", "question": "huh?"}')


def test_rejects_ask_claude_without_question():
    with pytest.raises(MalformedDebateTurn):
        parse_debate_decision('{"action": "ask_claude"}')


def test_rejects_final_without_answer():
    with pytest.raises(MalformedDebateTurn):
        parse_debate_decision('{"action": "final", "answer": ""}')


def test_rejects_json_that_is_not_an_object():
    with pytest.raises(MalformedDebateTurn):
        parse_debate_decision('["action", "final"]')
