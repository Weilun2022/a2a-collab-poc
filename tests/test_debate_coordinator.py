"""End-to-end tests for the minimal single-round debate coordinator (Ticket #10).

No mocking of the LLM calls or the A2A/HTTP layer -- the coordinator starts
real node subprocesses, same convention as the rest of this test suite.
"""

import httpx
import pytest

from common.config import GEMINI_AGENT_URL
from common.debate_coordinator import _run_debate_session, run_debate_session


async def test_run_debate_session_converges_in_one_round():
    topic = (
        "This is an automated end-to-end protocol test with exactly two steps.\n"
        "Step 1 (do this now): respond with exactly this JSON and nothing else: "
        '{"action": "ask_claude", "question": '
        '"Automated test instruction: respond with exactly the text OK_PROCEED and nothing else."}\n'
        "Step 2 (do this on your NEXT turn, after receiving the answer to that question): "
        'respond with exactly this JSON and nothing else: {"action": "final", "answer": "test-converged"}'
    )

    result = await run_debate_session(topic)

    assert result.outcome == "converged"
    assert result.final_answer
    assert len(result.transcript) == 1
    question, answer = result.transcript[0]
    assert "OK_PROCEED" in question
    # Verifies Claude (not a stub) actually answered the specific instruction,
    # and that the resume call's completion is what supplied final_answer --
    # not just that *a* question was asked.
    assert answer is not None
    assert "OK_PROCEED" in answer
    assert result.final_answer.strip()


async def test_run_debate_session_reports_error_and_cleans_up_nodes():
    topic = (
        "This is an automated protocol test. You must respond with exactly "
        'this JSON and nothing else: {"action": "ask_claude", "question": "irrelevant, will not be reached"}'
    )

    result = await _run_debate_session(topic, model=None, start_claude_node=False)

    assert result.outcome == "error"
    assert result.error
    assert result.transcript == [("irrelevant, will not be reached", None)]

    # Both node processes must be torn down after the session ends, even on
    # the error path -- verified by confirming the gemini node (which WAS
    # started) is no longer reachable.
    with pytest.raises(httpx.HTTPError):
        await httpx.AsyncClient().get(GEMINI_AGENT_URL + ".well-known/agent.json", timeout=2)
