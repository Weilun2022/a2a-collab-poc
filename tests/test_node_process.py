"""Unit tests for node_process.py's teardown logic (Ticket #12).

These use a mock process object rather than a real subprocess: on Windows,
Popen.terminate() calls TerminateProcess(), which (unlike POSIX SIGTERM)
can't be caught/ignored by the child -- so there's no reliable way to make a
*real* process "ignore" a graceful terminate on this platform. What's under
test is stop_node's own retry/fallback logic given a process that doesn't
exit promptly, not OS-level signal semantics.
"""

import subprocess
from unittest.mock import MagicMock

import pytest

from common.node_process import safe_stop_node, stop_node


def _mock_process(*, wait_side_effect) -> MagicMock:
    process = MagicMock()
    process.poll.return_value = None  # still running
    process.wait.side_effect = wait_side_effect
    return process


def test_stop_node_terminates_without_killing_if_it_exits_promptly():
    process = _mock_process(wait_side_effect=[None])

    stop_node(process)

    process.terminate.assert_called_once()
    process.kill.assert_not_called()


def test_stop_node_falls_back_to_kill_when_terminate_is_ignored():
    # First wait() times out (simulating a process that didn't react to
    # terminate()); second wait() (after kill()) succeeds.
    process = _mock_process(wait_side_effect=[subprocess.TimeoutExpired(cmd="x", timeout=5), None])

    stop_node(process)

    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_stop_node_already_exited_is_a_noop():
    process = MagicMock()
    process.poll.return_value = 0  # already exited

    stop_node(process)

    process.terminate.assert_not_called()
    process.wait.assert_not_called()


def test_safe_stop_node_swallows_errors():
    process = MagicMock()
    process.poll.return_value = None
    process.terminate.side_effect = OSError("simulated failure terminating process")

    safe_stop_node(process)  # must not raise


def test_safe_stop_node_does_not_mask_a_successful_return():
    """Ticket #12 review: the "don't mask" guarantee applies whether the
    protected block raised OR returned normally -- a cleanup failure in a
    `finally` can't retroactively break a successful result either."""
    process = MagicMock()
    process.poll.return_value = None
    process.terminate.side_effect = OSError("cleanup itself failed")

    def protected_call():
        try:
            return "the real result"
        finally:
            safe_stop_node(process)

    assert protected_call() == "the real result"


async def test_safe_stop_node_does_not_mask_a_primary_error():
    """Ticket #12: cleanup failing while handling an earlier error must not
    replace that error -- the caller decides what to do with the primary
    failure; safe_stop_node just must not itself raise into the mix."""
    process = MagicMock()
    process.poll.return_value = None
    process.terminate.side_effect = OSError("cleanup itself failed")

    primary_error = ValueError("the real problem")
    try:
        try:
            raise primary_error
        finally:
            safe_stop_node(process)
    except ValueError as caught:
        assert caught is primary_error
    else:
        pytest.fail("expected the primary ValueError to propagate")
