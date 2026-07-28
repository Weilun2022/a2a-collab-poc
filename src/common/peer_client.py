import uuid
from dataclasses import dataclass

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.client.errors import A2AClientError
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TaskState,
)


class PeerCallError(RuntimeError):
    """Raised when a peer A2A node cannot be reached or returns a non-success response."""


@dataclass(frozen=True)
class PeerTaskResult:
    """Full task envelope from a peer call — status/IDs, not just flattened answer text.

    `answer_text`'s meaning depends on `state` — always check `state` first:
    the final answer when `completed`, a follow-up question when
    `input_required`, or a diagnostic message when `failed`/`canceled`/
    `rejected`. Never assume it's a real answer without checking `state`.
    """

    task_id: str
    context_id: str
    state: TaskState
    answer_text: str
    raw_task: Task


def _extract_answer_text(task: Task) -> str:
    chunks: list[str] = []
    for artifact in task.artifacts or []:
        for part in artifact.parts:
            root = part.root
            if hasattr(root, "text"):
                chunks.append(root.text)
    if chunks:
        return "".join(chunks)

    # Debate mode's input-required pause has no artifact (nothing "final" has
    # been produced yet) -- its question text lives on the status message
    # instead. Existing one-shot completed-task callers are unaffected: they
    # always populate artifacts, so this fallback never triggers for them.
    status_message = task.status.message
    if status_message is not None:
        for part in status_message.parts:
            root = part.root
            if hasattr(root, "text"):
                chunks.append(root.text)
    return "".join(chunks)


async def _send_and_get_task(
    base_url: str,
    question: str,
    *,
    metadata: dict | None,
    context_id: str,
    task_id: str | None = None,
    timeout: float,
) -> Task:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as httpx_client:
        try:
            card = await A2ACardResolver(base_url=base_url, httpx_client=httpx_client).get_agent_card()
        except (httpx.HTTPError, A2AClientError) as exc:
            raise PeerCallError(f"Could not reach peer agent card at {base_url}: {exc}") from exc

        client = A2AClient(httpx_client=httpx_client, agent_card=card)
        message_id = str(uuid.uuid4())
        message_payload = {
            "role": "user",
            "parts": [{"type": "text", "text": question}],
            "messageId": message_id,
            "contextId": context_id,
            "metadata": metadata or {},
        }
        if task_id is not None:
            message_payload["taskId"] = task_id
        payload = {"message": message_payload}
        request = SendMessageRequest(id=message_id, params=MessageSendParams.model_validate(payload))

        try:
            response = await client.send_message(request)
        except (httpx.HTTPError, A2AClientError) as exc:
            raise PeerCallError(f"Call to peer at {base_url} failed: {exc}") from exc

        if not isinstance(response.root, SendMessageSuccessResponse):
            raise PeerCallError(f"Peer at {base_url} returned a non-success response: {response.root}")

        result = response.root.result
        if not isinstance(result, Task):
            raise PeerCallError(f"Peer at {base_url} returned a non-task result: {result}")

        return result


async def ask_peer(base_url: str, question: str, *, metadata: dict | None = None, timeout: float = 180) -> str:
    """Sends `question` to the A2A peer at `base_url` and returns its text answer.

    `metadata` rides alongside the message unchanged (e.g. a model/system override
    the receiving node's executor may choose to honor) — it's not part of the A2A
    spec's message text, just Message.metadata passthrough.
    """
    task = await _send_and_get_task(
        base_url, question, metadata=metadata, context_id=str(uuid.uuid4()), timeout=timeout
    )
    return _extract_answer_text(task)


async def ask_peer_task(
    base_url: str,
    question: str,
    *,
    metadata: dict | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
    timeout: float = 180,
) -> PeerTaskResult:
    """Sends `question` to the A2A peer at `base_url` and returns the full task envelope.

    Unlike `ask_peer`, this exposes the task's state and server-issued `task_id`
    (needed to detect `input-required` and to resume a paused task later) instead
    of always flattening the response down to plain answer text.

    `context_id` is caller-supplied session correlation, distinct from `task_id`
    (normally server-issued and read back off the response) — passing your own
    `context_id` does not give you control over or continuity with the task
    itself, it only threads through as metadata.

    Pass `task_id` to continue an existing paused (`input-required`) task rather
    than starting a new one — the server rejects continuations against an unknown
    or already-terminal task_id with an error (surfaced here as `PeerCallError`).
    """
    resolved_context_id = context_id if context_id is not None else str(uuid.uuid4())
    task = await _send_and_get_task(
        base_url,
        question,
        metadata=metadata,
        context_id=resolved_context_id,
        task_id=task_id,
        timeout=timeout,
    )
    return PeerTaskResult(
        task_id=task.id,
        context_id=task.context_id,
        state=task.status.state,
        answer_text=_extract_answer_text(task),
        raw_task=task,
    )
