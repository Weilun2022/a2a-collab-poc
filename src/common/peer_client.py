import uuid

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.client.errors import A2AClientError
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
)


class PeerCallError(RuntimeError):
    """Raised when a peer A2A node cannot be reached or returns a non-success response."""


def _extract_answer_text(task: Task) -> str:
    chunks: list[str] = []
    for artifact in task.artifacts or []:
        for part in artifact.parts:
            root = part.root
            if hasattr(root, "text"):
                chunks.append(root.text)
    return "".join(chunks)


async def ask_peer(base_url: str, question: str) -> str:
    """Sends `question` to the A2A peer at `base_url` and returns its text answer."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as httpx_client:
        try:
            card = await A2ACardResolver(base_url=base_url, httpx_client=httpx_client).get_agent_card()
        except (httpx.HTTPError, A2AClientError) as exc:
            raise PeerCallError(f"Could not reach peer agent card at {base_url}: {exc}") from exc

        client = A2AClient(httpx_client=httpx_client, agent_card=card)
        message_id = str(uuid.uuid4())
        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": question}],
                "messageId": message_id,
                "contextId": str(uuid.uuid4()),
            },
        }
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

        return _extract_answer_text(result)
