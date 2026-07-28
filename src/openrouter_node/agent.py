import ssl

import httpx
import truststore

from common.config import OPENROUTER_MODEL, load_openrouter_api_key
from openrouter_node.debate import DEBATE_SYSTEM_PROMPT, DebateDecision, parse_debate_decision

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterAgent:
    """Answers a free-text question using an OpenRouter-hosted model."""

    def __init__(self, model: str = OPENROUTER_MODEL):
        self.model = model
        self._api_key = load_openrouter_api_key()

    async def ask(self, question: str, *, model: str | None = None, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})

        # Uses the Windows system certificate store (via truststore) instead of
        # certifi's bundle, because this machine's corporate network terminates
        # TLS with a self-signed root CA that only the OS trust store knows about.
        ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        async with httpx.AsyncClient(timeout=60, verify=ssl_context) as client:
            response = await client.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": model or self.model,
                    "messages": messages,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def debate_turn(self, transcript: str, *, model: str | None = None) -> DebateDecision:
        """Gets one structured ask_claude/final decision for a debate-mode turn.

        `transcript` is the plain-text rendering of everything discussed so far
        (topic plus prior question/answer rounds) — building and growing that
        text is the caller's responsibility, this just gets one decision for it.
        Raises `MalformedDebateTurn` (from `parse_debate_decision`) if the model's
        raw output doesn't resolve to a valid decision.
        """
        raw = await self.ask(transcript, model=model, system=DEBATE_SYSTEM_PROMPT)
        return parse_debate_decision(raw)
