import ssl

import httpx
import truststore

from common.config import GEMINI_MODEL, load_openrouter_api_key

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterAgent:
    """Answers a free-text question using an OpenRouter-hosted model."""

    def __init__(self, model: str = GEMINI_MODEL):
        self.model = model
        self._api_key = load_openrouter_api_key()

    async def ask(self, question: str) -> str:
        # Uses the Windows system certificate store (via truststore) instead of
        # certifi's bundle, because this machine's corporate network terminates
        # TLS with a self-signed root CA that only the OS trust store knows about.
        ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        async with httpx.AsyncClient(timeout=60, verify=ssl_context) as client:
            response = await client.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": question}],
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
