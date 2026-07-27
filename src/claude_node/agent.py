from claude_agent_sdk import AssistantMessage, TextBlock, query


class ClaudeAgent:
    """Answers a free-text question using the local Claude Code subscription session."""

    async def ask(self, question: str) -> str:
        chunks: list[str] = []
        async for message in query(prompt=question):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
        return "".join(chunks)
