import json
from pathlib import Path

OPENROUTER_CONFIG_PATH = Path.home() / ".claude" / "tools" / "openrouter" / "config.json"

OPENROUTER_MODEL = "openai/gpt-5.6-luna"
CLAUDE_HOST = "localhost"
CLAUDE_PORT = 8081
OPENROUTER_HOST = "localhost"
OPENROUTER_PORT = 8082

CLAUDE_AGENT_URL = f"http://{CLAUDE_HOST}:{CLAUDE_PORT}/"
OPENROUTER_AGENT_URL = f"http://{OPENROUTER_HOST}:{OPENROUTER_PORT}/"

# Message.metadata key: truthy value opts a call into debate mode (structured
# ask_claude/final turns, input-required pause/resume). Absent/falsy preserves
# today's plain one-shot behavior unchanged.
DEBATE_MODE_KEY = "debate_mode"


def load_openrouter_api_key() -> str:
    config = json.loads(OPENROUTER_CONFIG_PATH.read_text(encoding="utf-8"))
    return config["api_key"]
