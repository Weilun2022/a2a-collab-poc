import json
from pathlib import Path

OPENROUTER_CONFIG_PATH = Path.home() / ".claude" / "tools" / "openrouter" / "config.json"

GEMINI_MODEL = "google/gemini-3.5-flash-lite"
CLAUDE_HOST = "localhost"
CLAUDE_PORT = 8081
GEMINI_HOST = "localhost"
GEMINI_PORT = 8082

CLAUDE_AGENT_URL = f"http://{CLAUDE_HOST}:{CLAUDE_PORT}/"
GEMINI_AGENT_URL = f"http://{GEMINI_HOST}:{GEMINI_PORT}/"


def load_openrouter_api_key() -> str:
    config = json.loads(OPENROUTER_CONFIG_PATH.read_text(encoding="utf-8"))
    return config["api_key"]
