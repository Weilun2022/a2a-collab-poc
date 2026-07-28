# a2a-collab-poc

A real [A2A (Agent2Agent) protocol](https://a2a-protocol.org/) proof-of-concept enabling two-way collaboration between a **Claude node** (wraps the Claude Agent SDK, using your existing Claude Code subscription login — no separate API key) and an **OpenRouter node** (wraps any OpenRouter-hosted model, e.g. `openai/gpt-5.6-luna` or `google/gemini-3.5-flash-lite`).

Two ways to use it:

1. **One-shot** (`ask_openrouter.py`/`ask_openrouter.ps1`) — ask a single question, get one answer, done. Drop-in replacement for a plain OpenRouter API call.
2. **Debate mode** (`common/debate_coordinator.py`) — a genuine multi-round, bidirectional back-and-forth: the OpenRouter node can pause and ask Claude a real follow-up question before giving its final answer, instead of answering in one shot.

## Install

```bash
pip install -r requirements.txt
```

Requires:
- **Claude Agent SDK auth**: the `claude_node` uses your existing Claude Code subscription login (`claude_agent_sdk.query()`) — no separate API key needed, but you must already be logged into Claude Code on this machine.
- **OpenRouter API key**: the `openrouter_node` reads it from `~/.claude/tools/openrouter/config.json` (a JSON file with an `"api_key"` field) — see `common/config.py`'s `load_openrouter_api_key()`.
- Both nodes run as local subprocesses on fixed loopback ports (`localhost:8081` for Claude, `localhost:8082` for the OpenRouter node, see `common/config.py`) — nothing is exposed off-machine, and only one debate/one-shot session is assumed at a time (no concurrent-session support).

## Usage: one-shot questions

```powershell
# PowerShell wrapper (Windows), lives outside this repo at:
#   ~/.claude/tools/a2a-ask/ask_openrouter.ps1
.\ask_openrouter.ps1 -Prompt "your question" [-Model "openai/gpt-5.6-luna"] [-System "system prompt"]
.\ask_openrouter.ps1 -PromptFile "C:\path\to\long-prompt.txt"   # use for multi-line/special-char prompts
```

```bash
# Or call the underlying Python script directly, from this repo:
python ask_openrouter.py "your question" [--model MODEL] [--system "system prompt"]
python ask_openrouter.py --prompt-file path/to/prompt.txt
```

- If `-System`/`--system` is omitted, a **default adversarial system prompt** is used automatically — it asks the model to find holes and edge cases in your plan rather than just agree with it. Pass your own `-System` to override this for non-review use cases.
- `-Prompt`/`--prompt` auto-rejects risky content (multi-line, special punctuation, >200 chars) and tells you to use `-PromptFile`/`--prompt-file` instead — this avoids PowerShell argument-quoting corruption on long/special-character prompts.
- This mode starts the OpenRouter node fresh, sends exactly one `message/send` call, gets the answer, and shuts the node back down. No memory between calls, no bidirectionality — see debate mode below for that.

## Usage: debate mode (multi-round, bidirectional)

There's no CLI wrapper for this yet (see `docs/agents/` and issue #7 — it's an internal mechanism, not an end-user command). Call it directly from Python — note that `common`/`openrouter_node`/`claude_node` live under `src/`, which isn't on `sys.path` by default, so either set `PYTHONPATH=src` or insert it yourself as shown:

```python
import asyncio
import sys
sys.path.insert(0, "src")  # or set PYTHONPATH=src instead

from common.debate_coordinator import run_debate_session

async def main():
    result = await run_debate_session(
        "I'm planning to use a single global variable for session state. Thoughts?",
        model="openai/gpt-5.6-luna",  # optional, this is also OPENROUTER_MODEL's current default
    )
    print(result.outcome)        # "converged" | "forced_final" | "round_limit" | "time_limit" | "error"
    print(result.final_answer)   # None unless outcome is "converged" or "forced_final"
    print(result.transcript)     # list[(question, answer)] for every ask_claude round that happened
    print(result.error)          # set when outcome is "error"/"round_limit"/"time_limit"

asyncio.run(main())
```

**How it works:** the OpenRouter node answers in a structured `{"action": "ask_claude", "question": ...}` / `{"action": "final", "answer": ...}` shape (never free prose). On `ask_claude`, its A2A task pauses at `input-required`; the coordinator relays the question to the Claude node (which has no memory between calls, so it's given the topic + running transcript every time), sends Claude's answer back as a continuation against the *same* OpenRouter-node task, and repeats. Hard caps prevent it running forever: 3 Claude follow-ups, 6 total model calls, a 5-minute wall-clock session deadline (`common/debate_coordinator.py`'s `MAX_CLAUDE_FOLLOWUPS`/`MAX_TOTAL_MODEL_CALLS`/`SESSION_TIME_LIMIT_SECONDS`). Hitting a cap sends the OpenRouter node one forced-final turn instead of abruptly cutting the session.

Both node subprocesses are started for the session and always torn down afterward (converged, hit a limit, or crashed) — this is a local, single-user, run-to-completion dev tool, not a persistent service; there's no crash recovery or durable session state.

See [docs/pocock-a2a-hybrid-workflow.md](docs/pocock-a2a-hybrid-workflow.md) for the broader workflow this tool is used in (Claude+GPT autonomously discussing/pressure-testing plans before a human reviews a final digest), and how to phrase prompts so the model actually pushes back instead of rubber-stamping.

## Architecture

```
src/
  claude_node/        A2A server wrapping Claude Agent SDK (localhost:8081)
  openrouter_node/    A2A server wrapping an OpenRouter model (localhost:8082)
                      openrouter_node/debate.py: structured ask_claude/final decision protocol
  common/
    peer_client.py        ask_peer() (one-shot, flattened text) / ask_peer_task() (full task envelope)
    debate_coordinator.py run_debate_session() -- the multi-round debate entry point
    node_process.py        start_node()/stop_node()/safe_stop_node() subprocess helpers
    config.py               ports, model defaults, OpenRouter API key loading
```

## Running tests

```bash
pytest
```

All tests run real node subprocesses end-to-end (no mocking of the LLM calls or the A2A/HTTP layer) except where explicitly noted otherwise in a test's docstring (a few edge cases — a real model refusing a direct instruction, or a process ignoring a graceful terminate on Windows — aren't reliably forceable from real components on demand, so those specific tests use a stub/mock and say so).

## History

This started as a POC to validate whether the A2A protocol could reliably replace a plain OpenRouter API call (`ask.ps1`) for one-shot use — see [docs/poc-results.md](docs/poc-results.md) (now historical/superseded). It was later extended with the multi-round debate mode described above — see GitHub issue #7 and tickets #8–#13 for the full design history.
