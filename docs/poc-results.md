# A2A POC Stability Check Results

**Historical / obsolete as of Ticket #13 (2026-07-28):** the `RELAY_PREFIX`
mechanism and `scripts/stability_check.py` that generated these results below
have both been removed and are no longer reproducible — superseded by the
"debate mode" multi-round coordinator (`common/debate_coordinator.py`). Kept
here only as a dated record that bidirectional A2A communication was proven
to work on this machine before the relay mechanism was retired.

Run at: 2026-07-27T06:16:22.578980+00:00
Rounds per direction: 4

## claude->gemini: 4/4 succeeded

| Round | Success | Elapsed (s) | Answer / Error |
|---|---|---|---|
| 1 | True | 1.93 | pong1 |
| 2 | True | 2.06 | pong2 |
| 3 | True | 1.15 | pong3 |
| 4 | True | 1.63 | pong4 |

## gemini->claude: 4/4 succeeded

| Round | Success | Elapsed (s) | Answer / Error |
|---|---|---|---|
| 1 | True | 5.74 | pong1 |
| 2 | True | 6.74 | pong2 |
| 3 | True | 4.75 | pong3 |
| 4 | True | 4.55 | pong4 |

## Conclusion

All rounds succeeded in both directions. The A2A protocol reliably carries requests and responses between the Claude Agent SDK node and the OpenRouter (Gemini) node on this machine. Stable enough to consider replacing `ask.ps1`.