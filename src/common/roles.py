"""Named system-prompt roles for the one-shot A2A CLI (ask_openrouter.py).

Debate mode is explicitly out of scope: its own structured-JSON system
prompt (openrouter_node.debate.DEBATE_SYSTEM_PROMPT) lives independently and
must stay unreferenced by this module -- see tests/test_role_registry_isolation.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Role:
    """A named persona for the one-shot CLI's --role flag.

    No `name` field: the ROLES dict key is the sole identifier, so there's no
    key/field pair that can drift out of sync. No `output_contract` field:
    nothing in this tool enforces or consumes an output shape today, so a
    field for it would be unused metadata -- add one if/when a real
    enforcement mechanism exists.
    """

    responsibilities: str
    system_prompt: str


# Migrated verbatim from ask_openrouter.py's former DEFAULT_ADVERSARIAL_SYSTEM
# constant -- see tests/test_roles.py for a regression check against the
# exact text, not just "a" role being present.
_ADVERSARIAL_REVIEWER_SYSTEM = (
    "Act as a skeptical senior engineer reviewing a colleague's plan. "
    "Do not simply state whether you agree. Actively look for holes in the "
    "approach and list concrete edge cases the author may have missed "
    "(concurrency, failure recovery, data growth, backwards compatibility, "
    "security, etc.). If you genuinely see no issues after actively looking, "
    "say so explicitly and explain what you checked — but default to finding "
    "problems, not to agreeing. "
    "Be concise: no rhetorical opening line, no restating how severe the "
    "problem is before listing it, no filler sentences. State each issue in "
    "one line unless it genuinely needs more. Merge points that share the "
    "same root cause instead of listing them separately."
)

_IMPLEMENTATION_ADVISOR_SYSTEM = (
    "You are pairing with another engineer on their codebase. For every "
    "suggestion, include a concrete code example grounded only in the "
    "context actually provided in this prompt -- if that context is "
    "insufficient to write real code, say exactly what's missing instead of "
    "inventing unstated files, APIs, or dependencies. This is a suggestion "
    "for the other engineer to review before use, not verified or executed "
    "code. Be concise: no filler, no restating the problem before showing "
    "code."
)

ROLES: dict[str, Role] = {
    "adversarial_reviewer": Role(
        responsibilities=(
            "Pressure-test a plan or decision -- find holes and edge cases "
            "instead of agreeing with it."
        ),
        system_prompt=_ADVERSARIAL_REVIEWER_SYSTEM,
    ),
    "implementation_advisor": Role(
        responsibilities=(
            "Given a task and pasted project context, propose an approach "
            "grounded in a concrete, reviewable code example."
        ),
        system_prompt=_IMPLEMENTATION_ADVISOR_SYSTEM,
    ),
}

DEFAULT_ROLE = "adversarial_reviewer"


class RoleResolutionError(ValueError):
    """Raised when --system/--role can't be resolved to a single system prompt.

    A dedicated type (mirroring InvalidModelId/PeerCallError elsewhere in this
    codebase) so callers can catch resolution failures specifically instead of
    a bare ValueError, which could accidentally swallow an unrelated one.
    """


def resolve_system_prompt(system: str | None, role: str | None) -> str:
    """Resolves the final system prompt for a one-shot call.

    Precedence is mutual exclusivity, not silent override: passing both
    `system` and `role` raises, so a stale caller passing both doesn't
    silently get one of them ignored. Passing neither resolves DEFAULT_ROLE,
    preserving today's default behavior exactly. Callers must call this
    before starting any node subprocess or making any network call, so bad
    input fails fast and cheaply.
    """
    if system is not None and role is not None:
        raise RoleResolutionError("--system and --role are mutually exclusive; pass at most one.")
    if system is not None:
        return system
    role_key = role if role is not None else DEFAULT_ROLE
    try:
        return ROLES[role_key].system_prompt
    except KeyError:
        valid = ", ".join(sorted(ROLES))
        raise RoleResolutionError(f"unknown role {role_key!r}; valid roles: {valid}") from None
