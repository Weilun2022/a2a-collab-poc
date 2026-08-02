import pytest

from common.roles import ROLES, DEFAULT_ROLE, resolve_system_prompt

# Copied verbatim from the pre-migration DEFAULT_ADVERSARIAL_SYSTEM constant in
# ask_openrouter.py, so this test catches any accidental wording drift during
# the move into the registry -- not just "some" role being present.
_EXPECTED_ADVERSARIAL_REVIEWER_TEXT = (
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


def test_default_role_is_adversarial_reviewer():
    assert DEFAULT_ROLE == "adversarial_reviewer"


def test_resolve_with_neither_arg_returns_adversarial_reviewer_text_verbatim():
    assert resolve_system_prompt(None, None) == _EXPECTED_ADVERSARIAL_REVIEWER_TEXT
    assert resolve_system_prompt(None, None) == ROLES["adversarial_reviewer"].system_prompt


def test_resolve_with_role_returns_that_roles_prompt():
    assert resolve_system_prompt(None, "implementation_advisor") == ROLES["implementation_advisor"].system_prompt


def test_resolve_with_raw_system_returns_it_unchanged():
    assert resolve_system_prompt("custom text", None) == "custom text"


def test_resolve_with_empty_string_system_is_still_an_explicit_override():
    assert resolve_system_prompt("", None) == ""


def test_resolve_rejects_both_system_and_role():
    with pytest.raises(ValueError):
        resolve_system_prompt("custom text", "implementation_advisor")


def test_resolve_rejects_unknown_role_lists_valid_keys():
    with pytest.raises(ValueError) as exc_info:
        resolve_system_prompt(None, "not_a_real_role")
    message = str(exc_info.value)
    assert "adversarial_reviewer" in message
    assert "implementation_advisor" in message


def test_implementation_advisor_role_registered():
    assert "implementation_advisor" in ROLES


def test_roles_have_no_name_or_output_contract_fields():
    role = ROLES["adversarial_reviewer"]
    assert not hasattr(role, "name")
    assert not hasattr(role, "output_contract")
    assert hasattr(role, "responsibilities")
    assert hasattr(role, "system_prompt")
