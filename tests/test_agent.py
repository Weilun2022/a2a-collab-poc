import pytest

from openrouter_node.agent import InvalidModelId, _validate_model_id


def test_accepts_well_formed_model_id():
    _validate_model_id("openai/gpt-5.6-luna")


def test_rejects_empty_model_id():
    with pytest.raises(InvalidModelId):
        _validate_model_id("")


def test_rejects_whitespace_only_model_id():
    with pytest.raises(InvalidModelId):
        _validate_model_id("   ")


def test_rejects_model_id_without_provider_prefix():
    with pytest.raises(InvalidModelId):
        _validate_model_id("gpt-5.6-luna")
