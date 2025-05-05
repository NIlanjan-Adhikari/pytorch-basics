import pytest
from unittest.mock import MagicMock
from your_module import _post_process_address_field  # replace with actual module path

@pytest.fixture
def base_model_mock():
    mock = MagicMock()
    mock.splitter_instance.side_effect = lambda addr, _: {"street": "Main St", "zipcode": "12345", "full_address": addr}
    mock.best_match.return_value = "Best Match Address"
    return mock

def test_post_process_both_addresses(base_model_mock):
    doc_model_output = {
        "fields": {
            "registered_address": ["123 Main St, City, Zip"],
            "establishment_address": ["456 Side St, Other City, Zip"]
        },
        "model_id": "model-1",
        "version": "v1"
    }

    result = _post_process_address_field(doc_model_output, base_model_mock)

    reg = result["fields"]["registered_address"][0]
    est = result["fields"]["establishment_address"][0]

    assert reg["name"] == "Best Match Address"
    assert reg["best_match"] == "Best Match Address"
    assert reg["street"] == "Main St"
    assert reg["zipcode"] == "12345"

    assert est["name"] == "Best Match Address"
    assert est["best_match"] == "Best Match Address"
    assert est["street"] == "Main St"
    assert est["zipcode"] == "12345"

def test_post_process_missing_registered_address(base_model_mock):
    doc_model_output = {
        "fields": {
            "establishment_address": ["456 Side St, Other City, Zip"]
        },
        "model_id": "model-2",
        "version": "v2"
    }

    result = _post_process_address_field(doc_model_output, base_model_mock)

    assert "registered_address" not in result["fields"] or isinstance(result["fields"]["registered_address"], list)

def test_post_process_missing_establishment_address(base_model_mock):
    doc_model_output = {
        "fields": {
            "registered_address": ["123 Main St, City, Zip"]
        },
        "model_id": "model-3",
        "version": "v3"
    }

    result = _post_process_address_field(doc_model_output, base_model_mock)

    assert "establishment_address" not in result["fields"] or isinstance(result["fields"]["establishment_address"], list)

def test_post_process_no_addresses(base_model_mock):
    doc_model_output = {
        "fields": {},
        "model_id": "model-4",
        "version": "v4"
    }

    result = _post_process_address_field(doc_model_output, base_model_mock)

    assert result["fields"] == {}
