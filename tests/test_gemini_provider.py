import asyncio
from types import SimpleNamespace

import pytest
from google.genai import errors, models, types
from google.genai._api_client import retry_args

from app import AnalysisResponse
from diagnose_gemini_config import schema_metrics
from provider import (
    GEMINI_CONFIG_VARIANTS,
    GeminiProvider,
    GeminiProviderError,
    POTENTIALLY_PROBLEMATIC_SCHEMA_KEYS,
    SUPPORTED_JSON_SCHEMA_KEYS,
    create_gemini_provider,
)


VALID_JSON = (
    '{"participants":[{"status":"stated","statement":"Студенты.",'
    '"evidence":["참가 대상: 대학생"]}],"exclusions":[],"actions":[],'
    '"documents":[],"deadlines":[],"unknowns":[],'
    '"organizer_question":"참가 대상이 맞는지 확인 부탁드립니다."}'
)


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeAsyncClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)
        self.closed = False

    async def aclose(self):
        self.closed = True


class FakeClient:
    def __init__(self, response=None, error=None):
        self.aio = FakeAsyncClient(response=response, error=error)
        self.closed = False

    def close(self):
        self.closed = True


def response_with_text(text, finish_reason=types.FinishReason.STOP):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        prompt_feedback=None,
    )


def make_provider(response=None, error=None):
    fake_client = FakeClient(response=response, error=error)
    return GeminiProvider(
        api_key="test-only-not-a-real-key",
        model="gemini-3.8-flash",
        client=fake_client,
    ), fake_client


def test_gemini_adapter_uses_selected_model_json_schema_and_single_input():
    async def run():
        response = response_with_text(VALID_JSON)
        provider, client = make_provider(response=response)
        result = await provider.analyze("참가 대상: 대학생")
        return result, client

    result, client = asyncio.run(run())
    assert result["participants"][0]["evidence"] == ["참가 대상: 대학생"]
    call = client.aio.models.calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].response_schema is None
    assert call["config"].model_fields_set == {
        "response_mime_type",
        "response_json_schema",
        "max_output_tokens",
        "automatic_continuation",
    }
    assert call["config"].response_json_schema["properties"].keys() == {
        "participants",
        "exclusions",
        "actions",
        "documents",
        "deadlines",
        "unknowns",
        "organizer_question",
    }
    assert set(call["config"].response_json_schema["required"]) == (
        call["config"].response_json_schema["properties"].keys()
    )
    assert call["config"].max_output_tokens == 2048
    assert call["config"].automatic_continuation is False
    assert "참가 대상: 대학생" in call["contents"]
    assert "tools" not in call["config"].model_fields_set
    assert len(client.aio.models.calls) == 1


def test_outgoing_schema_uses_only_documented_json_schema_fields():
    config = GeminiProvider.build_generation_config(AnalysisResponse)
    schema = config.response_json_schema
    assert isinstance(schema, dict)
    assert set(schema) <= SUPPORTED_JSON_SCHEMA_KEYS
    assert schema["type"] == "object"
    assert set(schema["required"]) == set(schema["properties"])

    def assert_supported(node):
        if isinstance(node, dict):
            assert set(node) <= SUPPORTED_JSON_SCHEMA_KEYS
            for key, value in node.items():
                if key in {"properties", "$defs"}:
                    for child in value.values():
                        assert_supported(child)
                elif key in {"items", "additionalProperties"} and isinstance(value, dict):
                    assert_supported(value)
                elif key in {"anyOf", "oneOf", "prefixItems"}:
                    for child in value:
                        assert_supported(child)

    assert_supported(schema)
    serialized = str(schema)
    assert "maxLength" not in serialized
    assert "max_length" not in serialized


def test_sdk_serializes_json_schema_without_dropping_required_fields():
    class OfflineApiClient:
        vertexai = False

    config = GeminiProvider.build_generation_config(AnalysisResponse)
    sdk_request = models._GenerateContentConfig_to_mldev(
        OfflineApiClient(),
        config,
    )
    assert set(sdk_request) == {
        "maxOutputTokens",
        "responseJsonSchema",
        "responseMimeType",
    }
    assert sdk_request["responseMimeType"] == "application/json"
    assert sdk_request["maxOutputTokens"] == 2048
    schema = sdk_request["responseJsonSchema"]
    assert set(schema["properties"]) == {
        "participants",
        "exclusions",
        "actions",
        "documents",
        "deadlines",
        "unknowns",
        "organizer_question",
    }
    assert set(schema["required"]) == set(schema["properties"])


def test_config_diagnostic_reports_only_schema_keywords_and_structure():
    config = GeminiProvider.build_generation_config(AnalysisResponse)
    property_count, max_depth, found_keys = schema_metrics(
        config.response_json_schema
    )
    assert property_count == 25
    assert max_depth == 5
    assert found_keys & POTENTIALLY_PROBLEMATIC_SCHEMA_KEYS


def test_variants_a_through_d_preserve_required_product_fields():
    expected_fields = {
        "participants",
        "exclusions",
        "actions",
        "documents",
        "deadlines",
        "unknowns",
        "organizer_question",
    }
    for variant in GEMINI_CONFIG_VARIANTS:
        config = GeminiProvider.build_generation_config(
            AnalysisResponse,
            variant=variant,
        )
        schema = config.response_json_schema
        assert set(schema["properties"]) == expected_fields
        assert set(schema["required"]) == expected_fields
        assert config.response_mime_type == "application/json"
        assert config.automatic_continuation is False
        assert ("max_output_tokens" in config.model_fields_set) == (
            variant in {"A", "C"}
        )


def test_simplified_variants_use_only_object_string_and_array_types():
    for variant in ("C", "D"):
        config = GeminiProvider.build_generation_config(
            AnalysisResponse,
            variant=variant,
        )
        property_count, max_depth, found_keys = schema_metrics(
            config.response_json_schema
        )
        schema = config.response_json_schema
        assert property_count >= 7
        assert max_depth >= 3
        assert not found_keys & POTENTIALLY_PROBLEMATIC_SCHEMA_KEYS

        def assert_simple_types(node):
            if not isinstance(node, dict):
                return
            if "type" in node:
                assert node["type"] in {"object", "string", "array"}
            for value in node.values():
                if isinstance(value, dict):
                    assert_simple_types(value)
                elif isinstance(value, list):
                    for item in value:
                        assert_simple_types(item)

        assert_simple_types(schema)


def test_default_configuration_remains_variant_a_unless_explicitly_selected(monkeypatch):
    monkeypatch.delenv("NTA_PROVIDER_API_KEY", raising=False)
    monkeypatch.delenv("NTA_PROVIDER_MODEL", raising=False)
    monkeypatch.delenv("NTA_GEMINI_CONFIG_VARIANT", raising=False)
    default_provider = create_gemini_provider()
    assert default_provider is None
    monkeypatch.setenv("NTA_PROVIDER_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("NTA_PROVIDER_MODEL", "gemini-3.8-flash")
    default_provider = create_gemini_provider()
    assert default_provider._config_variant == "A"
    monkeypatch.setenv("NTA_GEMINI_CONFIG_VARIANT", "C")
    selected_provider = create_gemini_provider()
    assert selected_provider._config_variant == "C"
    assert GeminiProvider.build_generation_config(
        AnalysisResponse
    ).model_fields_set == {
        "response_mime_type",
        "response_json_schema",
        "max_output_tokens",
        "automatic_continuation",
    }


def test_truncated_generation_is_rejected():
    async def run():
        provider, _ = make_provider(response=response_with_text(
            '{"participants":',
            finish_reason=types.FinishReason.MAX_TOKENS,
        ))
        with pytest.raises(GeminiProviderError, match="truncated"):
            await provider.analyze("notice")

    asyncio.run(run())


def test_empty_response_is_rejected():
    async def run():
        provider, _ = make_provider(response=response_with_text(""))
        with pytest.raises(GeminiProviderError, match="empty"):
            await provider.analyze("notice")

    asyncio.run(run())


def test_refusal_is_rejected():
    async def run():
        response = SimpleNamespace(
            text=None,
            candidates=[SimpleNamespace(finish_reason=types.FinishReason.SAFETY)],
            prompt_feedback=None,
        )
        provider, _ = make_provider(response=response)
        with pytest.raises(GeminiProviderError, match="refused"):
            await provider.analyze("notice")

    asyncio.run(run())


def test_blocked_prompt_without_candidates_is_rejected():
    async def run():
        response = SimpleNamespace(
            text=None,
            candidates=[],
            prompt_feedback=SimpleNamespace(block_reason="SAFETY"),
        )
        provider, _ = make_provider(response=response)
        with pytest.raises(GeminiProviderError, match="refused"):
            await provider.analyze("notice")

    asyncio.run(run())


def test_non_json_response_is_rejected():
    async def run():
        provider, _ = make_provider(response=response_with_text("not json"))
        with pytest.raises(GeminiProviderError, match="invalid_response"):
            await provider.analyze("notice")

    asyncio.run(run())


def test_provider_api_errors_are_classified_with_safe_diagnostics_only():
    async def run():
        cases = {
            (400, "INVALID_ARGUMENT", None): "invalid_parameters",
            (401, "UNAUTHENTICATED", "API_KEY_INVALID"): "invalid_key",
            (403, "PERMISSION_DENIED", "SERVICE_DISABLED"): "access_denied",
            (404, "NOT_FOUND", "MODEL_NOT_FOUND"): "model_unavailable",
            (429, "RESOURCE_EXHAUSTED", "QUOTA_EXCEEDED"): "quota",
            (504, "DEADLINE_EXCEEDED", None): "provider_timeout",
            (500, "INTERNAL", None): "unknown",
        }
        for (http_code, status, reason_code), expected in cases.items():
            details = {
                "error": {
                    "status": status,
                    "message": "sensitive provider message must not leak",
                    "details": (
                        [{"reason": reason_code}]
                        if reason_code
                        else []
                    ),
                }
            }
            api_error = errors.ClientError(
                http_code,
                details,
            )
            provider, _ = make_provider(error=api_error)
            with pytest.raises(GeminiProviderError) as raised:
                await provider.analyze("notice")
            assert raised.value.reason == expected
            assert raised.value.http_code == http_code
            assert raised.value.status == status
            assert raised.value.reason_code == reason_code
            assert "sensitive provider message" not in str(raised.value)

    asyncio.run(run())


def test_unrecognized_error_metadata_is_not_exposed():
    async def run():
        api_error = errors.ClientError(
            418,
            {
                "error": {
                    "status": "PRIVATE_DATA",
                    "details": [{"reason": "PRIVATE_REASON"}],
                }
            },
        )
        provider, _ = make_provider(error=api_error)
        with pytest.raises(GeminiProviderError) as raised:
            await provider.analyze("notice")
        assert raised.value.reason == "unknown"
        assert raised.value.http_code == 418
        assert raised.value.status is None
        assert raised.value.reason_code is None

    asyncio.run(run())


def test_unexpected_sdk_exception_is_classified_as_unknown():
    async def run():
        provider, _ = make_provider(error=RuntimeError("secret detail"))
        with pytest.raises(GeminiProviderError) as raised:
            await provider.analyze("notice")
        assert raised.value.reason == "unknown"
        assert "secret detail" not in str(raised.value)

    asyncio.run(run())


def test_provider_timeout_is_classified():
    async def run():
        provider, _ = make_provider(error=TimeoutError())
        with pytest.raises(GeminiProviderError, match="timeout"):
            await provider.analyze("notice")

    asyncio.run(run())


def test_client_configuration_uses_long_timeout_and_bounded_retries(monkeypatch):
    captured = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return FakeClient(response=response_with_text(VALID_JSON))

    monkeypatch.setattr("provider.genai.Client", fake_client)
    async def run():
        provider = GeminiProvider(
            api_key="test-only-not-a-real-key",
            model="gemini-3.8-flash",
        )
        await provider.analyze("notice")
        await provider.aclose()

    asyncio.run(run())
    http_options = captured["http_options"]
    assert http_options.timeout == 1_200_000
    retry_options = http_options.retry_options
    assert retry_options.attempts == 3
    assert retry_options.initial_delay == 1.0
    assert retry_options.max_delay == 30
    assert retry_options.http_status_codes == [408, 429, 500, 502, 503, 504]


def test_sdk_retry_policy_retries_only_configured_http_statuses():
    retry_options = types.HttpRetryOptions(
        attempts=3,
        initial_delay=1.0,
        max_delay=30,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    )
    should_retry = retry_args(retry_options)["retry"].predicate
    for http_code in (408, 429, 500, 502, 503, 504):
        assert should_retry(errors.ClientError(http_code, {"error": {}}))
    for http_code in (400, 401, 403, 404):
        assert not should_retry(errors.ClientError(http_code, {"error": {}}))


def test_provider_requires_both_environment_settings(monkeypatch):
    monkeypatch.delenv("NTA_PROVIDER_API_KEY", raising=False)
    monkeypatch.delenv("NTA_PROVIDER_MODEL", raising=False)
    assert create_gemini_provider() is None

    monkeypatch.setenv("NTA_PROVIDER_API_KEY", "test-only-not-a-real-key")
    assert create_gemini_provider() is None

    monkeypatch.delenv("NTA_PROVIDER_API_KEY")
    monkeypatch.setenv("NTA_PROVIDER_MODEL", "gemini-3.8-flash")
    assert create_gemini_provider() is None


def test_adapter_closes_injected_client():
    async def run():
        provider, client = make_provider(response=response_with_text(VALID_JSON))
        await provider.aclose()
        return client

    client = asyncio.run(run())
    assert client.aio.closed
    assert client.closed
