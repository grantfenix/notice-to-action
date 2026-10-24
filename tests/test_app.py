import asyncio
import logging
import os
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app as prototype
from provider import GeminiProviderError


NOTICE = "참가 대상: 대학생. 신청 기간: 10월 1일 ~ 10월 8일."
VALID_RESULT = {
    "participants": [
        {
            "status": "stated",
            "statement": "Участвовать могут студенты.",
            "evidence": ["참가 대상: 대학생"],
        }
    ],
    "exclusions": [],
    "actions": [],
    "documents": [],
    "deadlines": [
        {
            "status": "stated",
            "statement": "Период подачи указан.",
            "evidence": ["신청 기간: 10월 1일 ~ 10월 8일"],
        },
        {
            "status": "needs_clarification",
            "statement": "Год для указанного периода не приведён.",
            "evidence": ["신청 기간: 10월 1일 ~ 10월 8일"],
        },
    ],
    "unknowns": [
        {
            "status": "not_stated",
            "statement": "В предоставленном тексте не указано время окончания.",
            "evidence": [],
        }
    ],
    "organizer_question": "신청 마감 시간은 언제인가요?",
}


class StubProvider:
    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.result = VALID_RESULT if result is None else result
        self.error = error
        self.calls = 0

    async def analyze(self, notice: str):
        self.calls += 1
        if self.error:
            raise self.error
        if callable(self.result):
            return await self.result(notice)
        return self.result

    async def aclose(self):
        return None


@pytest.fixture
def client():
    prototype.limiter.reset()
    prototype.app.dependency_overrides.clear()
    prototype.app.dependency_overrides[prototype.get_provider] = lambda: None
    with TestClient(prototype.app) as test_client:
        yield test_client
    prototype.app.dependency_overrides.clear()
    prototype.limiter.reset()


def inject_provider(provider: StubProvider) -> None:
    prototype.app.dependency_overrides[prototype.get_provider] = lambda: provider


def test_demo_page_is_served_without_provider(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert (
        "Демо использует подготовленный разбор. Реальная AI-обработка "
        "находится в технической проверке"
    ) in response.text
    assert "Анализ может занять несколько минут" in response.text
    assert "максимальное ожидание — около 61 минуты" in response.text
    assert "Запустить AI-анализ" in response.text
    assert "if (analysisInProgress) return;" in response.text
    assert 'const sampleNotice = normalizeNoticeText(noticeInput.defaultValue);' in response.text
    assert "noticeInput.value = sampleNotice;" in response.text
    assert "demoDismissedForAnalysis = false;" in response.text


def test_prepared_demo_state_is_tied_to_restored_source_text():
    html = (prototype.BASE_DIR / "index.html").read_text(encoding="utf-8")
    assert "function normalizeNoticeText(text)" in html
    assert (
        "const isPreparedSample = normalizeNoticeText(noticeInput.value) === sampleNotice;"
    ) in html
    assert "preparedAnalysis.hidden = !showPreparedDemo;" in html
    assert "otherTextMessage.hidden = isPreparedSample;" in html
    assert (
        'document.getElementById("restore-sample").addEventListener("click", restorePreparedSample);'
    ) in html


def test_start_script_keeps_model_and_variant_explicitly_configured():
    script = (prototype.BASE_DIR / "start_nta.ps1").read_text(encoding="utf-8-sig")
    assert "$env:NTA_PROVIDER_MODEL =" not in script
    assert "$env:NTA_GEMINI_CONFIG_VARIANT =" not in script
    assert "NTA_PROVIDER_API_KEY" in script
    assert "AsSecureString" in script
    assert "Test-Path -LiteralPath $envFile -PathType Leaf" in script
    assert "GetNetworkCredential().Password" in script


def test_gitignore_excludes_local_secrets_caches_and_user_material():
    patterns = set((prototype.BASE_DIR / ".gitignore").read_text().splitlines())
    assert {
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".env",
        "*.env",
        "!.env.example",
    } <= patterns
    assert {
        "*.key",
        "*.pem",
        "*api_key*",
        "*secret*",
        "*.docx",
        "*.pdf",
        "*.csv",
        "personal/",
        "private/",
    } <= patterns


def test_local_env_file_is_loaded_without_logging_contents(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("NTA_PROVIDER_MODEL", raising=False)
    monkeypatch.delenv("NTA_PROVIDER_API_KEY", raising=False)
    fake_key = "test-only-not-a-real-key"
    env_file = tmp_path / "local.env"
    env_file.write_text(
        f"NTA_PROVIDER_MODEL=gemini-test\nNTA_PROVIDER_API_KEY={fake_key}\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.DEBUG):
        prototype.load_local_env(env_file)

    assert os.environ["NTA_PROVIDER_MODEL"] == "gemini-test"
    assert os.environ["NTA_PROVIDER_API_KEY"] == fake_key
    assert fake_key not in caplog.text


def test_process_environment_takes_precedence_over_local_env(tmp_path, monkeypatch):
    monkeypatch.setenv("NTA_PROVIDER_MODEL", "process-model")
    monkeypatch.setenv("NTA_PROVIDER_API_KEY", "process-test-key")
    env_file = tmp_path / "override.env"
    env_file.write_text(
        "NTA_PROVIDER_MODEL=file-model\nNTA_PROVIDER_API_KEY=file-test-key\n",
        encoding="utf-8",
    )

    prototype.load_local_env(env_file)

    assert os.environ["NTA_PROVIDER_MODEL"] == "process-model"
    assert os.environ["NTA_PROVIDER_API_KEY"] == "process-test-key"


def test_explicit_empty_process_value_is_not_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("NTA_PROVIDER_API_KEY", "")
    env_file = tmp_path / "empty-process-value.env"
    env_file.write_text(
        "NTA_PROVIDER_API_KEY=file-test-key\n",
        encoding="utf-8",
    )

    prototype.load_local_env(env_file)

    assert os.environ["NTA_PROVIDER_API_KEY"] == ""


def test_local_env_loader_ignores_missing_files(tmp_path):
    prototype.load_local_env(tmp_path / "missing.env")


def test_local_env_loader_only_imports_notice_to_action_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("PATH_FROM_LOCAL_ENV", raising=False)
    monkeypatch.delenv("NTA_PROVIDER_MODEL", raising=False)
    env_file = tmp_path / "limited.env"
    env_file.write_text(
        "PATH_FROM_LOCAL_ENV=should-not-load\nNTA_PROVIDER_MODEL=file-model\n",
        encoding="utf-8",
    )

    prototype.load_local_env(env_file)

    assert "PATH_FROM_LOCAL_ENV" not in os.environ
    assert os.environ["NTA_PROVIDER_MODEL"] == "file-model"


def test_env_example_contains_template_only_and_ignore_exempts_it():
    example = (prototype.BASE_DIR / ".env.example").read_text(encoding="utf-8")
    assert "NTA_PROVIDER_MODEL=gemini-3.8-flash" in example
    assert "NTA_PROVIDER_API_KEY=вставь_ключ_сюда" in example
    patterns = (prototype.BASE_DIR / ".gitignore").read_text().splitlines()
    assert patterns.index("!.env.example") > patterns.index("*.env")


def test_unconfigured_analysis_does_not_return_prepared_result(client):
    response = client.post("/api/analyze", json={"text": "Иное объявление"})
    assert response.status_code == 503
    assert "AI-анализ пока не настроен." in response.json()["detail"]
    assert "participants" not in response.json()


def test_tests_do_not_use_environment_credentials(client, monkeypatch):
    monkeypatch.setenv("NTA_PROVIDER_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("NTA_PROVIDER_MODEL", "gemini-3.8-flash")
    response = client.post("/api/analyze", json={"text": "Иное объявление"})
    assert response.status_code == 503
    assert "participants" not in response.json()


def test_empty_input_is_rejected(client):
    response = client.post("/api/analyze", json={"text": "  "})
    assert response.status_code == 422


def test_request_over_25_kib_is_rejected(client):
    response = client.post(
        "/api/analyze",
        content=b'{"text":"' + b"a" * (25 * 1024 + 1) + b'"}',
    )
    assert response.status_code == 413


def test_utf8_text_over_25_kib_is_rejected(client):
    provider = StubProvider()
    inject_provider(provider)
    response = client.post("/api/analyze", json={"text": "한" * 9000})
    assert response.status_code == 413
    assert provider.calls == 0


def test_valid_structured_response_and_multiple_deadlines(client):
    provider = StubProvider()
    inject_provider(provider)
    response = client.post("/api/analyze", json={"text": NOTICE})
    assert response.status_code == 200
    assert len(response.json()["deadlines"]) == 2
    assert provider.calls == 1


def test_invalid_response_structure_is_rejected(client):
    provider = StubProvider({"unexpected": "field"})
    inject_provider(provider)
    response = client.post("/api/analyze", json={"text": NOTICE})
    assert response.status_code == 502


def test_missing_source_excerpt_is_rejected(client):
    result = {**VALID_RESULT, "participants": [{
        "status": "stated",
        "statement": "Несуществующий источник.",
        "evidence": ["Текста нет в объявлении"],
    }]}
    inject_provider(StubProvider(result))
    response = client.post("/api/analyze", json={"text": NOTICE})
    assert response.status_code == 502


def test_stated_finding_without_excerpt_is_rejected(client):
    result = {**VALID_RESULT, "participants": [{
        "status": "stated",
        "statement": "Нет ссылки.",
        "evidence": [],
    }]}
    inject_provider(StubProvider(result))
    response = client.post("/api/analyze", json={"text": NOTICE})
    assert response.status_code == 502


def test_timeout_returns_safe_error_without_prepared_fallback(client, monkeypatch):
    async def timeout(_: str):
        await asyncio.sleep(1)

    monkeypatch.setattr(prototype, "ANALYSIS_TIMEOUT_SECONDS", 0.01)
    inject_provider(StubProvider(result=timeout))
    response = client.post("/api/analyze", json={"text": NOTICE})
    assert response.status_code == 504
    assert "participants" not in response.json()


def test_provider_error_returns_safe_diagnostics_without_provider_details(client, caplog):
    inject_provider(StubProvider(
        error=GeminiProviderError(
            "invalid_parameters",
            http_code=400,
            status="INVALID_ARGUMENT",
            reason_code="BAD_REQUEST",
        )
    ))
    response = client.post("/api/analyze", json={"text": NOTICE})
    assert response.status_code == 502
    assert "параметры запроса" in response.json()["detail"]
    assert "http=400" in response.json()["detail"]
    assert "status=INVALID_ARGUMENT" in response.json()["detail"]
    assert "reason=BAD_REQUEST" in response.json()["detail"]
    assert "категория=invalid_parameters" in caplog.text
    assert "http=400" in caplog.text
    assert "status=INVALID_ARGUMENT" in caplog.text
    assert "reason=BAD_REQUEST" in caplog.text
    assert "participants" not in response.json()


def test_provider_deadline_error_returns_retryable_timeout_message(client):
    inject_provider(StubProvider(
        error=GeminiProviderError(
            "provider_timeout",
            http_code=504,
            status="DEADLINE_EXCEEDED",
        )
    ))
    response = client.post("/api/analyze", json={"text": NOTICE})
    assert response.status_code == 504
    assert response.json()["detail"].startswith(
        "Gemini не успел завершить анализ. Это временная ошибка; "
        "можно повторить попытку позже."
    )
    assert "http=504" in response.json()["detail"]
    assert "status=DEADLINE_EXCEEDED" in response.json()["detail"]


def test_server_timeout_allows_three_twenty_minute_attempts():
    assert prototype.ANALYSIS_TIMEOUT_SECONDS >= 3 * 1200 + 2 * 30


def test_rate_limit_allows_five_attempts_and_rejects_sixth(client):
    for _ in range(5):
        response = client.post("/api/analyze", json={"text": NOTICE})
        assert response.status_code == 503
    response = client.post("/api/analyze", json={"text": NOTICE})
    assert response.status_code == 429


def test_attempt_limiter_expires_attempts_after_a_minute():
    limiter = prototype.AttemptLimiter()
    for attempt in range(5):
        assert limiter.allow("local", now=float(attempt))
    assert not limiter.allow("local", now=5.0)
    assert limiter.allow("local", now=60.0)


def test_unicode_normalization_is_applied_to_evidence():
    result = prototype.AnalysisResponse.model_validate(VALID_RESULT)
    prototype.validate_evidence(
        result,
        "참가 대상:\u00a0 대학생. 신청 기간:\u00a010월 1일 ~ 10월 8일.",
    )


def test_arbitrary_input_has_no_prepared_answer_fallback(client):
    response = client.post("/api/analyze", json={"text": "임의의 다른 공고"})
    assert response.status_code == 503
    assert "participants" not in response.json()


@pytest.mark.parametrize(
    "path",
    [
        "/app.py",
        "/provider.py",
        "/index.html",
        "/README.md",
        "/requirements.txt",
        "/tests/test_app.py",
        "/.env",
    ],
)
def test_server_does_not_expose_project_files(client, path):
    response = client.get(path)
    assert response.status_code == 404
