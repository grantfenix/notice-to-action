from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import unicodedata
from collections import defaultdict, deque
from enum import Enum
from pathlib import Path
from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from provider import (
    PROVIDER_HTTP_TIMEOUT_MS,
    PROVIDER_RETRY_ATTEMPTS,
    PROVIDER_RETRY_MAX_DELAY_SECONDS,
    GeminiProviderError,
    Provider,
    create_gemini_provider,
)


BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"
ENV_FILE = BASE_DIR / ".env"
LOCAL_ENV_KEYS = {
    "NTA_PROVIDER_MODEL",
    "NTA_PROVIDER_API_KEY",
    "NTA_GEMINI_CONFIG_VARIANT",
}


def load_local_env(env_file: Path = ENV_FILE) -> None:
    if not env_file.is_file():
        return

    for line_number, raw_line in enumerate(
        env_file.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()

        key, separator, raw_value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(f"Invalid .env assignment on line {line_number}")
        if key not in LOCAL_ENV_KEYS:
            continue

        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()

        if key not in os.environ:
            os.environ[key] = value


load_local_env()
MAX_INPUT_BYTES = 25 * 1024
MAX_REQUEST_BYTES = MAX_INPUT_BYTES * 6 + 1024
MAX_ATTEMPTS_PER_MINUTE = 5
ANALYSIS_TIMEOUT_SECONDS = (
    PROVIDER_RETRY_ATTEMPTS * PROVIDER_HTTP_TIMEOUT_MS // 1000
    + (PROVIDER_RETRY_ATTEMPTS - 1) * PROVIDER_RETRY_MAX_DELAY_SECONDS
    + 5
)
logger = logging.getLogger(__name__)


class FindingStatus(str, Enum):
    STATED = "stated"
    NOT_STATED = "not_stated"
    NEEDS_CLARIFICATION = "needs_clarification"
    BASIS_UNCONFIRMED = "basis_unconfirmed"


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: FindingStatus
    statement: str = Field(min_length=1, max_length=1000)
    evidence: list[str] = Field(max_length=6)

    @field_validator("evidence")
    @classmethod
    def validate_evidence_items(cls, evidence: list[str]) -> list[str]:
        if any(not excerpt.strip() or len(excerpt) > 1000 for excerpt in evidence):
            raise ValueError("Evidence excerpts must be non-empty and at most 1000 characters")
        return evidence


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    participants: list[Finding] = Field(max_length=20)
    exclusions: list[Finding] = Field(max_length=20)
    actions: list[Finding] = Field(max_length=20)
    documents: list[Finding] = Field(max_length=20)
    deadlines: list[Finding] = Field(max_length=20)
    unknowns: list[Finding] = Field(max_length=20)
    organizer_question: str = Field(max_length=2000)

    @field_validator(
        "participants", "exclusions", "actions", "documents", "deadlines", "unknowns"
    )
    @classmethod
    def validate_finding_statuses(cls, findings: list[Finding]) -> list[Finding]:
        for finding in findings:
            if finding.status == FindingStatus.STATED and not finding.evidence:
                raise ValueError("Stated findings require at least one source excerpt")
            if finding.status == FindingStatus.NOT_STATED and finding.evidence:
                raise ValueError("Not-stated findings cannot invent source excerpts")
        return findings


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class AttemptLimiter:
    def __init__(self) -> None:
        self._attempts: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, client_key: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        attempts = self._attempts[client_key]
        while attempts and current - attempts[0] >= 60:
            attempts.popleft()
        if len(attempts) >= MAX_ATTEMPTS_PER_MINUTE:
            return False
        attempts.append(current)
        return True

    def reset(self) -> None:
        self._attempts.clear()


limiter = AttemptLimiter()
provider_slots = asyncio.Semaphore(1)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, debug=False)


def get_provider() -> Provider | None:
    return create_gemini_provider()


def normalize_excerpt(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return " ".join(normalized.split())


def validate_evidence(response: AnalysisResponse, notice: str) -> None:
    normalized_notice = normalize_excerpt(notice)
    collections = (
        response.participants,
        response.exclusions,
        response.actions,
        response.documents,
        response.deadlines,
        response.unknowns,
    )
    for findings in collections:
        for finding in findings:
            for excerpt in finding.evidence:
                if normalize_excerpt(excerpt) not in normalized_notice:
                    raise ValueError("A source excerpt was not found in the submitted notice")


@app.middleware("http")
async def restrict_to_loopback(request: Request, call_next):
    client_host = request.client.host if request.client else None
    if client_host not in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return JSONResponse(status_code=403, content={"detail": "Local access only"})
    return await call_next(request)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(
        INDEX_FILE,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/analyze")
async def analyze(request: Request, provider: Provider | None = Depends(get_provider)):
    client_host = request.client.host if request.client else "unknown"
    if not limiter.allow(client_host):
        return JSONResponse(
            status_code=429,
            content={"detail": "Слишком много попыток. Подождите минуту и попробуйте снова."},
        )

    content_length = request.headers.get("content-length")
    if content_length and content_length.isdecimal() and int(content_length) > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Размер запроса превышает допустимый предел."},
        )
    body_parts: list[bytes] = []
    body_size = 0
    async for part in request.stream():
        body_size += len(part)
        if body_size > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={"detail": "Размер запроса превышает допустимый предел."},
            )
        body_parts.append(part)
    raw_body = b"".join(body_parts)
    try:
        payload = AnalyzeRequest.model_validate_json(raw_body)
    except ValidationError:
        return JSONResponse(status_code=422, content={"detail": "Некорректный запрос."})

    if not payload.text.strip():
        return JSONResponse(status_code=422, content={"detail": "Введите текст объявления."})
    if len(payload.text.encode("utf-8")) > MAX_INPUT_BYTES:
        return JSONResponse(
            status_code=413,
            content={"detail": "Размер текста превышает лимит 25 КБ UTF-8."},
        )
    if provider is None:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "AI-анализ пока не настроен. Задайте "
                    "NTA_PROVIDER_API_KEY и NTA_PROVIDER_MODEL в окружении сервера."
                )
            },
        )

    try:
        async with provider_slots:
            raw_result = await asyncio.wait_for(
                provider.analyze(payload.text),
                timeout=ANALYSIS_TIMEOUT_SECONDS,
            )
    except TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "detail": (
                    "Gemini не успел завершить анализ. Это временная ошибка; "
                    "можно повторить попытку позже"
                )
            },
        )
    except GeminiProviderError as error:
        messages = {
            "timeout": "Истекло время ожидания анализа. Повторите попытку вручную.",
            "provider_timeout": (
                "Gemini не успел завершить анализ. Это временная ошибка; "
                "можно повторить попытку позже."
            ),
            "invalid_key": "Gemini отклонил ключ. Проверьте локальную конфигурацию.",
            "access_denied": "Доступ к выбранной модели Gemini отклонён.",
            "model_unavailable": "Выбранная модель Gemini недоступна для этого API-проекта.",
            "quota": "Квота или лимит запросов Gemini исчерпаны.",
            "invalid_parameters": "Gemini отклонил параметры запроса.",
            "unknown": "Gemini вернул неизвестную ошибку провайдера.",
            "refused": "Gemini не сформировал ответ для этого текста.",
            "empty": "Gemini вернул пустой ответ.",
            "truncated": "Ответ Gemini обрезан. Повторите попытку вручную.",
            "invalid_response": "Ответ Gemini не является корректным JSON.",
        }
        diagnostic = (
            f"категория={error.reason}; "
            f"http={error.http_code if error.http_code is not None else 'нет'}; "
            f"status={error.status or 'нет'}; "
            f"reason={error.reason_code or 'нет'}"
        )
        logger.warning("Gemini API failure %s", diagnostic)
        return JSONResponse(
            status_code=504 if error.reason == "provider_timeout" else 502,
            content={
                "detail": f"{messages.get(error.reason, messages['unknown'])} {diagnostic}",
            },
        )
    except Exception:
        logger.error("Unknown analysis failure; provider diagnostics unavailable")
        return JSONResponse(
            status_code=502,
            content={"detail": "Не удалось выполнить анализ. Проверьте текст или попробуйте позже."},
        )
    finally:
        await provider.aclose()

    try:
        result = AnalysisResponse.model_validate(raw_result)
        validate_evidence(result, payload.text)
    except (ValidationError, ValueError, TypeError):
        return JSONResponse(
            status_code=502,
            content={"detail": "Ответ анализа не прошёл проверку структуры и источников."},
        )
    return JSONResponse(content=json.loads(result.model_dump_json()))
