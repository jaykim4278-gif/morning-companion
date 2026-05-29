# 공급자 어댑터 공통 유틸 — HTTP 응답·예외를 Provider 계약으로 분류한다.
from __future__ import annotations

import httpx

from execution.ai_client.fallback_chain import FallbackError, RetryableError


def classify_http_exception(exc: Exception) -> Exception:
    """httpx 네트워크 예외를 체인 예외로 분류.

    - TimeoutException / ConnectError / ReadError → RetryableError (일시적)
    - 그 외 httpx.HTTPError → RetryableError (보수적)
    """
    if isinstance(exc, httpx.TimeoutException):
        return RetryableError(f"timeout: {exc}")
    if isinstance(exc, httpx.NetworkError):
        return RetryableError(f"network: {exc}")
    if isinstance(exc, httpx.HTTPError):
        return RetryableError(f"http: {exc}")
    return exc  # unknown — 호출자에게 그대로 전파


def classify_http_status(status: int, body_preview: str) -> Exception | None:
    """HTTP 상태코드를 체인 예외로 분류. 2xx 는 None 반환(정상).

    - 429            → FallbackError (quota 소진, 재시도 무의미)
    - 401 / 403      → FallbackError (인증 실패, 재시도 무의미)
    - 408 / 5xx      → RetryableError (일시 장애)
    - 그 외 4xx      → FallbackError (요청 자체 문제 — 재시도 무의미)
    """
    if 200 <= status < 300:
        return None
    if status == 429:
        return FallbackError(f"quota 429: {body_preview}")
    if status in (401, 403):
        return FallbackError(f"auth {status}: {body_preview}")
    if status == 408 or 500 <= status < 600:
        return RetryableError(f"server {status}: {body_preview}")
    return FallbackError(f"client {status}: {body_preview}")


def build_user_content(prompt: str, payload: dict) -> str:
    """OpenAI 호환 포맷의 user content — 프롬프트 + 구조화 데이터 JSON."""
    import json
    return f"{prompt}\n\n---\nDATA (JSON):\n{json.dumps(payload, ensure_ascii=False)}"
