# Gemini 2.5 Flash 어댑터 — 무료 250/일, 한국어 자연스러움이 주력 선택 근거.
from __future__ import annotations

from typing import Any

import httpx

from execution.ai_client._base import (
    build_user_content,
    classify_http_exception,
    classify_http_status,
)
from execution.ai_client.fallback_chain import FallbackError

_DEFAULT_URL = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiClient:
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        base_url: str = _DEFAULT_URL,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._client = client or httpx.Client(timeout=timeout)

    def generate(self, payload: dict[str, Any], prompt: str) -> str:
        url = f"{self._base_url}/{self._model}:generateContent"
        body = {
            "contents": [
                {"role": "user", "parts": [{"text": build_user_content(prompt, payload)}]}
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1500,
                # Gemini 2.5 Flash 는 기본적으로 thinking 토큰을 소비해 실제 출력이 잘릴 수 있음.
                # 본 프로젝트 브리핑은 reasoning 이 거의 필요 없어 예산=0 으로 둔다.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        }
        try:
            resp = self._client.post(url, params={"key": self._api_key}, json=body)
        except Exception as exc:                                       # httpx 네트워크 계열만 분류
            raise classify_http_exception(exc) from exc

        err = classify_http_status(resp.status_code, resp.text[:120])
        if err is not None:
            raise err

        return _extract_text(resp.json())


def _extract_text(data: dict[str, Any]) -> str:
    # 응답 형식:
    # {"candidates":[{"content":{"parts":[{"text": "..."}]}}]}
    try:
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise FallbackError(f"gemini malformed response: {exc}") from exc
    if not text:
        raise FallbackError("gemini empty completion")
    return text
