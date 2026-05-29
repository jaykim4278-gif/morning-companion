# Groq Llama 3.3 70B 어댑터 — OpenAI 호환 포맷, 무료 14,400/일 폴백 1.
from __future__ import annotations

from typing import Any

import httpx

from execution.ai_client._base import (
    build_user_content,
    classify_http_exception,
    classify_http_status,
)
from execution.ai_client.fallback_chain import FallbackError

_DEFAULT_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient:
    name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        url: str = _DEFAULT_URL,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = url
        self._client = client or httpx.Client(timeout=timeout)

    def generate(self, payload: dict[str, Any], prompt: str) -> str:
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": "You are a health briefing assistant. Respond in Korean."},
                {"role": "user", "content": build_user_content(prompt, payload)},
            ],
            "temperature": 0.7,
            "max_tokens": 800,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self._client.post(self._url, headers=headers, json=body)
        except Exception as exc:
            raise classify_http_exception(exc) from exc

        err = classify_http_status(resp.status_code, resp.text[:120])
        if err is not None:
            raise err

        return _extract_text(resp.json())


def _extract_text(data: dict[str, Any]) -> str:
    # OpenAI 호환: {"choices":[{"message":{"content":"..."}}]}
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError, TypeError) as exc:
        raise FallbackError(f"groq malformed response: {exc}") from exc
    if not content:
        raise FallbackError("groq empty completion")
    return content
