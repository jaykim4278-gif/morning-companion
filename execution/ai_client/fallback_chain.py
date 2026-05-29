# Gemini(주력) -> Groq -> OpenRouter 3단 폴백 — 단일 공급자 SPOF 제거.
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from execution.ai_client.anonymize import anonymize_payload

logger = logging.getLogger(__name__)


# ============================================================
# §1  Provider interface
# ============================================================
class Provider(Protocol):
    """모든 AI 제공자가 구현해야 하는 최소 계약."""
    name: str

    def generate(self, payload: dict[str, Any], prompt: str) -> str:
        """성공 시 텍스트 반환. 재시도 가능한 오류는 RetryableError,
        즉시 폴백해야 하는 오류는 FallbackError 로 발생시킨다."""
        ...


# ============================================================
# §2  Exception taxonomy
# ============================================================
class RetryableError(Exception):
    """지수 백오프 후 재시도 가능 (네트워크, 5xx, 일시적 timeout)."""


class FallbackError(Exception):
    """같은 공급자 재시도 의미 없음 — 즉시 다음 공급자로 (429 quota, 인증 실패)."""


class AllProvidersFailed(Exception):
    """체인 전체 소진 — 호출자는 템플릿 폴백으로 전환해야 함."""


# ============================================================
# §3  Chain configuration
# ============================================================
@dataclass(frozen=True)
class ChainConfig:
    max_retries_per_provider: int = 2
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 8.0


@dataclass
class ChainResult:
    text: str
    provider: str                               # 성공 공급자 이름
    attempts: list[tuple[str, str]]             # [(provider_name, outcome), ...]
    provider_index: int = 0                     # 성공 공급자의 providers list 내 인덱스 (sticky cursor 용)


# ============================================================
# §4  Core dispatcher
# ============================================================
def call_with_fallback(
    payload: dict[str, Any],
    prompt: str,
    providers: list[Provider],
    drug_map: dict[str, str] | None = None,
    config: ChainConfig | None = None,
    sleep: Callable[[float], None] = time.sleep,
    start_index: int = 0,
) -> ChainResult:
    """PII 익명화 후 공급자 체인 순회.

    - 매 호출마다 anonymize_payload() 를 한 번만 수행 (체인 전체에서 재사용).
    - 각 공급자에서 RetryableError 는 지수 백오프 재시도, FallbackError 는 즉시 다음 공급자.
    - 전 공급자 소진 시 AllProvidersFailed 발생 → 호출자가 템플릿 브리핑으로 전환.

    Args:
        payload: AI 에 전달할 구조화 데이터 (user_id 등 PII 포함 가능 — 여기서 제거)
        prompt: 시스템/유저 프롬프트 원문 (string)
        providers: [gemini, groq, openrouter] 순서 (우선순위)
        drug_map: real_name(lower) -> code_name
        config: 재시도/백오프 설정
        sleep: 테스트에서 monkey-patch 가능한 슬리퍼
        start_index: 이 인덱스 이후 provider 만 시도. sticky cursor 용 — 직전 호출에서
                     이미 죽은 provider 를 다시 두드리지 않도록 caller 가 갱신해서 전달.
                     0 (기본) = 처음부터 (Gemini → Groq → OpenRouter).

    Returns:
        ChainResult(text, provider, attempts, provider_index)
        provider_index 는 성공한 provider 의 절대 인덱스 — caller 가 다음 호출의 start_index 로 사용.
    """
    if not providers:
        raise ValueError("providers list must not be empty")
    if start_index >= len(providers):
        raise AllProvidersFailed(
            f"start_index={start_index} >= len(providers)={len(providers)} — 사용 가능 provider 없음"
        )

    cfg = config or ChainConfig()
    sanitized = anonymize_payload(payload, drug_map=drug_map or {}, strict=True)
    attempts: list[tuple[str, str]] = []

    for idx in range(start_index, len(providers)):
        provider = providers[idx]
        backoff = cfg.base_backoff_seconds
        for attempt_num in range(cfg.max_retries_per_provider + 1):
            try:
                text = provider.generate(sanitized, prompt)
            except FallbackError as e:
                attempts.append((provider.name, f"fallback:{e}"))
                logger.warning("Provider %s signalled fallback: %s", provider.name, e)
                break  # next provider
            except RetryableError as e:
                attempts.append((provider.name, f"retry:{e}"))
                if attempt_num >= cfg.max_retries_per_provider:
                    logger.warning(
                        "Provider %s exhausted %d retries: %s",
                        provider.name, cfg.max_retries_per_provider, e,
                    )
                    break  # next provider
                sleep(min(backoff, cfg.max_backoff_seconds))
                backoff *= 2
                continue
            else:
                attempts.append((provider.name, "ok"))
                return ChainResult(
                    text=text, provider=provider.name, attempts=attempts, provider_index=idx,
                )

    raise AllProvidersFailed(f"All {len(providers) - start_index} providers failed: {attempts}")
