# Layer 4 Sensor — Gemini->Groq->OpenRouter 전환 및 PII 익명화 계약 검증.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from execution.ai_client.fallback_chain import (
    AllProvidersFailed,
    ChainConfig,
    FallbackError,
    RetryableError,
    call_with_fallback,
)


# ---------- 테스트용 Provider 더블 ----------

@dataclass
class FakeProvider:
    name: str
    responses: list[Any] = field(default_factory=list)   # 각 호출의 응답 또는 예외
    seen_payloads: list[dict[str, Any]] = field(default_factory=list)
    call_count: int = 0

    def generate(self, payload: dict[str, Any], prompt: str) -> str:
        self.seen_payloads.append(payload)
        idx = self.call_count
        self.call_count += 1
        if idx >= len(self.responses):
            return f"{self.name}-default"
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def fast_config() -> ChainConfig:
    return ChainConfig(max_retries_per_provider=2, base_backoff_seconds=0.0)


@pytest.fixture
def no_sleep():
    return lambda _s: None


# ---------- 핵심 전환 로직 ----------

def test_primary_success_skips_fallbacks(fast_config, no_sleep) -> None:
    gemini = FakeProvider("gemini", ["briefing OK"])
    groq = FakeProvider("groq")
    openrouter = FakeProvider("openrouter")

    result = call_with_fallback(
        payload={"vitals": {"bp_sys": 138}},
        prompt="make briefing",
        providers=[gemini, groq, openrouter],
        config=fast_config,
        sleep=no_sleep,
    )
    assert result.text == "briefing OK"
    assert result.provider == "gemini"
    assert groq.call_count == 0
    assert openrouter.call_count == 0


def test_quota_429_falls_back_to_groq(fast_config, no_sleep) -> None:
    gemini = FakeProvider("gemini", [FallbackError("429 quota")])
    groq = FakeProvider("groq", ["from groq"])
    openrouter = FakeProvider("openrouter")

    result = call_with_fallback(
        payload={"vitals": {}},
        prompt="p",
        providers=[gemini, groq, openrouter],
        config=fast_config,
        sleep=no_sleep,
    )
    assert result.provider == "groq"
    assert gemini.call_count == 1
    assert openrouter.call_count == 0
    assert ("gemini", "fallback:429 quota") in result.attempts


def test_retryable_error_retries_then_succeeds(fast_config, no_sleep) -> None:
    gemini = FakeProvider(
        "gemini",
        [RetryableError("timeout"), RetryableError("timeout"), "recovered"],
    )
    groq = FakeProvider("groq")

    result = call_with_fallback(
        payload={},
        prompt="p",
        providers=[gemini, groq],
        config=fast_config,
        sleep=no_sleep,
    )
    assert result.provider == "gemini"
    assert result.text == "recovered"
    assert gemini.call_count == 3
    assert groq.call_count == 0


def test_retries_exhausted_falls_through(fast_config, no_sleep) -> None:
    # max_retries_per_provider=2 => 총 3회 시도
    gemini = FakeProvider(
        "gemini",
        [RetryableError("t")] * 3,
    )
    groq = FakeProvider("groq", ["ok"])

    result = call_with_fallback(
        payload={},
        prompt="p",
        providers=[gemini, groq],
        config=fast_config,
        sleep=no_sleep,
    )
    assert result.provider == "groq"
    assert gemini.call_count == 3


def test_all_providers_fail_raises(fast_config, no_sleep) -> None:
    providers = [
        FakeProvider("gemini", [FallbackError("429")]),
        FakeProvider("groq", [FallbackError("429")]),
        FakeProvider("openrouter", [FallbackError("429")]),
    ]
    with pytest.raises(AllProvidersFailed) as exc:
        call_with_fallback(
            payload={},
            prompt="p",
            providers=providers,
            config=fast_config,
            sleep=no_sleep,
        )
    assert "3 providers" in str(exc.value)


# ---------- 익명화 계약 (security-principles.md §2) ----------

def test_payload_is_anonymized_before_dispatch(fast_config, no_sleep) -> None:
    gemini = FakeProvider("gemini", ["ok"])
    payload = {
        "vitals": {"bp_sys": 138, "user_id": "uuid-leak"},
        "medications": [{"display": "Lisinopril 10mg"}],
        "patient_name": "John Doe",        # 최상위 deny
    }
    call_with_fallback(
        payload=payload,
        prompt="p",
        providers=[gemini],
        drug_map={"lisinopril": "고혈압약A"},
        config=fast_config,
        sleep=no_sleep,
    )
    seen = gemini.seen_payloads[0]
    assert "patient_name" not in seen
    assert "user_id" not in seen["vitals"]
    assert "Lisinopril" not in str(seen)
    assert "고혈압약A" in str(seen)


def test_anonymization_runs_only_once(fast_config, no_sleep) -> None:
    # Gemini 2회 실패 후 Groq 성공 — 둘 다 동일 익명화 결과를 봐야 함.
    gemini = FakeProvider("gemini", [RetryableError("t"), RetryableError("t"), RetryableError("t")])
    groq = FakeProvider("groq", ["ok"])

    call_with_fallback(
        payload={"vitals": {"bp_sys": 138}, "patient_name": "X"},
        prompt="p",
        providers=[gemini, groq],
        config=fast_config,
        sleep=no_sleep,
    )
    assert gemini.seen_payloads[0] == gemini.seen_payloads[-1]
    assert gemini.seen_payloads[0] == groq.seen_payloads[0]
    assert "patient_name" not in groq.seen_payloads[0]


# ---------- 백오프 슬리퍼 주입 ----------

def test_backoff_sleep_is_called_between_retries(fast_config) -> None:
    calls: list[float] = []
    gemini = FakeProvider("gemini", [RetryableError("t"), "ok"])

    call_with_fallback(
        payload={},
        prompt="p",
        providers=[gemini],
        config=ChainConfig(max_retries_per_provider=2, base_backoff_seconds=1.0),
        sleep=calls.append,
    )
    # 첫 실패 -> 1회 sleep, 그 다음 성공 -> 추가 sleep 없음
    assert calls == [1.0]


# ---------- 입력 검증 ----------

def test_empty_provider_list_raises(fast_config, no_sleep) -> None:
    with pytest.raises(ValueError):
        call_with_fallback(
            payload={}, prompt="p", providers=[], config=fast_config, sleep=no_sleep
        )


# ---------- 2026-05-13 PR — start_index sticky cursor ----------

def test_start_index_skips_earlier_providers(fast_config, no_sleep) -> None:
    # start_index=1 → Gemini 건너뛰고 Groq 부터 시도.  Gemini.call_count == 0 검증.
    from execution.ai_client.fallback_chain import AllProvidersFailed  # noqa: F401 (import 검증)
    gemini = FakeProvider("gemini", [])  # 호출되면 IndexError default 응답 — 다행히 안 호출됨
    groq = FakeProvider("groq", ["from groq"])
    result = call_with_fallback(
        payload={}, prompt="p", providers=[gemini, groq],
        config=fast_config, sleep=no_sleep, start_index=1,
    )
    assert result.text == "from groq"
    assert result.provider == "groq"
    assert result.provider_index == 1
    # Gemini 는 한 번도 호출되지 않음.
    assert gemini.call_count == 0


def test_provider_index_reflects_success_position(fast_config, no_sleep) -> None:
    # Gemini FallbackError → Groq 성공.  result.provider_index == 1 (Groq 의 인덱스).
    gemini = FakeProvider("gemini", [FallbackError("dead")])
    groq = FakeProvider("groq", ["ok"])
    result = call_with_fallback(
        payload={}, prompt="p", providers=[gemini, groq],
        config=fast_config, sleep=no_sleep,
    )
    assert result.provider_index == 1


def test_start_index_at_end_raises_all_failed(fast_config, no_sleep) -> None:
    # start_index == len(providers) → 시도할 provider 없음 → AllProvidersFailed 즉시.
    from execution.ai_client.fallback_chain import AllProvidersFailed
    gemini = FakeProvider("gemini", ["should-not-be-called"])
    with pytest.raises(AllProvidersFailed):
        call_with_fallback(
            payload={}, prompt="p", providers=[gemini],
            config=fast_config, sleep=no_sleep, start_index=1,
        )
    assert gemini.call_count == 0


def test_start_index_default_zero_is_backward_compat(fast_config, no_sleep) -> None:
    # 기존 호출자들은 start_index 안 넘김 — default 0 으로 동작 유지.
    gemini = FakeProvider("gemini", ["ok"])
    result = call_with_fallback(
        payload={}, prompt="p", providers=[gemini],
        config=fast_config, sleep=no_sleep,
    )
    assert result.text == "ok"
    assert result.provider_index == 0
