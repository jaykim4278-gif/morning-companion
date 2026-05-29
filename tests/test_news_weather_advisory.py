# W8a 확장 (2026-04-21) — 날씨 조언 LLM + 규칙 기반 폴백 테스트.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from execution.ai_client.fallback_chain import ChainConfig, FallbackError
from execution.github_actions.news.weather import (
    DailyForecast,
    WeatherAlert,
    WeatherSnapshot,
    WeatherWeek,
)
from execution.github_actions.news.weather_advisory import (
    _heuristic_advisory,
    generate_weather_advisory,
)


FAST = ChainConfig(max_retries_per_provider=1, base_backoff_seconds=0.0, max_backoff_seconds=0.0)


@dataclass
class FakeProvider:
    name: str
    responses: list[Any] = field(default_factory=list)
    calls: int = 0

    def generate(self, payload: dict, prompt: str) -> str:
        idx = self.calls
        self.calls += 1
        if idx >= len(self.responses):
            return f"{self.name}-default"
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        return item


def _snap(temp_f: float = 72, precip: int = 10, label: str = "맑음") -> WeatherSnapshot:
    return WeatherSnapshot(
        temp_c=(temp_f - 32) * 5 / 9, temp_f=temp_f, weather_label=label,
        today_high_c=25, today_low_c=18,
        today_high_f=77, today_low_f=64, precip_prob_pct=precip,
    )


def _week(rainy_days: int = 0) -> WeatherWeek:
    days = []
    for i in range(7):
        is_rainy = i > 0 and i <= rainy_days
        days.append(DailyForecast(
            date_iso=f"2026-04-{21+i}",
            high_f=78.0, low_f=60.0,
            weather_code=63 if is_rainy else 0,
            weather_label="비" if is_rainy else "맑음",
            precip_prob_pct=80 if is_rainy else 10,
            precip_sum_mm=10.0 if is_rainy else 0.0,
        ))
    return WeatherWeek(days=tuple(days))


# ============================================================
# generate_weather_advisory — LLM 우선, 폴백 규칙
# ============================================================
class TestGenerateAdvisory:
    def test_llm_success_returns_stripped_text(self) -> None:
        p = FakeProvider("gemini", ["오늘 72°F 로 쾌적합니다. 이번 주 후반 비 예보로 우산 챙기세요."])
        txt = generate_weather_advisory(_snap(), _week(3), [], [p], config=FAST)
        assert "72°F" in txt
        assert "우산" in txt

    def test_llm_all_fail_falls_back_to_heuristic(self) -> None:
        p = FakeProvider("gemini", [FallbackError("quota")] * 5)
        txt = generate_weather_advisory(_snap(temp_f=55), _week(4), [], [p], config=FAST)
        # 규칙 기반: 쌀쌀 + 긴팔·재킷 + 비 예보.
        assert "55°F" in txt
        assert "긴팔" in txt or "재킷" in txt

    def test_snapshot_none_returns_none(self) -> None:
        p = FakeProvider("gemini", ["should not be called"])
        assert generate_weather_advisory(None, None, [], [p], config=FAST) is None
        assert p.calls == 0

    def test_no_providers_uses_heuristic(self) -> None:
        # providers=[] 직접 heuristic 분기.
        txt = generate_weather_advisory(_snap(temp_f=90, precip=80), _week(0), [], [], config=FAST)
        assert "더위" in txt or "90°F" in txt
        assert "우산" in txt  # precip 80% 분기

    def test_markdown_stripped_from_llm_output(self) -> None:
        p = FakeProvider("gemini", ["**오늘 맑음**. 우산 불필요."])
        txt = generate_weather_advisory(_snap(), _week(), [], [p], config=FAST)
        # **·_·`·"·' 흔적 제거.
        assert not txt.startswith("*")
        assert not txt.startswith("_")

    def test_cjk_triggers_retry_and_accepts_clean(self) -> None:
        # 첫 응답 한자 포함 → 재시도 → 순수 한글 수용.
        p = FakeProvider("gemini", [
            "오늘은 비가 氏作. 우산 준비.",
            "오늘은 비가 옵니다. 우산을 준비하세요.",
        ])
        txt = generate_weather_advisory(_snap(temp_f=62, precip=70), _week(), [], [p], config=FAST)
        assert "氏" not in txt
        assert "우산" in txt

    def test_cjk_twice_falls_back_to_heuristic(self) -> None:
        # 두 번 다 한자 → heuristic (순수 한글 보장).
        p = FakeProvider("gemini", [
            "오늘은 비가 氏作.",
            "오늘은 비가 物質.",
        ])
        txt = generate_weather_advisory(_snap(temp_f=55, precip=80), _week(), [], [p], config=FAST)
        # heuristic 결과: 한자 없음 + 핵심 키워드 포함.
        assert "氏" not in txt
        assert "物" not in txt
        # heuristic 의 55°F + precip 80% 두 조건은 "쌀쌀·긴팔" + "우산" 를 생성.
        assert "우산" in txt


# ============================================================
# _heuristic_advisory — 규칙 기반 직접 테스트
# ============================================================
class TestHeuristicAdvisory:
    def test_cold_temperature_suggests_jacket(self) -> None:
        txt = _heuristic_advisory(_snap(temp_f=40), _week(), [])
        assert "40°F" in txt
        assert "외투" in txt

    def test_mild_temperature_mentions_comfortable(self) -> None:
        txt = _heuristic_advisory(_snap(temp_f=68), _week(), [])
        assert "쾌적" in txt

    def test_hot_temperature_suggests_hydration(self) -> None:
        txt = _heuristic_advisory(_snap(temp_f=92), _week(), [])
        assert "수분" in txt

    def test_high_precip_warns_umbrella(self) -> None:
        txt = _heuristic_advisory(_snap(precip=75), _week(), [])
        assert "우산" in txt

    def test_alert_mentioned_first(self) -> None:
        alert = WeatherAlert(event="Tornado Warning", headline="x", severity="Extreme", ends_at=None)
        txt = _heuristic_advisory(_snap(), _week(), [alert])
        # 2026-04-21 비서 톤: "현재 X 기상경보가 발령 중입니다..."
        assert "기상경보" in txt.split(".")[0]
        assert "Tornado Warning" in txt

    def test_week_rainy_mentions_long_rain(self) -> None:
        txt = _heuristic_advisory(_snap(temp_f=70), _week(5), [])
        assert "이번 주" in txt


class TestHeuristicPoliteTone:
    """2026-04-21 hotfix — 반말 금지, 비서 톤 존댓말 보장."""

    # 금지 어미 (평서문·반말).
    _BANNED_ENDINGS = ["좋겠다.", "하라.", "한다.", "이다.", "된다.", "먹어라.", "해라."]

    def _assert_polite(self, text: str) -> None:
        for banned in self._BANNED_ENDINGS:
            assert banned not in text, f"반말 어미 발견: '{banned}' in {text!r}"

    def test_cold_polite(self) -> None:
        self._assert_polite(_heuristic_advisory(_snap(temp_f=40), _week(), []))

    def test_mild_polite(self) -> None:
        self._assert_polite(_heuristic_advisory(_snap(temp_f=68), _week(), []))

    def test_hot_polite(self) -> None:
        self._assert_polite(_heuristic_advisory(_snap(temp_f=92), _week(), []))

    def test_precip_polite(self) -> None:
        self._assert_polite(_heuristic_advisory(_snap(precip=75), _week(), []))

    def test_alert_polite(self) -> None:
        alert = WeatherAlert(event="Flood Warning", headline="x", severity="Severe", ends_at=None)
        self._assert_polite(_heuristic_advisory(_snap(), _week(), [alert]))

    def test_rainy_week_polite(self) -> None:
        self._assert_polite(_heuristic_advisory(_snap(), _week(5), []))
