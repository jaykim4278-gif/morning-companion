# W8a 확장 — 날씨 섹션 (Open-Meteo + NWS) 테스트.
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

import httpx
import pytest

from execution.github_actions.news.weather import (
    WeatherAlert,
    WeatherSnapshot,
    fetch_alerts,
    fetch_weather,
)


def _mock(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0)


# ============================================================
# Open-Meteo fetch_weather
# ============================================================
def _sample_payload(*, include_week: bool = True) -> dict:
    """7일 forecast 포함 Open-Meteo 응답 샘플."""
    if include_week:
        return {
            "current": {"temperature_2m": 22.5, "weather_code": 2},
            "daily": {
                "time": [f"2026-04-{21+i}" for i in range(7)],
                "temperature_2m_max": [26.0, 28.0, 25.0, 22.0, 20.0, 24.0, 27.0],
                "temperature_2m_min": [18.0, 19.0, 17.0, 15.0, 14.0, 16.0, 18.0],
                "weather_code": [2, 1, 61, 63, 65, 2, 0],
                "precipitation_probability_max": [30, 20, 70, 90, 80, 10, 0],
                "precipitation_sum": [0.0, 0.0, 5.2, 15.8, 10.1, 0.0, 0.0],
            },
        }
    return {
        "current": {"temperature_2m": 22.5, "weather_code": 2},
        "daily": {
            "temperature_2m_max": [26.0],
            "temperature_2m_min": [18.0],
            "precipitation_probability_max": [30],
        },
    }


class TestFetchWeather:
    def test_success_parses_temps_and_code(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert "open-meteo.com" in str(req.url)
            # forecast_days=7 요청 확인.
            assert req.url.params.get("forecast_days") == "7"
            return httpx.Response(200, json=_sample_payload())

        result = fetch_weather(0.0, 0.0, "UTC", client=_mock(handler))
        assert result is not None
        snap, week = result
        assert snap.temp_c == 22.5
        # F = 22.5 * 9/5 + 32 = 72.5
        assert 72 < snap.temp_f < 73
        assert snap.weather_label == "구름 조금"
        assert snap.today_high_c == 26.0
        assert snap.today_low_c == 18.0
        # 화씨 필드도 채워짐 (2026-04-21 hotfix).
        assert abs(snap.today_high_f - 78.8) < 0.5
        assert abs(snap.today_low_f - 64.4) < 0.5
        assert snap.precip_prob_pct == 30
        # 주간 예보 7일.
        assert len(week.days) == 7
        assert week.days[0].weather_label == "구름 조금"
        assert week.days[2].precip_prob_pct == 70

    def test_unknown_code_falls_back_to_numeric_label(self) -> None:
        payload = _sample_payload()
        payload["current"]["weather_code"] = 999
        result = fetch_weather(0.0, 0.0, "UTC", client=_mock(lambda _r: httpx.Response(200, json=payload)))
        assert result is not None
        assert "999" in result[0].weather_label

    def test_http_error_returns_none(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="unavailable")
        assert fetch_weather(0.0, 0.0, "UTC", client=_mock(handler)) is None

    def test_network_error_returns_none(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns fail")
        assert fetch_weather(0.0, 0.0, "UTC", client=_mock(handler)) is None

    def test_malformed_json_returns_none(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")
        assert fetch_weather(0.0, 0.0, "UTC", client=_mock(handler)) is None

    def test_missing_current_fields_returns_none(self) -> None:
        # current.temperature_2m 누락 → snapshot 파싱 실패 → None.
        payload = {
            "current": {"weather_code": 0},
            "daily": {
                "temperature_2m_max": [], "temperature_2m_min": [],
                "precipitation_probability_max": [],
            },
        }
        assert fetch_weather(0.0, 0.0, "UTC", client=_mock(lambda _r: httpx.Response(200, json=payload))) is None

    def test_week_parse_failure_still_returns_snapshot(self) -> None:
        # daily time 배열 없음 — snapshot 은 [0] 인덱스로 살리고 week.days=().
        payload = {
            "current": {"temperature_2m": 20.0, "weather_code": 0},
            "daily": {
                "temperature_2m_max": [25.0],
                "temperature_2m_min": [15.0],
                "precipitation_probability_max": [10],
            },
        }
        result = fetch_weather(0.0, 0.0, "UTC", client=_mock(lambda _r: httpx.Response(200, json=payload)))
        assert result is not None
        snap, week = result
        assert snap.temp_c == 20.0
        # week 는 time 없음 → days 빈 tuple (안전 폴백).
        assert len(week.days) == 0


class TestWeatherSnapshotFormatting:
    # 2026-04-21 hotfix: 화씨 전용 표기 (섭씨 제거).
    def _s(self, **overrides) -> WeatherSnapshot:
        base = dict(
            temp_c=22.0, temp_f=71.6, weather_label="맑음",
            today_high_c=26, today_low_c=18,
            today_high_f=78.8, today_low_f=64.4,
            precip_prob_pct=30,
        )
        base.update(overrides)
        return WeatherSnapshot(**base)

    def test_temp_line_format_fahrenheit_only(self) -> None:
        line = self._s().temp_line
        assert "72°F" in line  # rounded from 71.6
        assert "°C" not in line
        assert "맑음" in line

    def test_range_line_includes_precip_fahrenheit(self) -> None:
        s = self._s(today_high_f=70.2, today_low_f=56.9, precip_prob_pct=45)
        line = s.range_line
        assert "최고 70°F" in line
        assert "최저 57°F" in line
        assert "°C" not in line
        assert "45%" in line


# ============================================================
# NWS fetch_alerts
# ============================================================
class TestFetchAlerts:
    def test_success_returns_alert_list_sorted_by_severity(self) -> None:
        payload = {
            "features": [
                {"properties": {
                    "event": "Flood Advisory", "headline": "minor flood",
                    "severity": "Minor", "ends": "2026-04-22T00:00:00+00:00",
                }},
                {"properties": {
                    "event": "Tornado Warning", "headline": "tornado spotted",
                    "severity": "Severe", "ends": "2026-04-21T20:00:00+00:00",
                }},
            ],
        }

        def handler(req: httpx.Request) -> httpx.Response:
            assert "api.weather.gov" in str(req.url)
            # NWS 는 User-Agent 필수.
            assert req.headers.get("user-agent", "").startswith("morning-companion")
            return httpx.Response(200, json=payload)

        alerts = fetch_alerts(0.0, 0.0, client=_mock(handler))
        assert len(alerts) == 2
        # Severe 가 첫 번째.
        assert alerts[0].event == "Tornado Warning"
        assert alerts[0].severity == "Severe"
        assert alerts[1].event == "Flood Advisory"

    def test_empty_features_returns_empty_list(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"features": []})
        assert fetch_alerts(0.0, 0.0, client=_mock(handler)) == []

    def test_http_error_returns_empty(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(503)
        assert fetch_alerts(0.0, 0.0, client=_mock(handler)) == []

    def test_malformed_expires_handled(self) -> None:
        payload = {
            "features": [{"properties": {
                "event": "X", "headline": "y", "severity": "Minor", "ends": "garbage",
            }}],
        }
        alerts = fetch_alerts(0.0, 0.0, client=_mock(lambda _r: httpx.Response(200, json=payload)))
        assert len(alerts) == 1
        assert alerts[0].ends_at is None

    def test_headline_truncated_to_200(self) -> None:
        long = "X" * 500
        payload = {"features": [{"properties": {
            "event": "Alert", "headline": long, "severity": "Minor",
        }}]}
        alerts = fetch_alerts(0.0, 0.0, client=_mock(lambda _r: httpx.Response(200, json=payload)))
        assert len(alerts[0].headline) == 200


# ============================================================
# message_builder 통합 (날씨 섹션 렌더)
# ============================================================
from execution.github_actions.news.message_builder import build_message
from execution.github_actions.news.sources import Category


class TestWeatherSectionInMessage:
    def _snap(self, **over) -> WeatherSnapshot:
        base = dict(
            temp_c=22, temp_f=72, weather_label="맑음",
            today_high_c=26, today_low_c=18,
            today_high_f=78, today_low_f=64,
            precip_prob_pct=10,
        )
        base.update(over)
        return WeatherSnapshot(**base)

    def test_weather_snapshot_renders_section_fahrenheit(self) -> None:
        msg = build_message(
            indices=[], selected={}, weather=self._snap(),
            now=datetime(2026, 4, 21, 11, 0, tzinfo=timezone.utc),
        )
        assert "오늘 날씨" in msg
        assert "72°F" in msg
        assert "맑음" in msg
        assert "최고 78°F" in msg
        # 섭씨는 렌더되지 않음.
        assert "°C" not in msg

    def test_alert_appended_as_blockquote(self) -> None:
        alert = WeatherAlert(
            event="Flood Warning",
            headline="Major flooding expected along Buffalo Bayou.",
            severity="Severe", ends_at=None,
        )
        msg = build_message(
            indices=[], selected={}, weather=self._snap(), alerts=[alert],
            now=datetime(2026, 4, 21, 11, 0, tzinfo=timezone.utc),
        )
        assert ">_" in msg
        assert "Flood Warning" in msg
        assert "Major flooding" in msg

    def test_advisory_rendered_as_blockquote(self) -> None:
        # 2026-04-21 hotfix: 날씨 조언 문장 blockquote.
        msg = build_message(
            indices=[], selected={}, weather=self._snap(),
            weather_advisory="오늘 72°F 로 쾌적합니다. 이번 주 후반 비 예보로 우산을 준비하세요.",
            now=datetime(2026, 4, 21, 11, 0, tzinfo=timezone.utc),
        )
        assert "이번 주 후반 비 예보" in msg
        assert "우산" in msg
        # blockquote italic 래퍼.
        assert ">_" in msg

    def test_no_weather_no_alert_no_advisory_hides_section(self) -> None:
        msg = build_message(
            indices=[], selected={}, weather=None, alerts=[], weather_advisory=None,
            now=datetime(2026, 4, 21, 11, 0, tzinfo=timezone.utc),
        )
        assert "오늘 날씨" not in msg

    def test_alerts_only_without_weather_still_renders(self) -> None:
        alert = WeatherAlert(event="Tornado Warning", headline="funnel cloud",
                              severity="Extreme", ends_at=None)
        msg = build_message(
            indices=[], selected={}, weather=None, alerts=[alert],
            now=datetime(2026, 4, 21, 11, 0, tzinfo=timezone.utc),
        )
        assert "오늘 날씨" in msg
        assert "Tornado Warning" in msg
