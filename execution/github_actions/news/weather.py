# 뉴스 브리핑 날씨 섹션.
#
# 설계:
# - Open-Meteo 무료 API (키 불필요, 무제한) — 현재 온도·최고/최저·강수·코드
# - NWS (weather.gov) active alerts — 좌표 기반 active 기상경보
# - 두 호출 모두 실패 허용 (부분 실패 격리, 섹션 숨김 or 축약)
# - 좌표·timezone 은 호출 인자로 주입 (src/config/user_profile.py 의 USER_CITY env 경로)
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo forecast endpoint.  hourly 예보 생략, daily + current 만.
_OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"

# NWS active alerts — point 파라미터로 좌표 기반 필터.
_NWS_ALERTS_URL: str = "https://api.weather.gov/alerts/active"

# WMO weather code → 한국어 간단 라벨.
_WMO_CODE_KR: dict[int, str] = {
    0: "맑음", 1: "대체로 맑음", 2: "구름 조금", 3: "흐림",
    45: "안개", 48: "짙은 안개",
    51: "가벼운 이슬비", 53: "이슬비", 55: "강한 이슬비",
    61: "약한 비", 63: "비", 65: "강한 비",
    71: "약한 눈", 73: "눈", 75: "강한 눈",
    77: "싸락눈",
    80: "소나기", 81: "강한 소나기", 82: "매우 강한 소나기",
    85: "눈 소나기", 86: "강한 눈 소나기",
    95: "뇌우", 96: "뇌우 + 작은 우박", 99: "뇌우 + 큰 우박",
}

_TIMEOUT = 10.0


@dataclass(frozen=True)
class WeatherSnapshot:
    """오늘 날씨 요약 — 메시지 렌더용 최소 필드.  화씨 표시, 내부 C 유지."""
    temp_c: float              # 현재 기온 (℃, 내부 계산용)
    temp_f: float              # 현재 기온 (℉, 표시 전용)
    weather_label: str         # 한국어 라벨 ('맑음' 등)
    today_high_c: float        # 내부 계산용
    today_low_c: float
    today_high_f: float        # 표시 전용
    today_low_f: float
    precip_prob_pct: int       # 오늘 강수 확률 (0~100)

    @property
    def temp_line(self) -> str:
        """'62°F · 맑음' 형태 (화씨 전용)."""
        return f"{self.temp_f:.0f}°F · {self.weather_label}"

    @property
    def range_line(self) -> str:
        """'최고 70°F / 최저 56°F · 강수 79%'"""
        return (
            f"최고 {self.today_high_f:.0f}°F / 최저 {self.today_low_f:.0f}°F"
            f" · 강수 {self.precip_prob_pct}%"
        )


@dataclass(frozen=True)
class WeatherWeek:
    """7일 일간 예보 — 주간 날씨 코멘트용 데이터.

    days 는 오늘 포함 향후 7일 순서.  각 원소는 high/low/code/precip.
    """
    days: tuple["DailyForecast", ...]

    def summary_labels(self) -> tuple[str, ...]:
        """각 요일 한국어 라벨 (중복 제거 전)."""
        return tuple(d.weather_label for d in self.days)


@dataclass(frozen=True)
class DailyForecast:
    """하루 예보."""
    date_iso: str           # 'YYYY-MM-DD' (로컬 TZ 기준)
    high_f: float
    low_f: float
    weather_code: int
    weather_label: str      # 한국어
    precip_prob_pct: int
    precip_sum_mm: float    # 총 강수량


@dataclass(frozen=True)
class WeatherAlert:
    """NWS active alert 1건."""
    event: str       # 'Flood Warning', 'Severe Thunderstorm Watch'
    headline: str    # 짧은 요약 (영문 또는 번역)
    severity: str    # 'Severe', 'Moderate', 'Minor', 'Unknown'
    ends_at: datetime | None  # 종료 시각 (UTC).  None = 무기한.


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def fetch_weather(
    lat: float,
    lon: float,
    tz: str,
    *,
    client: httpx.Client | None = None,
) -> tuple[WeatherSnapshot, WeatherWeek] | None:
    """Open-Meteo 호출 → (snapshot, week) 튜플 반환.

    실패 시 None (호출자는 섹션 숨김).  부분 실패(week 구성 오류) 시에도 snapshot 있으면
    가능한 만큼 week 채움.
    """
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "timezone": tz,
        "current": "temperature_2m,weather_code",
        "daily": (
            "temperature_2m_max,temperature_2m_min,weather_code,"
            "precipitation_probability_max,precipitation_sum"
        ),
        "forecast_days": 7,
    }
    http = client or httpx.Client(timeout=_TIMEOUT)
    try:
        resp = http.get(_OPEN_METEO_URL, params=params)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Open-Meteo 호출 실패: %s", exc)
        return None

    try:
        current = data["current"]
        daily = data["daily"]
        temp_c = float(current["temperature_2m"])
        code = int(current["weather_code"])
        high_c = float(daily["temperature_2m_max"][0])
        low_c = float(daily["temperature_2m_min"][0])
        precip = int(daily["precipitation_probability_max"][0] or 0)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("Open-Meteo 응답 파싱 실패: %s  (data=%s)", exc, str(data)[:200])
        return None

    snapshot = WeatherSnapshot(
        temp_c=temp_c,
        temp_f=_c_to_f(temp_c),
        weather_label=_WMO_CODE_KR.get(code, f"코드 {code}"),
        today_high_c=high_c,
        today_low_c=low_c,
        today_high_f=_c_to_f(high_c),
        today_low_f=_c_to_f(low_c),
        precip_prob_pct=precip,
    )

    # 7일 예보 — 실패 격리 (snapshot 은 살린다).
    days: list[DailyForecast] = []
    try:
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        codes = daily.get("weather_code", [])
        precip_probs = daily.get("precipitation_probability_max", [])
        precip_sums = daily.get("precipitation_sum", [])
        n = min(len(dates), len(highs), len(lows), len(codes), 7)
        for i in range(n):
            c = int(codes[i])
            days.append(DailyForecast(
                date_iso=str(dates[i]),
                high_f=_c_to_f(float(highs[i])),
                low_f=_c_to_f(float(lows[i])),
                weather_code=c,
                weather_label=_WMO_CODE_KR.get(c, f"코드 {c}"),
                precip_prob_pct=int(precip_probs[i] or 0) if i < len(precip_probs) else 0,
                precip_sum_mm=float(precip_sums[i] or 0) if i < len(precip_sums) else 0.0,
            ))
    except (TypeError, ValueError, KeyError) as exc:
        logger.info("주간 예보 파싱 부분 실패 — 가능한 만큼 유지: %s", exc)

    return snapshot, WeatherWeek(days=tuple(days))


def fetch_alerts(
    point_lat: float,
    point_lon: float,
    *,
    client: httpx.Client | None = None,
) -> list[WeatherAlert]:
    """NWS active alerts — 지정 좌표에서 효력 있는 기상경보.

    실패 시 빈 리스트 (비 critical — 날씨 스냅샷은 이미 유효).
    headline 번역은 message_builder 단계에서 처리 (여기선 영문 유지).
    """
    params = {"point": f"{point_lat},{point_lon}"}
    http = client or httpx.Client(timeout=_TIMEOUT)
    try:
        resp = http.get(
            _NWS_ALERTS_URL,
            params=params,
            headers={
                "Accept": "application/geo+json",
                # NWS 는 User-Agent 필수 (contact 포함) — 미제공 시 403.
                "User-Agent": "morning-companion/1.0",
            },
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("NWS alerts 호출 실패 (경보 섹션 생략): %s", exc)
        return []

    alerts: list[WeatherAlert] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        event = str(props.get("event") or "Alert").strip()
        headline = str(props.get("headline") or props.get("description") or "").strip()
        severity = str(props.get("severity") or "Unknown").strip()
        expires_raw = props.get("ends") or props.get("expires")
        ends_at: datetime | None = None
        if expires_raw:
            try:
                ends_at = datetime.fromisoformat(expires_raw.replace("Z", "+00:00"))
                if ends_at.tzinfo is None:
                    ends_at = ends_at.replace(tzinfo=timezone.utc)
            except ValueError:
                ends_at = None
        alerts.append(WeatherAlert(
            event=event,
            headline=headline[:200],  # 길면 자름
            severity=severity,
            ends_at=ends_at,
        ))
    # severity 우선 (Severe > Moderate > Minor > Unknown) 정렬.
    severity_rank = {"Extreme": 0, "Severe": 1, "Moderate": 2, "Minor": 3, "Unknown": 4}
    alerts.sort(key=lambda a: severity_rank.get(a.severity, 5))
    return alerts


__all__ = [
    "WeatherSnapshot",
    "WeatherWeek",
    "DailyForecast",
    "WeatherAlert",
    "fetch_weather",
    "fetch_alerts",
]
