"""런타임 env 에서 사용자 위치/페르소나를 읽는 단일 진입점.

이 모듈은 소스 코드에 어떤 식별 정보도 두지 않기 위해 존재한다.
모든 도시명·좌표·페르소나 문자열은 GitHub Secrets / 로컬 .env 로부터 주입된다.
필수 값이 빠지면 해당 기능 (날씨, 지역 뉴스, 페르소나 기반 코칭) 은 비활성된다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class UserCity:
    """위치 기반 기능에 필요한 최소 필드."""

    name_en: str            # 영문 표시명 (예: 메시지 헤더, NWS area)
    name_ko: str            # 한국어 표시명 (Telegram 메시지)
    lat: float
    lon: float
    timezone: str           # IANA tz
    nws_zone_prefix: str    # NWS area code prefix (예: 'TX' 또는 빈 문자열 → alerts 비활성)


def load_user_city() -> UserCity | None:
    """필수 env 모두 있을 때만 UserCity 반환.  아니면 None → 호출자가 섹션 skip."""
    name_en = os.environ.get("USER_CITY", "").strip()
    name_ko = os.environ.get("USER_CITY_KO", "").strip()
    lat_raw = os.environ.get("USER_LAT", "").strip()
    lon_raw = os.environ.get("USER_LON", "").strip()
    tz = os.environ.get("USER_TZ", "").strip()
    nws = os.environ.get("USER_NWS_ZONE_PREFIX", "").strip()
    if not (name_en and name_ko and lat_raw and lon_raw and tz):
        return None
    try:
        lat = float(lat_raw)
        lon = float(lon_raw)
    except ValueError:
        return None
    return UserCity(
        name_en=name_en,
        name_ko=name_ko,
        lat=lat,
        lon=lon,
        timezone=tz,
        nws_zone_prefix=nws,
    )


def load_user_persona_system_prompt() -> str | None:
    """LLM system prompt 로 주입할 페르소나 설명.

    USER_PERSONA_PROMPT env 가 있으면 그대로 반환, 없으면 None →
    호출자가 기본 (페르소나 미지정) 시스템 프롬프트 사용.
    """
    raw = os.environ.get("USER_PERSONA_PROMPT", "").strip()
    return raw or None


def load_user_interests() -> tuple[str, ...]:
    """뉴스 큐레이션 시 LLM hint 로 주입할 관심 키워드.

    USER_INTERESTS env (쉼표 구분).  없으면 빈 tuple → hint 비활성.
    """
    raw = os.environ.get("USER_INTERESTS", "").strip()
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


__all__ = [
    "UserCity",
    "load_user_city",
    "load_user_persona_system_prompt",
    "load_user_interests",
]
