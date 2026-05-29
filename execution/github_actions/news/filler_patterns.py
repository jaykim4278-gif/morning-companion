# 키워드 기반 filler 기사 사전 필터.
#
# LLM cluster_select 의 exclude 플래그에만 의존하면, 3 provider 전부 rate limit
# 맞았을 때 cluster_deterministic 폴백이 filter 를 못 함.
# 본 모듈은 LLM 가용성 무관하게 작동하는 보수적 필터 — 명백한 filler·개인 사건만 제거.
#
# 설계 원칙:
# - 보수적: 애매하면 "포함" (카테고리 완전 비는 상황 방지)
# - 카테고리별 규칙 (RESTAURANT/LOCAL 은 피드 특성상 filler 비율 높음)
# - 정규식 기반, 대소문자 무시
from __future__ import annotations

import re
from dataclasses import dataclass

from execution.github_actions.news.sources import Category


@dataclass(frozen=True)
class _Rule:
    name: str              # 로깅용
    pattern: re.Pattern    # compile 된 정규식
    category: Category | None = None  # None = 전체, 지정 시 해당 카테고리만


# 전체 카테고리 공통 — 팟캐스트·뉴스레터·리스트 안내 등 유형 filler.
_GLOBAL_PATTERNS: tuple[_Rule, ...] = (
    _Rule(
        "podcast_promo",
        re.compile(r"\b(podcast|newsletter)\b", re.IGNORECASE),
    ),
    _Rule(
        "listen_to",
        re.compile(r"\blisten to\b", re.IGNORECASE),
    ),
    _Rule(
        "top_N_list",
        # "Top 500 Restaurants", "Top 100 Chains", "Top 50 Companies"
        re.compile(
            r"\btop\s+\d{2,}\s+(restaurants?|chains?|companies|stocks?|qsr|brands?)\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "see_the_list",
        # "see the list" / "check out the full list" / "read the full episode notes" 등.
        re.compile(
            r"\b(see|check(\s+out)?|read|view)\s+(the\s+)?(full\s+)?(list|ranking|episode|rundown|roundup)\b",
            re.IGNORECASE,
        ),
    ),
    _Rule(
        "sponsored",
        re.compile(r"\b(sponsored|partner\s+content|advertorial)\b", re.IGNORECASE),
    ),
    _Rule(
        "daily_headlines_pointer",
        re.compile(r"\b(all the\s+headlines|today'?s?\s+top\s+stories)\b", re.IGNORECASE),
    ),
    # 라디오 단편 시리즈 패턴 차단 ("Episode: 1567 ..." / "The Engines of Our Ingenuity 1234").
    # RSS 에 흘러들어와 영문 폴백으로 노출되는 사고 방지.
    _Rule(
        "radio_episode_series",
        re.compile(
            r"^(episode\b\s*:?\s*\d+|the\s+engines\s+of\s+our\s+ingenuity\b)",
            re.IGNORECASE,
        ),
    ),
)


# RESTAURANT 특화 — 업계 trade publication 이 자주 내는 promo·LTO.
_RESTAURANT_PATTERNS: tuple[_Rule, ...] = (
    _Rule(
        "restaurant_lto",
        # "announces limited time offer", "unveils new menu item"
        re.compile(
            r"\b(limited time offer|LTO|unveils new menu|announces new flavor|"
            r"launches new drink)\b",
            re.IGNORECASE,
        ),
        category=Category.RESTAURANT,
    ),
)


# LOCAL 특화 — 개인 범죄·교통사고·화재·실종 등 거시성 없는 단발 사건.
_LOCAL_PATTERNS: tuple[_Rule, ...] = (
    _Rule(
        "personal_crime",
        # murder, assault, rape, robbery, shooting (단건)
        re.compile(
            r"\b(murder|murdered|homicide|stabbed|shot\s+(dead|and\s+killed)|"
            r"assault|rape|robbery|pleads\s+(not\s+)?guilty|indicted)\b",
            re.IGNORECASE,
        ),
        category=Category.LOCAL,
    ),
    _Rule(
        "accident_fire",
        # traffic accident, fire, crash
        re.compile(
            r"\b(crash|car\s+accident|house\s+fire|apartment\s+fire|structure\s+fire|"
            r"fatal\s+shooting|missing|kidnap)\b",
            re.IGNORECASE,
        ),
        category=Category.LOCAL,
    ),
    _Rule(
        "gossip_celeb",
        # 연예인 가십·개인사
        re.compile(
            r"\b(pleads?\s+not\s+guilty|charged\s+with\s+first-degree|"
            r"14-year-old|mutilating)\b",
            re.IGNORECASE,
        ),
        category=Category.LOCAL,
    ),
)


def is_filler(title_en: str, summary_en: str, category: Category) -> str | None:
    """filler 로 판정되면 규칙 이름 반환, 아니면 None.

    title + summary 를 합쳐 매칭.  빈 문자열은 False (empty 로 filler 아님).
    """
    text = f"{title_en or ''} {summary_en or ''}".strip()
    if not text:
        return None

    rules: list[_Rule] = list(_GLOBAL_PATTERNS)
    if category == Category.RESTAURANT:
        rules.extend(_RESTAURANT_PATTERNS)
    elif category == Category.LOCAL:
        rules.extend(_LOCAL_PATTERNS)

    for rule in rules:
        if rule.pattern.search(text):
            return rule.name
    return None


__all__ = ["is_filler"]
