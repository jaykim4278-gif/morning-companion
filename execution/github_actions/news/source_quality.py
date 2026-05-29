# W8a — 매체 객관성 Tier 점수 + 결정론적 best 선택 (LLM 클러스터링 폴백).
# docs/design/w8a-news-briefing-design.md §3 근거.
# LLM 이 살아있을 때는 cluster_select.py 의 의미 기반 선택이 우선,
# LLM 실패 시 여기의 점수·최신성·slug 3단 정렬로 결정론적 선택.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from execution.github_actions.news.sources import RssSource, by_slug

# T1=최객관(통신사) → T5=커뮤니티·지역방송.
# 외부 표준(AllSides / Ad Fontes) 절대 평가가 아닌, 본 프로젝트 컨텍스트(일일 브리핑·사실 위주)
# 에서의 상대 점수 — 변경 시 design §3.2 근거 표 동시 갱신 필요.
TIER_SCORES: dict[int, int] = {
    1: 10,
    2: 8,
    3: 7,
    4: 6,
    5: 5,
}

# HN 전용 popularity bonus — 500pt 이상 항목은 T5(5점) + 1 = 6점 (T4 수준 인정).
# 다른 RSS 는 view count 미제공이라 본 보정은 HN 에만 적용.
HN_POINTS_BONUS_THRESHOLD: int = 500
HN_POINTS_BONUS: int = 1
HN_SLUG: str = "hn"


def score_item(source_slug: str, points: int = 0) -> int:
    """slug 로 CATALOG 조회 → tier 점수 반환.  HN + 500pt+ 는 +1 보정.

    unknown slug → 0 (LLM 폴백 정렬에서 최하위로 밀려남, 메시지 배제 아님).
    """
    src = by_slug(source_slug)
    if src is None:
        return 0
    base = TIER_SCORES.get(src.quality_tier, 0)
    if source_slug == HN_SLUG and points >= HN_POINTS_BONUS_THRESHOLD:
        return base + HN_POINTS_BONUS
    return base


@dataclass(frozen=True)
class _SortableItem:
    """정렬 전용 경량 뷰 — RssItem 실 type 은 PR #2 에서 추가되므로 여기선 Protocol 대신 Any 로 받음."""
    source: str
    url: str
    points: int
    pub_date: datetime


def pick_best_deterministic(candidates: list):
    """LLM 클러스터링 실패 시 결정론적 best 선택.

    정렬 키 (내림차순):
        1. score_item(source_slug, points)  — T1 > T5
        2. pub_date                          — 최신 우선
        3. source slug alphabetical           — 완전 동점 방지

    최상위 1건 반환.  빈 리스트 시 ValueError.
    """
    if not candidates:
        raise ValueError("pick_best_deterministic requires non-empty candidates")
    return sorted(
        candidates,
        key=lambda it: (
            -score_item(getattr(it, "source"), getattr(it, "points", 0)),
            -getattr(it, "pub_date").timestamp(),
            getattr(it, "source"),
        ),
    )[0]


__all__ = [
    "TIER_SCORES",
    "HN_POINTS_BONUS_THRESHOLD",
    "HN_POINTS_BONUS",
    "score_item",
    "pick_best_deterministic",
]
