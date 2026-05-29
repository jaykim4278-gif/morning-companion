# W8a PR #1 — source_quality 결정론 테스트.
# docs/design/w8a-news-briefing-design.md §3 · §10.1 근거.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from execution.github_actions.news.source_quality import (
    HN_POINTS_BONUS,
    HN_POINTS_BONUS_THRESHOLD,
    TIER_SCORES,
    pick_best_deterministic,
    score_item,
)


@dataclass
class _FakeItem:
    source: str
    url: str = ""
    points: int = 0
    pub_date: datetime = datetime(2026, 4, 20, tzinfo=timezone.utc)


class TestScoreItem:
    def test_npr_is_tier1_top_score(self) -> None:
        # 2026-04-21 재편 이후 NPR 이 TOP tier1.
        assert score_item("npr_news") == TIER_SCORES[1]

    def test_env_local_source_is_tier2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # LOCAL 소스는 env(USER_LOCAL_RSS) 주입 — slug 'local_media', tier2.
        monkeypatch.setenv("USER_LOCAL_RSS", "[Local]|https://example.com/feed")
        assert score_item("local_media") == TIER_SCORES[2]

    def test_hn_under_threshold_no_bonus(self) -> None:
        # HN = tier4 기본, 499pt → 기본 점수 유지.
        assert score_item("hn", points=HN_POINTS_BONUS_THRESHOLD - 1) == TIER_SCORES[4]

    def test_hn_above_threshold_gets_bonus(self) -> None:
        expected = TIER_SCORES[4] + HN_POINTS_BONUS
        assert score_item("hn", points=HN_POINTS_BONUS_THRESHOLD) == expected
        assert score_item("hn", points=847) == expected

    def test_bonus_only_applies_to_hn(self) -> None:
        # NPR(tier1)에는 points 무관.
        assert score_item("npr_news", points=1000) == TIER_SCORES[1]

    def test_unknown_slug_scores_zero(self) -> None:
        assert score_item("nonexistent") == 0


class TestPickBestDeterministic:
    def test_tier_beats_timestamp(self) -> None:
        # NPR(T1) 이 CNBC(T3) 보다 점수 높음 — 최신성 무관하게 NPR 선택.
        newer_cnbc = _FakeItem("cnbc", pub_date=datetime(2026, 4, 20, 12, tzinfo=timezone.utc))
        older_npr = _FakeItem("npr_news", pub_date=datetime(2026, 4, 20, 6, tzinfo=timezone.utc))
        best = pick_best_deterministic([newer_cnbc, older_npr])
        assert best.source == "npr_news"

    def test_same_tier_prefers_newer(self) -> None:
        # NPR·BBC 둘 다 T1 — pub_date 최신 선택.
        older = _FakeItem("npr_news", pub_date=datetime(2026, 4, 20, 6, tzinfo=timezone.utc))
        newer = _FakeItem("bbc_us", pub_date=datetime(2026, 4, 20, 12, tzinfo=timezone.utc))
        best = pick_best_deterministic([older, newer])
        assert best.source == "bbc_us"

    def test_full_tie_falls_back_to_alphabetical_slug(self) -> None:
        # 동일 시각 + 동일 tier → slug 알파벳 순.
        same_time = datetime(2026, 4, 20, 6, tzinfo=timezone.utc)
        bbc = _FakeItem("bbc_us", pub_date=same_time)
        npr = _FakeItem("npr_news", pub_date=same_time)
        best = pick_best_deterministic([npr, bbc])
        assert best.source == "bbc_us"  # alphabetical

    def test_hn_bonus_elevates_rank(self) -> None:
        # HN 500pt+ = tier4(6) + bonus(1) = 7.  MarketWatch T3 = 7.  동점 → 최신.
        hn_hot = _FakeItem("hn", points=1000,
                            pub_date=datetime(2026, 4, 20, 12, tzinfo=timezone.utc))
        marketwatch = _FakeItem("marketwatch",
                                 pub_date=datetime(2026, 4, 20, 6, tzinfo=timezone.utc))
        best = pick_best_deterministic([marketwatch, hn_hot])
        assert best.source == "hn"

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError):
            pick_best_deterministic([])
