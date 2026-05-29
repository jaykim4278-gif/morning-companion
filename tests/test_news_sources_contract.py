# W8a PR #1 — RSS CATALOG contract test.
# 카탈로그 = data only, 동작 없음.  URL·slug·tier 정합성만 검증.
# docs/design/w8a-news-briefing-design.md §1 · §10.1.
from __future__ import annotations

import re
from urllib.parse import urlparse

import pytest

from execution.github_actions.news.sources import (
    CATALOG,
    CATEGORY_HEADERS,
    CATEGORY_MAX_ITEMS,
    CATEGORY_ORDER,
    Category,
    by_category,
    by_slug,
)


class TestCatalogIntegrity:
    def test_slug_uniqueness(self) -> None:
        slugs = [s.slug for s in CATALOG]
        assert len(slugs) == len(set(slugs)), f"중복 slug: {slugs}"

    def test_url_uniqueness(self) -> None:
        urls = [s.url for s in CATALOG]
        assert len(urls) == len(set(urls)), f"중복 URL: {urls}"

    def test_slug_format(self) -> None:
        # DB source 컬럼에 들어가므로 소문자·숫자·언더스코어만 허용.
        pat = re.compile(r"^[a-z][a-z0-9_]*$")
        for s in CATALOG:
            assert pat.fullmatch(s.slug), f"잘못된 slug 형식: {s.slug}"

    def test_quality_tier_in_range(self) -> None:
        for s in CATALOG:
            assert 1 <= s.quality_tier <= 5, f"{s.slug}: tier {s.quality_tier} 범위 초과"

    def test_url_valid_https(self) -> None:
        for s in CATALOG:
            p = urlparse(s.url)
            assert p.scheme == "https", f"{s.slug}: non-https {s.url}"
            assert p.netloc, f"{s.slug}: netloc 없음"

    def test_hn_requires_points(self) -> None:
        # HN 만 requires_points > 0, 나머지는 0.
        for s in CATALOG:
            if s.slug == "hn":
                assert s.requires_points >= 200, "HN 은 200pt+ 필터 필수"
            else:
                assert s.requires_points == 0, f"{s.slug}: requires_points 는 HN 전용"


class TestCategoryMapping:
    def test_category_order_covers_headers(self) -> None:
        for cat in CATEGORY_ORDER:
            assert cat in CATEGORY_HEADERS, f"헤더 누락: {cat}"

    def test_non_rss_categories_excluded_from_max_items(self) -> None:
        # INDEX(yfinance) · WEATHER(Open-Meteo) 는 RSS 아님 → MAX_ITEMS 제외.
        _NON_RSS = {Category.INDEX, Category.WEATHER}
        for cat in Category:
            if cat in _NON_RSS:
                assert cat not in CATEGORY_MAX_ITEMS
            else:
                assert cat in CATEGORY_MAX_ITEMS
                assert CATEGORY_MAX_ITEMS[cat] >= 1

    def test_each_rss_category_has_at_least_one_source(self) -> None:
        # INDEX/WEATHER 는 RSS 아님, LOCAL 은 env(USER_LOCAL_RSS) 주입이라 미설정 시 빈 정상.
        _SKIP = {Category.INDEX, Category.WEATHER, Category.LOCAL}
        for cat in Category:
            if cat in _SKIP:
                continue
            srcs = by_category(cat)
            assert len(srcs) >= 1, f"{cat}: 소스 0개"

    def test_non_rss_categories_have_no_rss_source(self) -> None:
        # INDEX / WEATHER — 전용 API 경로 (yfinance · Open-Meteo) — CATALOG 미포함.
        assert by_category(Category.INDEX) == ()
        assert by_category(Category.WEATHER) == ()


class TestHelpers:
    def test_by_slug_found(self) -> None:
        # 2026-04-21 재편 이후 NPR 이 TOP 에 상주.
        assert by_slug("npr_news") is not None
        assert by_slug("npr_news").display_name == "[NPR]"

    def test_by_slug_missing_returns_none(self) -> None:
        assert by_slug("nonexistent") is None

    def test_removed_sources_absent(self) -> None:
        # 2026-04-21 재편 — micro/dead 피드 제거 확인 (회귀 방지).
        for removed in (
            "reuters", "ap",            # dead feeds
            "techcrunch",               # 개별 스타트업 펀딩·제품 위주 (techcrunch_ai 와 별개)
            "franchise_times",          # dead + micro
            "axios_tech",               # 2026-05-13 제거 — feed 가 실제로는 메인 (정치·금융 혼재)
        ):
            assert by_slug(removed) is None, f"{removed} 은 제거됐어야 함"

    def test_ai_focused_sources_present(self) -> None:
        # 2026-05-13 추가 — TECH 카테고리는 AI 전용 RSS 4종.
        for ai_slug in ("techcrunch_ai", "verge_ai", "wired_ai", "venturebeat_ai"):
            src = by_slug(ai_slug)
            assert src is not None, f"{ai_slug} 누락"
            assert src.category == Category.TECH
