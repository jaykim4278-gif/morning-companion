# W8a PR #2 — rss_fetcher 테스트 (feedparser mock + filter + 후보 선발).
# docs/design/w8a-news-briefing-design.md §1.3 · §10.1.
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from execution.github_actions.news.rss_fetcher import (
    MAX_AGE_HOURS,
    RssItem,
    _build_hn_summary,
    _extract_hn_comments,
    _extract_summary,
    _hash_url,
    _normalize_url_light,
    _strip_html,
    fetch_all,
    fetch_one,
    filter_already_sent,
)
from execution.github_actions.news.sources import CATALOG, Category, RssSource, by_slug


NOW = datetime(2026, 4, 20, 12, 0, tzinfo=timezone.utc)


def _make_entry(
    title: str,
    url: str,
    hours_ago: float = 1.0,
    summary: str = "",
) -> SimpleNamespace:
    pub = NOW - timedelta(hours=hours_ago)
    # feedparser 의 published_parsed 는 time.struct_time (mktime 가능).
    return SimpleNamespace(
        title=title,
        link=url,
        summary=summary,
        published_parsed=pub.timetuple(),
    )


class FakeFeedparser:
    """feedparser.parse 대체 — url → entries 매핑."""

    def __init__(self, by_url: dict[str, list[Any]] | Exception | None = None):
        self._by_url = by_url or {}

    def parse(self, url: str) -> SimpleNamespace:
        if isinstance(self._by_url, Exception):
            raise self._by_url
        entries = self._by_url.get(url, [])
        return SimpleNamespace(entries=entries, bozo=False)


class FakeSupabase:
    """news_sent_log 조회 mock — 넘겨준 hash set 을 'already sent' 로 응답."""

    def __init__(self, sent_hashes: set[str] | None = None):
        self.calls: list[tuple[str, Any]] = []
        self._sent = sent_hashes or set()

    def select(self, table, filters=(), *, order=None, select_cols=None):
        self.calls.append((table, filters))
        return [{"url_hash": h} for h in self._sent]


# ============================================================
class TestNormalizeUrl:
    def test_removes_utm(self) -> None:
        u = "https://example.com/a?utm_source=x&id=1"
        assert _normalize_url_light(u) == "https://example.com/a?id=1"

    def test_lowercase_host(self) -> None:
        assert _normalize_url_light("HTTPS://Example.COM/A") == "https://example.com/A"

    def test_removes_fragment(self) -> None:
        assert _normalize_url_light("https://example.com/a#top") == "https://example.com/a"

    def test_hash_is_stable(self) -> None:
        h1 = _hash_url("https://example.com/a?utm_source=x")
        h2 = _hash_url("https://example.com/a")
        assert h1 == h2


# ============================================================
class TestFetchOneFilters:
    def _src(self, slug="npr_news") -> RssSource:
        return by_slug(slug) or CATALOG[0]

    def test_skips_stale_older_than_24h(self) -> None:
        src = self._src()
        entries = [_make_entry("old", "https://r.com/old", hours_ago=MAX_AGE_HOURS + 1)]
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: entries}))
        assert items == []

    def test_accepts_fresh(self) -> None:
        src = self._src()
        entries = [_make_entry("fresh", "https://r.com/fresh", hours_ago=1)]
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: entries}))
        assert len(items) == 1
        assert items[0].title_en == "fresh"
        assert items[0].source == "npr_news"

    def test_hn_points_filter_requires_200(self) -> None:
        src = by_slug("hn")
        entries = [
            _make_entry("low-score (100 points)", "https://news.ycombinator.com/item?id=1"),
            _make_entry("hot (847 points)", "https://news.ycombinator.com/item?id=2"),
        ]
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: entries}))
        assert len(items) == 1
        assert items[0].points == 847

    def test_honors_max_items_per_fetch(self) -> None:
        src = self._src()
        # 10개 entry 줘도 source.max_items_per_fetch=5 만큼만.
        entries = [_make_entry(f"t{i}", f"https://r.com/{i}") for i in range(10)]
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: entries}))
        assert len(items) <= src.max_items_per_fetch

    def test_feedparser_crash_returns_empty_list(self) -> None:
        # 소스 단위 실패 격리 — 예외 안 터지고 빈 리스트.
        src = self._src()
        fp = FakeFeedparser(RuntimeError("network"))
        items = fetch_one(src, now=NOW, feedparser_module=fp)
        assert items == []

    def test_missing_title_or_link_skipped(self) -> None:
        src = self._src()
        bad = SimpleNamespace(
            title="", link="https://r.com/x",
            summary="", published_parsed=(NOW - timedelta(hours=1)).timetuple(),
        )
        good = _make_entry("ok", "https://r.com/ok")
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: [bad, good]}))
        assert len(items) == 1

    def test_missing_pub_date_skipped(self) -> None:
        src = self._src()
        no_pub = SimpleNamespace(title="x", link="https://r.com/x", summary="", published_parsed=None)
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: [no_pub]}))
        assert items == []


# ============================================================
class TestFilterAlreadySent:
    def test_empty_items_short_circuits(self) -> None:
        sb = FakeSupabase()
        assert filter_already_sent([], "u1", sb) == []
        assert sb.calls == []  # 쿼리 생략

    def _item(self, url: str) -> RssItem:
        return RssItem(
            source="npr_news",
            url=url,
            url_hash=_hash_url(url),
            title_en="t",
            pub_date=NOW,
            category=Category.TOP,
        )

    def test_removes_already_sent_hashes(self) -> None:
        it1 = self._item("https://a.com/1")
        it2 = self._item("https://a.com/2")
        sb = FakeSupabase(sent_hashes={it1.url_hash})
        result = filter_already_sent([it1, it2], "u1", sb)
        assert [r.url for r in result] == ["https://a.com/2"]

    def test_supabase_failure_returns_original(self) -> None:
        # 조회 실패 = 안전측으로 원본 반환 (발송 지속).
        class Crashing:
            def select(self, *a, **kw):
                raise RuntimeError("network")

        it = self._item("https://a.com/1")
        assert filter_already_sent([it], "u1", Crashing()) == [it]


# ============================================================
class TestFetchAll:
    def test_groups_by_category_and_limits_candidates(self) -> None:
        # Reuters + AP 동시 2건씩 → TOP 카테고리에 4건 → limit = 2+1=3 건
        reuters = by_slug("npr_news")
        ap = by_slug("bbc_us")
        entries_r = [
            _make_entry("r1", "https://r.com/1", hours_ago=1),
            _make_entry("r2", "https://r.com/2", hours_ago=2),
        ]
        entries_a = [
            _make_entry("a1", "https://a.com/1", hours_ago=1),
            _make_entry("a2", "https://a.com/2", hours_ago=2),
        ]
        fp = FakeFeedparser({reuters.url: entries_r, ap.url: entries_a})
        sb = FakeSupabase()
        result = fetch_all([reuters, ap], "u1", sb, now=NOW, feedparser_module=fp)
        assert Category.TOP in result
        # limit = CATEGORY_MAX_ITEMS[TOP]=2 + VOUCHER_EXTRA=1 = 3
        assert len(result[Category.TOP]) == 3

    def test_dedup_filters_before_selection(self) -> None:
        src = by_slug("npr_news")
        entries = [_make_entry("t1", "https://r.com/1"), _make_entry("t2", "https://r.com/2")]
        sb = FakeSupabase(sent_hashes={_hash_url("https://r.com/1")})
        fp = FakeFeedparser({src.url: entries})
        result = fetch_all([src], "u1", sb, now=NOW, feedparser_module=fp)
        urls = [it.url for it in result.get(Category.TOP, [])]
        assert urls == ["https://r.com/2"]


# ============================================================
# hotfix 2026-04-21 — HTML strip + summary 추출 + User-Agent contract.
# ============================================================
class TestStripHtml:
    def test_removes_basic_tags(self) -> None:
        assert _strip_html("<p>Hello <b>world</b>.</p>") == "Hello world ."

    def test_decodes_html_entities(self) -> None:
        assert _strip_html("A &amp; B &#x27;x&#x27;") == "A & B 'x'"

    def test_removes_script_block(self) -> None:
        result = _strip_html("Before<script>alert(1)</script>After")
        assert "alert" not in result
        assert "Before" in result and "After" in result

    def test_collapses_whitespace(self) -> None:
        assert _strip_html("a \n\t b   c") == "a b c"

    def test_empty_input_safe(self) -> None:
        assert _strip_html("") == ""
        assert _strip_html("   ") == ""


class TestExtractSummary:
    def test_uses_summary_field(self) -> None:
        e = SimpleNamespace(summary="<p>Core fact.</p>")
        assert _extract_summary(e) == "Core fact."

    def test_falls_back_to_description(self) -> None:
        e = SimpleNamespace(description="Fallback desc.")
        # summary attribute 없음 → description 사용.
        assert _extract_summary(e) == "Fallback desc."

    def test_truncates_long_summary(self) -> None:
        e = SimpleNamespace(summary="A " * 500)
        result = _extract_summary(e)
        assert len(result) <= 260
        assert result.endswith("…")

    def test_removes_hn_tail(self) -> None:
        e = SimpleNamespace(
            summary="Real content here. Points: 250 Comments URL: https://news.ycombinator.com/x Article URL: https://example.com",
        )
        result = _extract_summary(e)
        assert "Real content here" in result
        assert "Points:" not in result
        assert "Comments URL" not in result

    def test_empty_when_no_summary_fields(self) -> None:
        e = SimpleNamespace(title="x", link="y")
        assert _extract_summary(e) == ""


class TestFetchOneSummary:
    def _src(self) -> RssSource:
        return by_slug("npr_news")

    def test_populates_summary_en(self) -> None:
        src = self._src()
        entries = [_make_entry(
            "Title", "https://r.com/1", hours_ago=1,
            summary="<p>An <b>important</b> market update.</p>",
        )]
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: entries}))
        assert len(items) == 1
        assert items[0].summary_en == "An important market update."

    def test_summary_defaults_to_empty_when_missing(self) -> None:
        src = self._src()
        entries = [_make_entry("Title", "https://r.com/1", hours_ago=1, summary="")]
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: entries}))
        assert items[0].summary_en == ""

    def test_user_agent_accepted_by_real_feedparser_signature(self) -> None:
        # FakeFeedparser 는 request_headers 인자를 모르므로 TypeError 폴백이 작동해야 함.
        src = self._src()
        entries = [_make_entry("Title", "https://r.com/1", hours_ago=1)]
        # 이 호출이 TypeError 없이 성공해야 함.
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: entries}))
        assert len(items) == 1


# ============================================================
# HN 전용 요약 대체 — 2026-04-21 hotfix.
# ============================================================
class TestHnCommentsExtraction:
    def test_extracts_comment_count(self) -> None:
        e = SimpleNamespace(summary="Article URL: https://x.com Points: 250 # Comments: 85")
        assert _extract_hn_comments(e) == 85

    def test_no_comments_returns_zero(self) -> None:
        e = SimpleNamespace(summary="Article URL: https://x.com Points: 100")
        assert _extract_hn_comments(e) == 0

    def test_empty_summary_returns_zero(self) -> None:
        e = SimpleNamespace(summary="")
        assert _extract_hn_comments(e) == 0


class TestBuildHnSummary:
    def test_full_format_with_points_and_comments(self) -> None:
        assert _build_hn_summary(250, 85) == "해커뉴스 250점 추천글 · 댓글 85개"

    def test_points_only_omits_comments(self) -> None:
        assert _build_hn_summary(220, 0) == "해커뉴스 220점 추천글"

    def test_zero_zero_returns_empty(self) -> None:
        assert _build_hn_summary(0, 0) == ""


class TestFetchOneHnSummaryFallback:
    def test_hn_without_summary_gets_points_comments_fallback(self) -> None:
        # HN RSS 는 description 에 Article URL·Points·Comments 메타만 담아 옴.
        hn_src = by_slug("hn")
        hn_entry = _make_entry(
            "Some HN post title (250 points)",
            "https://hn.example/story",
            hours_ago=1,
            summary="Article URL: https://hn.example/story Comments URL: https://news.ycombinator.com/item?id=1 Points: 250 # Comments: 85",
        )
        items = fetch_one(hn_src, now=NOW, feedparser_module=FakeFeedparser({hn_src.url: hn_entry_sources(hn_entry)}))
        assert len(items) == 1
        it = items[0]
        # 메타 꼬리표 제거 후 summary_en 은 비어야 함 (기존 동작).
        assert it.summary_en == ""
        # summary_ko 가 한국어 placeholder 로 pre-populate.
        assert it.summary_ko is not None
        assert "250점" in it.summary_ko
        assert "85개" in it.summary_ko

    def test_non_hn_without_summary_leaves_summary_ko_none(self) -> None:
        # 다른 소스는 summary 없어도 summary_ko pre-populate 안 함.
        src = by_slug("npr_news")
        entry = _make_entry("Plain NPR", "https://npr.example/a", hours_ago=1, summary="")
        items = fetch_one(src, now=NOW, feedparser_module=FakeFeedparser({src.url: [entry]}))
        assert len(items) == 1
        assert items[0].summary_en == ""
        assert items[0].summary_ko is None


def hn_entry_sources(entry) -> list:
    """Helper — HN requires_points 필터 통과용 리스트로."""
    return [entry]
