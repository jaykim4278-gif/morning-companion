# W8a PR #4 — L1 URL dedup 테스트 (정규화 + CRUD + UNIQUE 흡수).
# docs/design/w8a-news-briefing-design.md §6 · §10.1.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from execution.github_actions.news.dedup_url import (
    hash_url,
    is_sent,
    mark_all_sent,
    mark_sent,
    normalize_url,
)
from execution.github_actions.news.rss_fetcher import RssItem
from execution.github_actions.news.sources import Category


@dataclass
class FakeSupabase:
    sent_rows: dict[tuple[str, str], dict] = field(default_factory=dict)
    fail_select: bool = False
    fail_upsert: bool = False
    select_calls: list[tuple[str, Any]] = field(default_factory=list)
    upsert_calls: list[tuple[str, dict, str]] = field(default_factory=list)

    def select(self, table, filters=(), *, order=None, select_cols=None):
        self.select_calls.append((table, filters))
        if self.fail_select:
            raise RuntimeError("select down")
        # filters 가 eq.user_id, eq.url_hash 구조라고 가정.
        uid = next((f[1][3:] for f in filters if f[0] == "user_id"), None)
        h = next((f[1][3:] for f in filters if f[0] == "url_hash"), None)
        if (uid, h) in self.sent_rows:
            return [{"id": "row_id"}]
        return []

    def upsert(self, table, row, on_conflict, *, ignore_duplicates=False):
        self.upsert_calls.append((table, row, on_conflict))
        if self.fail_upsert:
            raise RuntimeError("upsert down")
        self.sent_rows[(row["user_id"], row["url_hash"])] = row


# ============================================================
class TestNormalizeUrl:
    def test_strips_utm_params(self) -> None:
        got = normalize_url("https://example.com/a?utm_source=x&utm_medium=y&id=1")
        assert got == "https://example.com/a?id=1"

    def test_strips_fbclid_gclid(self) -> None:
        got = normalize_url("https://example.com/a?fbclid=abc&gclid=xyz&id=1")
        assert got == "https://example.com/a?id=1"

    def test_lowercase_scheme_and_host(self) -> None:
        assert normalize_url("HTTPS://Example.COM/A") == "https://example.com/A"

    def test_removes_fragment(self) -> None:
        assert normalize_url("https://example.com/a#section") == "https://example.com/a"

    def test_root_path_preserved(self) -> None:
        assert normalize_url("https://example.com/") == "https://example.com/"

    def test_trailing_slash_removed_from_path(self) -> None:
        assert normalize_url("https://example.com/b/c/") == "https://example.com/b/c"

    def test_hash_is_deterministic_across_variations(self) -> None:
        # 추적 파라미터만 다르면 같은 해시가 나와야 함.
        h1 = hash_url("https://example.com/a?utm_source=x&id=1")
        h2 = hash_url("https://example.com/a?id=1&utm_campaign=y")
        assert h1 == h2


# ============================================================
class TestIsSent:
    def _item(self, url: str = "https://r.com/1") -> RssItem:
        return RssItem(
            source="reuters",
            url=url,
            url_hash=hash_url(url),
            title_en="t",
            pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
            category=Category.TOP,
        )

    def test_not_sent_returns_false(self) -> None:
        sb = FakeSupabase()
        assert is_sent("u1", "https://r.com/1", sb) is False

    def test_already_sent_returns_true(self) -> None:
        sb = FakeSupabase()
        sb.sent_rows[("u1", hash_url("https://r.com/1"))] = {"id": "x"}
        assert is_sent("u1", "https://r.com/1", sb) is True

    def test_select_failure_returns_false_conservative(self) -> None:
        # 조회 실패 시 False (재발송 가능 — 보수적으로 더 많이 발송 허용).
        sb = FakeSupabase(fail_select=True)
        assert is_sent("u1", "https://r.com/1", sb) is False


# ============================================================
class TestMarkSent:
    def _item(self, url="https://r.com/1", title_ko="번역") -> RssItem:
        return RssItem(
            source="reuters",
            url=url,
            url_hash=hash_url(url),
            title_en="Fed",
            pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
            category=Category.TOP,
            title_ko=title_ko,
        )

    def test_inserts_row(self) -> None:
        sb = FakeSupabase()
        mark_sent("u1", self._item(), cluster_topic="Fed rate", supabase=sb)
        assert len(sb.upsert_calls) == 1
        table, row, on_conflict = sb.upsert_calls[0]
        assert table == "news_sent_log"
        assert row["user_id"] == "u1"
        assert row["url_hash"] == hash_url("https://r.com/1")
        assert row["source"] == "reuters"
        assert row["cluster_topic"] == "Fed rate"
        assert row["title_ko"] == "번역"
        assert on_conflict == "user_id,url_hash"

    def test_upsert_failure_is_absorbed(self) -> None:
        # insert 실패해도 예외 안 터짐 (메시지는 이미 발송됨).
        sb = FakeSupabase(fail_upsert=True)
        mark_sent("u1", self._item(), cluster_topic=None, supabase=sb)
        # 예외 없이 통과


# ============================================================
class TestMarkAllSent:
    def test_inserts_each_item(self) -> None:
        sb = FakeSupabase()
        items = [
            RssItem("reuters", f"https://r.com/{i}", hash_url(f"https://r.com/{i}"),
                    "t", datetime(2026, 4, 20, tzinfo=timezone.utc), Category.TOP,
                    title_ko="t_ko")
            for i in range(3)
        ]
        mark_all_sent("u1", items, {items[0].url: "topic0", items[1].url: "topic1"}, sb)
        assert len(sb.upsert_calls) == 3
