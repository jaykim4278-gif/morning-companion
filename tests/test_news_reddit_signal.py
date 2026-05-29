# PR #88 — Reddit 24h top URLs 인기도 시그널 테스트.
from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from execution.github_actions.news.dedup_url import hash_url
from execution.github_actions.news.reddit_signal import (
    REDDIT_BOOST_SCORE,
    fetch_popular_url_hashes,
    fetch_subreddit_top_urls,
    reddit_boost,
)


def _mock_listing(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Reddit JSON listing 응답 형태 (children 배열)."""
    return {"data": {"children": [{"data": d} for d in items]}}


class TestFetchSubredditTopUrls:
    def test_external_links_extracted(self) -> None:
        body = _mock_listing([
            {"is_self": False, "url": "https://www.reuters.com/world/us/article-1"},
            {"is_self": False, "url": "https://www.cnbc.com/2026/05/03/rate-cut.html"},
        ])
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json=body),
        )
        result = fetch_subreddit_top_urls("news", transport=transport)
        assert hash_url("https://www.reuters.com/world/us/article-1") in result
        assert hash_url("https://www.cnbc.com/2026/05/03/rate-cut.html") in result
        assert len(result) == 2

    def test_self_post_skipped(self) -> None:
        body = _mock_listing([
            {"is_self": True, "url": "https://www.reddit.com/r/news/discussion"},
            {"is_self": False, "url": "https://www.bbc.com/news/world-us"},
        ])
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
        result = fetch_subreddit_top_urls("news", transport=transport)
        # self-post 는 제외 → BBC 1건만.
        assert result == {hash_url("https://www.bbc.com/news/world-us")}

    def test_reddit_internal_url_skipped(self) -> None:
        body = _mock_listing([
            {"is_self": False, "url": "https://www.reddit.com/r/news/comments/x1y2"},
        ])
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
        assert fetch_subreddit_top_urls("news", transport=transport) == set()

    def test_url_overridden_by_dest_preferred(self) -> None:
        # crosspost·미러 등에서 url_overridden_by_dest 가 진짜 외부 URL.
        body = _mock_listing([
            {
                "is_self": False,
                "url": "https://i.redd.it/preview.jpg",
                "url_overridden_by_dest": "https://www.npr.org/article-2",
            },
        ])
        transport = httpx.MockTransport(lambda req: httpx.Response(200, json=body))
        result = fetch_subreddit_top_urls("news", transport=transport)
        assert hash_url("https://www.npr.org/article-2") in result

    def test_404_returns_empty_set(self) -> None:
        # 비공개·삭제 서브레딧 → 404.  실패 격리 (다른 sub 와 main cron 영향 없음).
        transport = httpx.MockTransport(lambda req: httpx.Response(404, json={"error": 404}))
        assert fetch_subreddit_top_urls("nonexistentsub", transport=transport) == set()

    def test_invalid_json_returns_empty(self) -> None:
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text="not json"))
        assert fetch_subreddit_top_urls("news", transport=transport) == set()

    def test_user_agent_header_sent(self) -> None:
        seen_headers: dict[str, str] = {}

        def handler(req: httpx.Request) -> httpx.Response:
            seen_headers.update(req.headers)
            return httpx.Response(200, json=_mock_listing([]))

        transport = httpx.MockTransport(handler)
        fetch_subreddit_top_urls("news", transport=transport)
        # Reddit 비인증 API 는 User-Agent 필수 (없으면 429).
        assert "user-agent" in seen_headers
        assert "morning-companion" in seen_headers["user-agent"]


class TestFetchPopularUrlHashes:
    def test_combines_multiple_subreddits(self) -> None:
        # 두 서브레딧이 다른 URL 반환 → 합집합.
        seen_subs: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            sub = req.url.path.split("/")[2]  # /r/{sub}/top.json
            seen_subs.append(sub)
            mapping = {
                "news": _mock_listing([{"is_self": False, "url": "https://a.com/1"}]),
                "economy": _mock_listing([{"is_self": False, "url": "https://b.com/2"}]),
            }
            return httpx.Response(200, json=mapping.get(sub, _mock_listing([])))

        transport = httpx.MockTransport(handler)
        result = fetch_popular_url_hashes(["news", "economy"], transport=transport)
        assert hash_url("https://a.com/1") in result
        assert hash_url("https://b.com/2") in result
        assert {"news", "economy"}.issubset(set(seen_subs))

    def test_partial_failure_isolated(self) -> None:
        # 한 서브레딧 실패 → 다른 서브레딧 정상 결과 유지.
        def handler(req: httpx.Request) -> httpx.Response:
            sub = req.url.path.split("/")[2]
            if sub == "sports":
                return httpx.Response(503, text="server down")
            return httpx.Response(200, json=_mock_listing([
                {"is_self": False, "url": "https://x.com/1"},
            ]))

        transport = httpx.MockTransport(handler)
        result = fetch_popular_url_hashes(["news", "sports"], transport=transport)
        # sports 실패 무시, news 결과 1건 유지.
        assert hash_url("https://x.com/1") in result


class TestRedditBoost:
    def test_match_returns_boost_score(self) -> None:
        h = hash_url("https://x.com/1")
        assert reddit_boost(h, {h}) == REDDIT_BOOST_SCORE

    def test_no_match_returns_zero(self) -> None:
        assert reddit_boost(hash_url("https://x.com/1"), set()) == 0
        assert reddit_boost(hash_url("https://x.com/1"), {hash_url("https://y.com/2")}) == 0
