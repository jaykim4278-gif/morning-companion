# Reddit 인기도 시그널 — RSS 에 없는 "구독자 인기" 신호를 클러스터 가중치로 보강.
#
# 같은 URL 이 서브레딧 24h top 에 올라와 있으면 tier 기반 정렬에 +boost 부여.
# DEFAULT_SUBREDDITS 는 자유 교체 가능 — 배포자가 선호 카테고리에 맞춰 조정.
#
# API: Reddit JSON public endpoint (인증 불필요, IP 60 req/min 한도).
# https://www.reddit.com/r/{sub}/top.json?t=day&limit=50
from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import httpx

from execution.github_actions.news.dedup_url import hash_url

logger = logging.getLogger(__name__)


# 기본 서브레딧 — 배포자가 자유 교체 가능 (PR 또는 fetch 호출 인자로 override).
DEFAULT_SUBREDDITS: tuple[str, ...] = (
    "news",
    "worldnews",
    "economy",
    "personalfinance",
    "restaurateur",
    "smallbusiness",
    "technology",
)

REDDIT_API_BASE = "https://www.reddit.com"
USER_AGENT = "morning-companion/1.0"
HTTP_TIMEOUT = 10.0
DEFAULT_LIMIT_PER_SUB = 50
DEFAULT_TIME = "day"   # 24h top — 매일 06:01 cron 에 적합


def fetch_subreddit_top_urls(
    subreddit: str,
    *,
    limit: int = DEFAULT_LIMIT_PER_SUB,
    time_filter: str = DEFAULT_TIME,
    transport: httpx.BaseTransport | None = None,
) -> set[str]:
    """단일 서브레딧 top 글의 외부 링크 URL hash 집합.

    self-post (Reddit 내부 토론) 는 제외 — 우리 RSS 와 매칭 가능한 외부 링크만 의미 있음.
    실패 (404, timeout, 파싱 오류) 시 빈 set 반환 (다른 서브레딧·정상 cron 차단 금지).
    """
    url = f"{REDDIT_API_BASE}/r/{subreddit}/top.json"
    params = {"t": time_filter, "limit": str(limit)}
    headers = {"User-Agent": USER_AGENT}
    try:
        client_kwargs: dict[str, Any] = {"timeout": HTTP_TIMEOUT, "headers": headers}
        if transport is not None:
            client_kwargs["transport"] = transport
        with httpx.Client(**client_kwargs) as client:
            resp = client.get(url, params=params)
        if resp.status_code != 200:
            logger.warning("Reddit r/%s 응답 비정상: %s", subreddit, resp.status_code)
            return set()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Reddit r/%s fetch 실패: %s", subreddit, exc)
        return set()

    children = (data.get("data") or {}).get("children") or []
    out: set[str] = set()
    for child in children:
        post = child.get("data") or {}
        # is_self=True 면 self-post (Reddit 토론) — 외부 URL 없음.
        if post.get("is_self"):
            continue
        external = post.get("url_overridden_by_dest") or post.get("url")
        if not external or not isinstance(external, str):
            continue
        # Reddit 내부 URL 은 제외 (https://www.reddit.com/...).
        if external.startswith(REDDIT_API_BASE) or external.startswith("https://reddit.com"):
            continue
        try:
            out.add(hash_url(external))
        except Exception:  # pylint: disable=broad-except
            # normalize 실패한 URL 은 skip — 보안 측면 보수적.
            continue
    return out


def fetch_popular_url_hashes(
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    *,
    transport: httpx.BaseTransport | None = None,
) -> set[str]:
    """모든 사용자 관심 서브레딧의 24h top 외부 URL hash 합집합.

    부분 실패 허용 — 일부 서브레딧 실패해도 성공한 곳의 시그널 유지.
    """
    combined: set[str] = set()
    for sub in subreddits:
        combined |= fetch_subreddit_top_urls(sub, transport=transport)
    logger.info(
        "Reddit 인기도 시그널 — %d subreddits, total %d unique URL hashes",
        len(list(subreddits)), len(combined),
    )
    return combined


# 정렬 boost 점수 — score_item 의 T1=10 vs T5=5 격차(5점) 보다 작게 잡아
# Reddit 인기도가 매체 객관성 tier 를 완전히 뒤집지는 못하게 한다.
# 동률 시 Reddit 인기 항목이 우선 — sufficient.
REDDIT_BOOST_SCORE: int = 3


def reddit_boost(url_hash: str, popular_hashes: set[str]) -> int:
    """url_hash 가 popular set 에 있으면 +REDDIT_BOOST_SCORE, 없으면 0."""
    return REDDIT_BOOST_SCORE if url_hash in popular_hashes else 0


__all__ = [
    "DEFAULT_SUBREDDITS",
    "REDDIT_BOOST_SCORE",
    "fetch_subreddit_top_urls",
    "fetch_popular_url_hashes",
    "reddit_boost",
]
