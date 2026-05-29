# W8a PR #4 — L1 URL 영구 dedup (news_sent_log).
# docs/design/w8a-news-briefing-design.md §6 근거.
#
# 계층:
# - L1 (여기): 같은 URL 재발송 금지.  30일 보관 (pg_cron 자동 삭제).
# - L2 (cluster_select.py): 한 메시지 내 같은 사건 중복 1건만 — LLM 클러스터링.
#
# URL 정규화 규칙:
# - scheme/host 소문자
# - 추적 파라미터(utm_*, fbclid, gclid, ref, _ga, mc_cid 등) 제거
# - fragment (#) 제거
# - trailing slash 제거 (path="/" 는 유지)
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from execution.github_actions.news.rss_fetcher import RssItem

logger = logging.getLogger(__name__)


# 추적·ads 파라미터 — 동일 컨텐츠에 여러 URL 변형이 붙는 주된 원인.
_REMOVE_QUERY_KEYS: frozenset[str] = frozenset({
    # UTM — Google Analytics 표준
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    # 광고 클릭 식별자
    "fbclid", "gclid", "msclkid", "yclid", "dclid",
    # referral
    "ref", "ref_src", "ref_url", "referrer",
    # mailchimp
    "mc_cid", "mc_eid",
    # Google Analytics
    "_ga", "_gl",
})


def normalize_url(url: str) -> str:
    """URL 정규화 — dedup 키의 기반.

    예:
      HTTPS://Example.COM/a?id=1&utm_source=x#top  →  https://example.com/a?id=1
      https://example.com/                          →  https://example.com/
      https://example.com/b/                        →  https://example.com/b
    """
    try:
        p = urlparse(url.strip())
    except Exception:  # pylint: disable=broad-except
        # 파싱 실패 시 원본 그대로 (해시 기준은 동일 입력 → 동일 해시 유지).
        return url.strip()

    scheme = (p.scheme or "https").lower()
    netloc = p.netloc.lower()

    # path: 마지막 '/' 제거 (단 루트는 유지).
    path = p.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # query: 트래킹 키 제거, 나머지는 순서 유지.
    kept_pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
                   if k not in _REMOVE_QUERY_KEYS]
    query = urlencode(kept_pairs)

    return urlunparse((scheme, netloc, path, "", query, ""))  # fragment 제거


def hash_url(url: str) -> str:
    """SHA256(normalize_url).  news_sent_log.url_hash 값."""
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()


# ============================================================
# Supabase Protocol (structural) — 테스트에서 Fake 주입.
# ============================================================
class _SupabaseLike(Protocol):
    def select(
        self,
        table: str,
        filters: Sequence[tuple[str, str]] = ...,
        *,
        order: str | None = ...,
        select_cols: str | None = ...,
    ) -> list[dict[str, Any]]: ...

    def upsert(
        self,
        table: str,
        row: dict[str, Any],
        on_conflict: str,
        *,
        ignore_duplicates: bool = False,
    ) -> None: ...


# ============================================================
# CRUD
# ============================================================
def is_sent(user_id: str, url: str, supabase: _SupabaseLike) -> bool:
    """url_hash 가 news_sent_log 에 존재하는지.  조회 실패 시 False (보수적 — 재발송 감수)."""
    try:
        rows = supabase.select(
            "news_sent_log",
            filters=[
                ("user_id", f"eq.{user_id}"),
                ("url_hash", f"eq.{hash_url(url)}"),
            ],
            select_cols="id",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("is_sent 조회 실패 — False 반환 (재발송 가능): %s", exc)
        return False
    return bool(rows)


def mark_sent(
    user_id: str,
    item: RssItem,
    cluster_topic: str | None,
    supabase: _SupabaseLike,
) -> None:
    """발송 완료 후 news_sent_log insert.  UNIQUE(user_id, url_hash) 충돌은 흡수.

    item.url_hash 가 이미 있다면 rss_fetcher._hash_url 로 계산된 것일 텐데,
    정식 normalize_url 로 재계산 (경량 버전과의 동등성 보장).
    """
    row = {
        "user_id": user_id,
        "url_hash": hash_url(item.url),
        "source": item.source,
        "cluster_topic": cluster_topic,
        "title_ko": item.title_ko,
    }
    try:
        # ignore-duplicates: news_sent_log 도 append-only (INSERT 권한만으로 동작).
        supabase.upsert(
            "news_sent_log",
            row,
            on_conflict="user_id,url_hash",
            ignore_duplicates=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        # 이미 발송된 URL (UNIQUE 위반) 또는 네트워크 장애 — 메시지 발송은 이미 완료됐으므로 경고만.
        logger.warning("news_sent_log insert 실패 — %s: %s", item.url, exc)


def mark_all_sent(
    user_id: str,
    items: list[RssItem],
    cluster_topic_by_url: dict[str, str | None],
    supabase: _SupabaseLike,
) -> None:
    """메시지 발송 성공 후 호출 — 모든 아이템을 news_sent_log 에 기록."""
    for item in items:
        mark_sent(user_id, item, cluster_topic_by_url.get(item.url), supabase)


__all__ = [
    "normalize_url",
    "hash_url",
    "is_sent",
    "mark_sent",
    "mark_all_sent",
]
