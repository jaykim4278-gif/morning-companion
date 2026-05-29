# W8a PR #2 — RSS 수집 + L1 URL dedup 사전 필터 + 카테고리당 후보 2~3건 선발.
# docs/design/w8a-news-briefing-design.md §1.3 근거.
#
# 흐름:
#   1) 각 소스 feedparser.parse → 최근 N 항목 (max_items_per_fetch)
#   2) HN 의 경우 requires_points 필터
#   3) pub_date max_age (24h) 초과 항목 제외
#   4) 모든 아이템 url_hash 계산 (dedup_url.normalize_url → SHA256)
#   5) news_sent_log 배치 조회 → 이미 발송된 url_hash 제외
#   6) 카테고리별 그룹핑 → quality_tier + pub_date 정렬 → N+1 건 선발
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from execution.github_actions.news.sources import (
    CATEGORY_MAX_ITEMS,
    Category,
    RssSource,
)

logger = logging.getLogger(__name__)

# 뉴스 신선도 — 24시간 이전 항목은 'stale' 로 간주 (매일 새로운 뉴스 강제).
MAX_AGE_HOURS: int = 24

# 카테고리별 voucher 여유분 (primary + voucher 2개 = 3건 총합 원칙).
VOUCHER_EXTRA: int = 1


@dataclass
class RssItem:
    source: str                  # RssSource.slug
    url: str                     # 원본 URL (정규화 전)
    url_hash: str                # SHA256(normalize_url(url)) — L1 dedup 키
    title_en: str                # 영문 제목
    pub_date: datetime           # UTC aware
    category: Category
    points: int = 0              # HN 전용 (그 외 0)
    title_ko: str | None = None  # 번역 후 (PR #3 에서 채움)
    summary_en: str = ""         # RSS <description>/<summary> — HTML 제거·정규화된 원문 발췌
    summary_ko: str | None = None  # summary 번역 후 (best-effort, 실패 시 None 허용)
    translation_failed: bool = False  # title 번역 실패 → title_ko = title_en 폴백 표시


# ============================================================
# PostgREST 최소 인터페이스 — 테스트에서 Fake 주입.
# _supabase.SupabaseClient 가 이 형태를 만족 (Structural typing).
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


# ============================================================
# URL 정규화·해시 — PR #4 에서 dedup_url 로 정식 이전됨.
# 여기서는 얇은 re-export 로 circular import 방지 (news 모듈 내부 동일 계약).
# ============================================================
def _hash_url(url: str) -> str:
    """dedup_url.hash_url 로 위임.  정규화 규칙 단일 진입점 유지."""
    from execution.github_actions.news.dedup_url import hash_url
    return hash_url(url)


def _normalize_url_light(url: str) -> str:
    """하위 호환 shim — 테스트에서 직접 import 하는 경로 유지."""
    from execution.github_actions.news.dedup_url import normalize_url
    return normalize_url(url)


# ============================================================
# 단일 소스 fetch — 실패 격리 (빈 리스트 반환).
# ============================================================
def _parse_pub_date(entry: Any) -> datetime | None:
    """feedparser entry 에서 UTC datetime 추출.  parsed_published 우선, 실패 시 None.

    feedparser 는 published_parsed 를 **UTC struct_time** 으로 정규화해 전달한다
    (https://feedparser.readthedocs.io/en/latest/date-parsing.html).
    따라서 calendar.timegm 으로 UTC 해석해야 함 — time.mktime 은 local tz 로 잘못 해석.
    """
    pp = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if pp is None:
        return None
    try:
        import calendar
        ts = calendar.timegm(pp)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except (OverflowError, ValueError):
        return None


# RSS description/summary 원문 발췌 상한 — LLM 비용·메시지 길이 균형.
_MAX_SUMMARY_LEN = 260

# GH Actions IP 대상 차단을 피하기 위한 UA (일부 피드가 feedparser 기본값을 거절).
_USER_AGENT = (
    "Mozilla/5.0 (compatible; morning-companion/1.0)"
)


def _strip_html(text: str) -> str:
    """RSS summary 의 HTML 태그·HTML entity 정규화.

    feedparser 는 일부 피드에서 raw HTML 을 그대로 반환 — <p>, <a>, <img>, &amp; 등.
    경량 정규식 기반 (lxml/bs4 의존 회피, 정확도보다 견고성 우선).
    """
    import html
    import re

    if not text:
        return ""
    # script/style 블록 통째 제거.
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # 모든 HTML 태그 제거.
    text = re.sub(r"<[^>]+>", " ", text)
    # HTML entity (&amp; &#x27; 등) decode.
    text = html.unescape(text)
    # 공백 정규화.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_summary(entry: Any) -> str:
    """feedparser entry → 정규화된 summary 발췌 (최대 _MAX_SUMMARY_LEN).

    우선순위: entry.summary → entry.description → entry.subtitle.
    HN 피드의 'Points: xxx | Comments: xxx' 꼬리표는 제거.
    """
    import re

    raw = ""
    for attr in ("summary", "description", "subtitle"):
        val = getattr(entry, attr, None)
        if val:
            raw = str(val)
            break
    cleaned = _strip_html(raw)
    # HN 전용 꼬리표 제거 (Points: N · Comments: M · Article URL: ...).
    cleaned = re.sub(
        r"(Article URL|Comments URL|Points|# Comments):\s*\S.*?(?=(Article URL|Comments URL|Points|# Comments|$))",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if len(cleaned) <= _MAX_SUMMARY_LEN:
        return cleaned
    return cleaned[: _MAX_SUMMARY_LEN - 1].rstrip() + "…"


def _extract_hn_points(entry: Any) -> int:
    """HN RSS 는 title 에 '(123 points)' 접미사 또는 description 에 Points: 포함."""
    import re

    title = str(getattr(entry, "title", "") or "")
    # hnrss.org 포맷: '(123 points)' 접미사
    m = re.search(r"\((\d+)\s*points?\)", title)
    if m:
        return int(m.group(1))
    # description 폴백
    desc = str(getattr(entry, "summary", "") or "")
    m = re.search(r"Points:\s*(\d+)", desc)
    if m:
        return int(m.group(1))
    return 0


def _extract_hn_comments(entry: Any) -> int:
    """HN RSS description 의 '# Comments: 85' 에서 댓글 수 추출.  없으면 0."""
    import re

    desc = str(getattr(entry, "summary", "") or "")
    m = re.search(r"#\s*Comments:\s*(\d+)", desc, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _build_hn_summary(points: int, comments: int) -> str:
    """HN RSS 에 본문 요약이 없어 _extract_summary 가 빈 문자열을 돌려주는 경우용 대체.

    예: '해커뉴스 250점 · 댓글 85개'.  points 0 이면 빈 문자열 (message_builder 는 미렌더).
    """
    if points <= 0 and comments <= 0:
        return ""
    parts = [f"해커뉴스 {points}점 추천글"] if points > 0 else ["해커뉴스 추천글"]
    if comments > 0:
        parts.append(f"댓글 {comments}개")
    return " · ".join(parts)


def fetch_one(
    source: RssSource,
    *,
    now: datetime,
    feedparser_module: Any = None,
) -> list[RssItem]:
    """단일 소스 RSS fetch.

    Args:
        source: RssSource 카탈로그 엔트리.
        now: 시각 기준점 (UTC, max_age 계산).
        feedparser_module: 테스트 주입용 — 기본은 실 feedparser.

    Returns:
        RssItem 리스트.  fetch·파싱 실패·항목 0건 모두 빈 리스트 (소스 단위 실패 격리).
    """
    if feedparser_module is None:
        import feedparser as feedparser_module  # type: ignore[no-redef]

    try:
        # 실 feedparser 6.x 는 request_headers dict 를 받아 UA 지정 가능.
        # 테스트 주입(FakeFeedparser) 은 단순 parse(url) 만 구현하므로 TypeError 폴백.
        try:
            feed = feedparser_module.parse(
                source.url,
                request_headers={"User-Agent": _USER_AGENT},
            )
        except TypeError:
            feed = feedparser_module.parse(source.url)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("RSS fetch 실패 %s: %s", source.slug, exc)
        return []

    # feedparser bozo 는 HTTP 오류·파싱 일부 실패 경고 — entries 가 있으면 계속 진행.
    entries = getattr(feed, "entries", [])
    if not entries:
        logger.info("RSS 빈 결과 %s (bozo=%s)", source.slug, getattr(feed, "bozo", None))
        return []

    cutoff = now - timedelta(hours=MAX_AGE_HOURS)
    items: list[RssItem] = []
    for entry in entries[: source.max_items_per_fetch]:
        url = str(getattr(entry, "link", "") or "").strip()
        title = str(getattr(entry, "title", "") or "").strip()
        if not url or not title:
            continue

        pub = _parse_pub_date(entry)
        if pub is None:
            # pub_date 없는 항목은 신선도 판정 불가 → skip (RSS 표준 필수 필드).
            continue
        if pub < cutoff:
            continue

        points = _extract_hn_points(entry) if source.slug == "hn" else 0
        if source.requires_points and points < source.requires_points:
            continue

        summary = _extract_summary(entry)
        # HN RSS 는 본문 요약 없이 Points·Comments·URL 만 담아 오는 경우가 대부분
        # → _extract_summary 가 메타 꼬리표 제거 후 빈 문자열 반환. 사용자 가치를 위해
        # points·comments 기반 한국어 대체 요약 생성 (2026-04-21 hotfix).
        summary_ko_pre: str | None = None
        if source.slug == "hn" and not summary:
            # 이미 한국어 → summary_ko 에 직접 세팅, translator 호출 건너뜀.
            summary_ko_pre = _build_hn_summary(points, _extract_hn_comments(entry)) or None

        items.append(RssItem(
            source=source.slug,
            url=url,
            url_hash=_hash_url(url),
            title_en=title,
            pub_date=pub,
            category=source.category,
            points=points,
            summary_en=summary,
            summary_ko=summary_ko_pre,
        ))
    return items


# ============================================================
# 배치 dedup — news_sent_log 단일 쿼리 (N+1 방지).
# ============================================================
def filter_already_sent(
    items: list[RssItem],
    user_id: str,
    supabase: _SupabaseLike,
) -> list[RssItem]:
    """PostgREST: news_sent_log?url_hash=in.(...)&user_id=eq.<uid> → 존재하는 url_hash 제거.

    items 가 비어있으면 쿼리 생략.  Supabase 조회 실패 시 items 그대로 반환 (안전 측).
    """
    if not items:
        return items
    hashes = [it.url_hash for it in items]
    in_clause = f"in.({','.join(hashes)})"
    try:
        rows = supabase.select(
            "news_sent_log",
            filters=[
                ("user_id", f"eq.{user_id}"),
                ("url_hash", in_clause),
            ],
            select_cols="url_hash",
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("news_sent_log 조회 실패 — dedup 생략: %s", exc)
        return items
    sent_hashes = {r["url_hash"] for r in rows if r.get("url_hash")}
    return [it for it in items if it.url_hash not in sent_hashes]


# ============================================================
# 카테고리 그룹핑 + 후보 N+1 건 선발.
# ============================================================
def _select_candidates(
    items: list[RssItem],
    category_max: dict[Category, int],
    extra: int = VOUCHER_EXTRA,
    *,
    popular_url_hashes: set[str] | None = None,
) -> dict[Category, list[RssItem]]:
    """카테고리별 quality_tier + Reddit 인기도 + pub_date 정렬 → filler 필터 → 상위 (max + extra) 건.

    2026-04-21 — LLM 가용성 무관 filler 키워드 필터 추가.
    2026-05-03 (PR #88) — Reddit 24h top URLs 매칭 시 +REDDIT_BOOST_SCORE 정렬 가산.
    LLM cluster exclude 는 Gemini/Groq/OpenRouter 전부 rate limit 맞으면 작동 안 함 →
    키워드 필터가 최종 안전망 (팟캐스트·순위 나열·개인 범죄 기본 제거).

    정렬 키 (모두 내림차순):
      1) score_item (T1~T5 매체 객관성) + reddit_boost (사용자 인기 매칭)
      2) pub_date timestamp
      3) source slug (alphabetic, 동률 안정성)
    """
    from execution.github_actions.news.filler_patterns import is_filler
    from execution.github_actions.news.reddit_signal import reddit_boost
    from execution.github_actions.news.source_quality import score_item

    popular = popular_url_hashes or set()

    grouped: dict[Category, list[RssItem]] = {}
    for it in items:
        # filler 사전 제거 — 카테고리별 패턴 매칭.
        rule = is_filler(it.title_en, it.summary_en, it.category)
        if rule is not None:
            logger.info(
                "filler 제거 (%s) — source=%s, title=%r",
                rule, it.source, it.title_en[:80],
            )
            continue
        grouped.setdefault(it.category, []).append(it)

    selected: dict[Category, list[RssItem]] = {}
    for cat, lst in grouped.items():
        lst_sorted = sorted(
            lst,
            key=lambda x: (
                -(score_item(x.source, x.points) + reddit_boost(x.url_hash, popular)),
                -x.pub_date.timestamp(),
                x.source,
            ),
        )
        limit = category_max.get(cat, 2) + extra
        selected[cat] = lst_sorted[:limit]
    return selected


# ============================================================
# 전체 fetch pipeline.
# ============================================================
def fetch_all(
    catalog: Sequence[RssSource],
    user_id: str,
    supabase: _SupabaseLike,
    *,
    now: datetime,
    feedparser_module: Any = None,
    popular_url_hashes: set[str] | None = None,
) -> dict[Category, list[RssItem]]:
    """모든 소스 → L1 URL dedup 사전 필터 → 카테고리별 후보 N+1 건 선발.

    popular_url_hashes — Reddit 24h top URLs 의 hash 집합 (선택).  주어지면
    _select_candidates 정렬 시 매칭 항목에 +REDDIT_BOOST_SCORE 가산하여 사용자가
    "많이 본 뉴스" 가 우선 노출되게 한다.  None 이면 기존 tier 정렬만.
    """
    all_items: list[RssItem] = []
    for source in catalog:
        items = fetch_one(source, now=now, feedparser_module=feedparser_module)
        all_items.extend(items)

    # L1 dedup — news_sent_log 배치 조회.
    fresh = filter_already_sent(all_items, user_id, supabase)
    return _select_candidates(
        fresh, CATEGORY_MAX_ITEMS, popular_url_hashes=popular_url_hashes,
    )


__all__ = [
    "RssItem",
    "MAX_AGE_HOURS",
    "VOUCHER_EXTRA",
    "fetch_one",
    "fetch_all",
    "filter_already_sent",
]
