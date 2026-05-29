# W8a — 뉴스 브리핑 메인 entrypoint (매 30분 cron).
# blueprint §2 W8 · docs/design/w8a-news-briefing-design.md §8.
#
# 흐름:
#   1) _schedule.should_send_news_briefing gate (briefing_time+1, 오늘 미발송)
#   2) RSS 전 소스 fetch → L1 URL dedup 필터 → 카테고리별 후보 N+1 선발
#   3) yfinance 지수 fetch (부분 실패 허용)
#   4) LLM 클러스터링 + best 선택 (실패 시 결정론 폴백)
#   5) 번역 (9중 retry + voucher 폴백) → title_ko 채움
#   6) MarkdownV2 메시지 조립
#   7) Telegram 발송
#   8) news_sent_log mark_all_sent + user_preferences.last_news_sent_at
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from execution.ai_client.fallback_chain import Provider
from execution.github_actions._schedule import (
    UserSchedule,
    fetch_user_schedule,
    mark_news_sent,
    resolve_chat_id,
    should_send_news_briefing,
)
from execution.github_actions._supabase import SupabaseClient
from execution.github_actions._telegram import HttpTelegramSender, TelegramSendError
from execution.github_actions.news.cluster_select import cluster_and_select
from execution.github_actions.news.dedup_url import mark_all_sent
from execution.github_actions.news.market_indices import fetch_indices
from execution.github_actions.news.message_builder import build_message
from execution.github_actions.news.reddit_signal import fetch_popular_url_hashes
from execution.github_actions.news.rss_fetcher import RssItem, fetch_all
from execution.github_actions.news.sources import (
    CATALOG,
    CATEGORY_MAX_ITEMS,
    Category,
)
from execution.github_actions.news.translator import translate_category
from execution.github_actions.news.weather import fetch_alerts, fetch_weather
from execution.github_actions.news.weather_advisory import generate_weather_advisory

logger = logging.getLogger(__name__)


@dataclass
class NewsBriefingResult:
    """dry-run / 관측용 — 실제 로직 분기에 사용되지 않음."""
    sent: bool
    skipped: bool
    reason: str = ""
    items_count: int = 0
    indices_count: int = 0
    message_text: str = ""
    categories: dict[Category, int] = field(default_factory=dict)


def _select_top_per_category(
    clustered: dict[str, Any],
    candidates_by_cat: dict[Category, list[RssItem]],
) -> dict[Category, list[RssItem]]:
    """cluster_and_select 결과를 다시 카테고리별로 묶어 반환."""
    out: dict[Category, list[RssItem]] = {}
    # 본 함수는 현재 미사용 — 클러스터링은 카테고리별로 수행함 (run_news_briefing 참조).
    return out


def run_news_briefing(
    supabase: SupabaseClient,
    telegram: HttpTelegramSender,
    providers: Sequence[Provider],
    user_id: str,
    *,
    now_utc: datetime | None = None,
    schedule_override: UserSchedule | None = None,
    skip_gate: bool = False,
) -> NewsBriefingResult:
    """W8a 매 30분 cron 진입점.

    skip_gate=True 시 should_send_news_briefing 바이패스 (dry-run / manual).
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    schedule = schedule_override or fetch_user_schedule(supabase, user_id)

    # 1) Schedule gate
    if not skip_gate and not should_send_news_briefing(schedule, now_utc):
        return NewsBriefingResult(sent=False, skipped=True, reason="schedule gate")

    # 2) RSS 수집 + L1 URL dedup + Reddit 인기도 boost + 후보 선발
    # 2026-05-03 PR #88: Reddit 24h top URLs 매칭 시 정렬 +boost (사용자 인기 시그널).
    # 부분 실패 허용 — 빈 set 이어도 기존 tier 정렬만으로 정상 동작.
    popular_hashes: set[str] = set()
    try:
        popular_hashes = fetch_popular_url_hashes()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Reddit 인기도 fetch 실패 — boost 없이 진행: %s", exc)

    candidates_by_cat = fetch_all(
        CATALOG, user_id, supabase, now=now_utc, popular_url_hashes=popular_hashes,
    )
    total_candidates = sum(len(lst) for lst in candidates_by_cat.values())
    if total_candidates == 0:
        logger.warning("뉴스 후보 0건 — 발송 skip")
        return NewsBriefingResult(sent=False, skipped=True, reason="no candidates")

    # 3) yfinance 지수 fetch (부분 실패 허용)
    indices = fetch_indices()
    indices_failed = len(indices) == 0

    # 3b) 날씨 (Open-Meteo + NWS) — USER_CITY env 미설정 시 섹션 전체 skip.
    from src.config.user_profile import load_user_city
    city = load_user_city()
    if city is None:
        weather, week, alerts = None, None, []
    else:
        weather_result = fetch_weather(city.lat, city.lon, city.timezone)
        if weather_result is not None:
            weather, week = weather_result
        else:
            weather, week = None, None
        alerts = fetch_alerts(city.lat, city.lon)
    # 2~3문장 한국어 조언 — LLM 1회 (실패 시 규칙 기반 폴백).
    weather_advisory = generate_weather_advisory(weather, week, alerts, providers)

    # 4+5) 카테고리별 클러스터링 → 번역 → 발송 후보 확정
    # 2026-05-03 PR #89: user_interests 를 프롬프트 hint 에 주입 (사용자 관심 가중치).
    selected: dict[Category, list[RssItem]] = {}
    cluster_topic_by_url: dict[str, str | None] = {}
    user_interests = schedule.news_interests
    for cat, cands in candidates_by_cat.items():
        if not cands:
            continue
        # 클러스터링 + best 선택 (LLM 실패 시 결정론 폴백).
        # category + user_interests 를 전달해 프롬프트에 macro-first + 개인 관심 hint 삽입.
        bests, topics = cluster_and_select(
            cands, providers, category=cat, user_interests=user_interests,
        )
        cluster_topic_by_url.update(topics)
        # 번역 (category_max 건까지 채우도록 voucher 폴백).
        target = CATEGORY_MAX_ITEMS.get(cat, 2)
        translated = translate_category(bests, target, providers)
        if translated:
            selected[cat] = translated

    # 6) 메시지 조립
    message = build_message(
        indices=indices,
        selected=selected,
        now=now_utc,
        indices_failed=indices_failed,
        weather=weather,
        alerts=alerts,
        weather_advisory=weather_advisory,
    )

    all_items = [it for lst in selected.values() for it in lst]
    if not all_items and indices_failed:
        # 보낼 내용이 전혀 없음 — skip (다음 30분 cron 이 재시도).
        logger.warning("뉴스·지수 모두 실패 — 발송 skip")
        return NewsBriefingResult(sent=False, skipped=True, reason="no content")

    # 7) Telegram 발송
    chat_id = resolve_chat_id(
        schedule,
        env_fallback=int(os.environ["TELEGRAM_CHAT_ID"]) if os.environ.get("TELEGRAM_CHAT_ID") else None,
    )
    try:
        telegram.send(
            chat_id,
            message,
            parse_mode="MarkdownV2",
            disable_web_page_preview=True,
        )
    except TelegramSendError as exc:
        # 2026-05-14 — Telegram MDV2 파싱 실패 시 message_text preview 로깅.
        # "byte offset N Italic" 같은 에러는 message 컨텐츠 보이지 않으면 디버그 불가.
        msg_bytes = message.encode("utf-8")
        logger.error(
            "Telegram 발송 실패: %s — message len=%d bytes (%d chars), "
            "preview head:\n%s\n... preview tail:\n%s",
            exc, len(msg_bytes), len(message),
            message[:1200], message[-600:],
        )
        return NewsBriefingResult(sent=False, skipped=False, reason=f"telegram: {exc}")

    # 8) 기록 — news_sent_log + user_preferences.last_news_sent_at
    mark_all_sent(user_id, all_items, cluster_topic_by_url, supabase)
    mark_news_sent(supabase, user_id, now_utc)

    return NewsBriefingResult(
        sent=True,
        skipped=False,
        items_count=len(all_items),
        indices_count=len(indices),
        message_text=message,
        categories={cat: len(lst) for cat, lst in selected.items()},
    )


__all__ = ["NewsBriefingResult", "run_news_briefing"]
