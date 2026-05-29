# PR I — cron 재설계용 schedule gate.
# blueprint §2 W3/W4/W6 의 트리거 시각을 schedules.yaml 고정에서 user_preferences 테이블로 옮김.
# 매 30분 runner 가 user_preferences 를 읽고 지금 발송 대상인지 판정 → schedule-based 다중 사용자 확장 준비.
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from execution.github_actions._supabase import SupabaseClient, SupabaseError
from src.config.schedules import MetricSchedule, MetricTarget, Schedules

logger = logging.getLogger(__name__)


# ============================================================
# §1  UserSchedule — user_preferences 1 행을 값 객체로 표현
# ============================================================
_DEFAULT_TZ = "America/Chicago"
_DEFAULT_BRIEFING_TIME = time(6, 0)
_DEFAULT_WEEKLY_DAY = 6                    # 0=월, 6=일
_DEFAULT_WEEKLY_TIME = time(21, 0)


@dataclass(frozen=True)
class ReminderPlan:
    """user_preferences.reminder_plans jsonb 의 metric 1 개 값.

    preset 은 UI 복원용 참고 필드 — 서버 로직은 active/days/times 만 본다.
    """
    active: bool                            # preset="off" 또는 times 비어있으면 False
    days: tuple[int, ...]                   # 0=월 ~ 6=일
    times: tuple[time, ...]                 # 로컬 시각 (복수 가능)
    preset: str = ""                        # "morning_evening" 등 — 참고용
    context: str | None = None              # 혈당 "fasting"/"post_meal" (Stage 2)


@dataclass(frozen=True)
class UserSchedule:
    """user_preferences 의 schedule 관련 필드를 묶은 값 객체.

    fetch 실패/행 없음 → `_defaults(user_id)` 로 안전 fallback.
    timezone 파싱 실패 시에도 America/Chicago 로 폴백 (경고 로그).
    """
    user_id: str
    briefing_time: time
    reminder_times: dict[str, time]         # metric_kind(bp/weight/…) → 로컬 시각 (legacy, 단일 시각)
    reminder_plans: dict[str, ReminderPlan] # PR J — 프리셋 기반 풍부한 plan (복수 시각, 요일, context)
    weekly_day: int                         # 0=월, 6=일
    weekly_time: time
    timezone: ZoneInfo
    telegram_chat_id: int | None            # None 이면 env TELEGRAM_CHAT_ID 사용
    last_weekly_sent_at: datetime | None    # UTC
    last_news_sent_at: datetime | None = None   # W8a — 뉴스 브리핑 발송 판정용 (UTC)
    news_briefing_time: time | None = None  # W8a hotfix — None 이면 briefing_time+1분 폴백
    # W8b PR #89 — 뉴스 큐레이션 개인 관심 키워드 (jsonb array of strings).
    # cluster_select.cluster_and_select 의 카테고리 hint 에 동적 주입.
    news_interests: tuple[str, ...] = ()
    # W9 PR #91~ — 영어회화 한 마디.  None 이면 기능 비활성 (default).
    english_phrase_time: time | None = None
    last_english_sent_at: datetime | None = None


def _defaults(user_id: str) -> UserSchedule:
    return UserSchedule(
        user_id=user_id,
        briefing_time=_DEFAULT_BRIEFING_TIME,
        reminder_times={},
        reminder_plans={},
        weekly_day=_DEFAULT_WEEKLY_DAY,
        weekly_time=_DEFAULT_WEEKLY_TIME,
        timezone=ZoneInfo(_DEFAULT_TZ),
        telegram_chat_id=None,
        last_weekly_sent_at=None,
        last_news_sent_at=None,
        news_briefing_time=None,
        news_interests=(),
        english_phrase_time=None,
        last_english_sent_at=None,
    )


# ============================================================
# §2  파싱 헬퍼 — 외부 입력은 신뢰 안 함, 실패 시 기본값
# ============================================================
def _parse_time(raw: Any, default: time, field: str) -> time:
    """'HH:MM' 또는 'HH:MM:SS' 문자열 → time. 실패 시 default + 경고 로그."""
    if raw is None:
        return default
    if isinstance(raw, time):
        return raw
    if not isinstance(raw, str):
        logger.warning("invalid %s=%r (not a string) — using default %s", field, raw, default)
        return default
    parts = raw.split(":")
    if len(parts) < 2:
        logger.warning("invalid %s=%r — using default %s", field, raw, default)
        return default
    try:
        hh = int(parts[0])
        mm = int(parts[1])
        return time(hh, mm)
    except ValueError:
        logger.warning("invalid %s=%r — using default %s", field, raw, default)
        return default


def _parse_timezone(raw: Any) -> ZoneInfo:
    if not raw or not isinstance(raw, str):
        return ZoneInfo(_DEFAULT_TZ)
    try:
        return ZoneInfo(raw)
    except ZoneInfoNotFoundError:
        logger.warning("invalid timezone=%r — using %s", raw, _DEFAULT_TZ)
        return ZoneInfo(_DEFAULT_TZ)


def _parse_reminder_times(raw: Any) -> dict[str, time]:
    """user_preferences.reminder_times jsonb → {metric_kind: time}."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, time] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        parsed = _parse_time(value, default=time(0, 0), field=f"reminder_times.{key}")
        # 기본값과 구분: 명시적으로 잘못된 값은 skip 하고 싶으면 다시 파싱 필요.
        # 여기서는 '00:00' 도 유효한 사용자 입력일 수 있어 값 자체는 보존.
        if isinstance(value, str) and ":" in value:
            out[key] = parsed
    return out


def _parse_reminder_plans(raw: Any) -> dict[str, ReminderPlan]:
    """user_preferences.reminder_plans jsonb → {metric_kind: ReminderPlan}.

    스키마 예:
      {"bp": {"preset": "morning_evening", "days": [0,1,2,3,4,5,6], "times": ["08:00","20:00"]}}
    잘못된 필드는 스킵 (cron 을 막지 않음).
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, ReminderPlan] = {}
    for code, value in raw.items():
        if not isinstance(code, str) or not isinstance(value, dict):
            continue
        preset = value.get("preset", "") if isinstance(value.get("preset"), str) else ""
        # off 프리셋은 명시적 비활성 — times 비어있어도 이 상태를 구분
        if preset == "off":
            out[code] = ReminderPlan(
                active=False, days=(), times=(), preset=preset, context=None,
            )
            continue
        days_raw = value.get("days")
        days: tuple[int, ...] = ()
        if isinstance(days_raw, list):
            days = tuple(
                d for d in days_raw if isinstance(d, int) and 0 <= d <= 6
            )
        times_raw = value.get("times")
        times: tuple[time, ...] = ()
        if isinstance(times_raw, list):
            parsed_times: list[time] = []
            for idx, t in enumerate(times_raw):
                if not isinstance(t, str):
                    continue
                pt = _parse_time(t, default=time(0, 0), field=f"reminder_plans.{code}.times[{idx}]")
                if isinstance(t, str) and ":" in t:
                    parsed_times.append(pt)
            times = tuple(parsed_times)
        context = value.get("context")
        context_str = context if isinstance(context, str) else None
        # 시간 또는 요일이 비어있으면 active=False (실질적 비활성)
        active = bool(days) and bool(times)
        out[code] = ReminderPlan(
            active=active, days=days, times=times, preset=preset, context=context_str,
        )
    return out


def _parse_weekly_day(raw: Any) -> int:
    if isinstance(raw, int) and 0 <= raw <= 6:
        return raw
    try:
        coerced = int(raw)
        if 0 <= coerced <= 6:
            return coerced
    except (TypeError, ValueError):
        pass
    logger.warning("invalid weekly_day=%r — using default %d", raw, _DEFAULT_WEEKLY_DAY)
    return _DEFAULT_WEEKLY_DAY


def _parse_chat_id(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("invalid telegram_chat_id=%r — ignoring", raw)
        return None


def _parse_iso_utc(raw: Any) -> datetime | None:
    if not raw or not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


# ============================================================
# §3  Fetcher — user_preferences 조회 + 결측 시 defaults
# ============================================================
def fetch_user_schedule(client: SupabaseClient, user_id: str) -> UserSchedule:
    """user_preferences 1 행을 UserSchedule 값 객체로 변환.

    - 행이 없으면 defaults (신규 사용자 UX: settings 페이지 미방문 상태에서도 발송)
    - 네트워크/파싱 오류는 경고 로그 + defaults — cron 을 절대 막지 않음
    """
    try:
        rows = client.select(
            "user_preferences",
            filters=[("user_id", f"eq.{user_id}")],
            select_cols=(
                "user_id,briefing_time,news_briefing_time,news_interests,"
                "english_phrase_time,last_english_sent_at,"
                "reminder_times,reminder_plans,"
                "weekly_day,weekly_time,"
                "timezone,telegram_chat_id,last_weekly_sent_at,last_news_sent_at"
            ),
        )
    except SupabaseError as exc:
        logger.warning("fetch_user_schedule failed for %s: %s — using defaults", user_id, exc)
        return _defaults(user_id)

    if not rows:
        return _defaults(user_id)

    row = rows[0]
    return UserSchedule(
        user_id=user_id,
        briefing_time=_parse_time(row.get("briefing_time"), _DEFAULT_BRIEFING_TIME, "briefing_time"),
        reminder_times=_parse_reminder_times(row.get("reminder_times")),
        reminder_plans=_parse_reminder_plans(row.get("reminder_plans")),
        weekly_day=_parse_weekly_day(row.get("weekly_day")),
        weekly_time=_parse_time(row.get("weekly_time"), _DEFAULT_WEEKLY_TIME, "weekly_time"),
        timezone=_parse_timezone(row.get("timezone")),
        telegram_chat_id=_parse_chat_id(row.get("telegram_chat_id")),
        last_weekly_sent_at=_parse_iso_utc(row.get("last_weekly_sent_at")),
        last_news_sent_at=_parse_iso_utc(row.get("last_news_sent_at")),
        news_briefing_time=_parse_optional_time(row.get("news_briefing_time")),
        news_interests=_parse_news_interests(row.get("news_interests")),
        english_phrase_time=_parse_optional_time(row.get("english_phrase_time")),
        last_english_sent_at=_parse_iso_utc(row.get("last_english_sent_at")),
    )


def _parse_news_interests(raw: Any) -> tuple[str, ...]:
    """news_interests jsonb (array of strings) → tuple. 잘못된 값은 빈 튜플."""
    if not isinstance(raw, list):
        return ()
    out = tuple(s.strip() for s in raw if isinstance(s, str) and s.strip())
    # 최대 30개 (cluster_select 프롬프트 비대화 방지).
    return out[:30]


def _parse_optional_time(raw: Any) -> time | None:
    """'HH:MM' / 'HH:MM:SS' / None → time | None.  잘못된 값은 None (폴백 트리거)."""
    if raw is None:
        return None
    if isinstance(raw, time):
        return raw
    if not isinstance(raw, str):
        return None
    parts = raw.split(":")
    if len(parts) < 2:
        return None
    try:
        return time(int(parts[0]), int(parts[1]))
    except ValueError:
        return None


# ============================================================
# §4  Briefing gate — 옵션 B (DB sent_at 기반 중복 방지)
# ============================================================
def should_send_briefing(
    schedule: UserSchedule,
    client: SupabaseClient,
    user_id: str,
    now_utc: datetime,
) -> bool:
    """briefing 발송 여부 판정.

    True 조건 (AND):
      1) 사용자 로컬 시각 >= briefing_time
      2) daily_summary[yesterday].sent_at IS NULL (어제 데이터 브리핑 아직 미발송)

    아침 브리핑은 전날 데이터 요약 → summary_date = yesterday.
    이유: runner 가 지연/실패해도 다음 30분 cron 이 재시도하여 반드시 전달.
    """
    now_local = now_utc.astimezone(schedule.timezone)
    target_dt = datetime.combine(now_local.date(), schedule.briefing_time, schedule.timezone)
    if now_local < target_dt:
        return False

    yesterday = (now_local.date() - timedelta(days=1)).isoformat()
    try:
        rows = client.select(
            "daily_summary",
            filters=[
                ("user_id", f"eq.{user_id}"),
                ("summary_date", f"eq.{yesterday}"),
                ("sent_at", "not.is.null"),
            ],
            select_cols="summary_date",
        )
    except SupabaseError as exc:
        # DB 조회 실패 시 안전하게 skip — 중복 발송 위험보다 누락 방지.
        # 다음 30분 runner 가 재시도.
        logger.warning("should_send_briefing DB check failed: %s — skipping this tick", exc)
        return False

    return not rows


# ============================================================
# §5  Weekly gate — 요일 + 시각 + 이번 주 미발송
# ============================================================
def _week_start_monday(local_date: date) -> date:
    """해당 주의 월요일 (weekday(): 월=0)."""
    return local_date - timedelta(days=local_date.weekday())


def should_send_weekly(
    schedule: UserSchedule,
    now_utc: datetime,
) -> bool:
    """weekly_report 발송 여부 판정.

    True 조건 (AND):
      1) 사용자 로컬 요일 == weekly_day
      2) 사용자 로컬 시각 >= weekly_time
      3) last_weekly_sent_at < 이번 주 월요일 00:00 (로컬)
         (= 이번 주에는 아직 안 보냄)
    """
    now_local = now_utc.astimezone(schedule.timezone)
    if now_local.weekday() != schedule.weekly_day:
        return False
    target_dt = datetime.combine(now_local.date(), schedule.weekly_time, schedule.timezone)
    if now_local < target_dt:
        return False

    if schedule.last_weekly_sent_at is None:
        return True

    last_local = schedule.last_weekly_sent_at.astimezone(schedule.timezone)
    this_week_start_local = datetime.combine(
        _week_start_monday(now_local.date()), time(0, 0), schedule.timezone
    )
    return last_local < this_week_start_local


def should_send_news_briefing(
    schedule: UserSchedule,
    now_utc: datetime,
) -> bool:
    """뉴스 브리핑 (W8a) 발송 여부.

    True 조건 (AND):
      1) 로컬 시각 >= news_briefing_time (없으면 briefing_time + 1분 폴백)
      2) 오늘 로컬 날짜 기준 아직 뉴스 발송 안 됨
         → last_news_sent_at is None OR (로컬 date < 오늘)

    2026-04-21 hotfix — news_briefing_time 명시 시 그 시각 사용.
    None 이면 기존 동작(건강 브리핑 +1분) 유지로 하위 호환.
    """
    local_now = now_utc.astimezone(schedule.timezone)
    local_today = local_now.date()

    if schedule.news_briefing_time is not None:
        news_dt = datetime.combine(local_today, schedule.news_briefing_time, schedule.timezone)
    else:
        # 폴백 — 건강 브리핑 +1분.
        news_dt = datetime.combine(local_today, schedule.briefing_time, schedule.timezone)
        news_dt = news_dt + timedelta(minutes=1)
    if local_now < news_dt:
        return False

    if schedule.last_news_sent_at is None:
        return True
    last_local = schedule.last_news_sent_at.astimezone(schedule.timezone)
    return last_local.date() < local_today


def mark_news_sent(
    client: SupabaseClient,
    user_id: str,
    now_utc: datetime | None = None,
) -> None:
    """뉴스 발송 성공 후 last_news_sent_at 을 UTC 로 기록 (last_weekly_sent_at 패턴 복사)."""
    ts = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    row = {
        "user_id": user_id,
        "last_news_sent_at": ts.isoformat().replace("+00:00", "Z"),
    }
    try:
        client.upsert("user_preferences", row, on_conflict="user_id")
    except SupabaseError as exc:
        logger.warning("mark_news_sent upsert failed: %s", exc)


def should_send_english_phrase(
    schedule: UserSchedule,
    now_utc: datetime,
) -> bool:
    """영어회화 한 마디 발송 여부 (W9 PR #91~).

    True 조건 (AND):
      1) english_phrase_time 명시 (None 이면 기능 비활성 — default 비활성)
      2) 로컬 시각 >= english_phrase_time
      3) 오늘 로컬 날짜 기준 아직 미발송 (last_english_sent_at None 또는 어제 이전)
    """
    if schedule.english_phrase_time is None:
        return False
    local_now = now_utc.astimezone(schedule.timezone)
    local_today = local_now.date()
    target_dt = datetime.combine(local_today, schedule.english_phrase_time, schedule.timezone)
    if local_now < target_dt:
        return False
    if schedule.last_english_sent_at is None:
        return True
    last_local = schedule.last_english_sent_at.astimezone(schedule.timezone)
    return last_local.date() < local_today


def mark_english_sent(
    client: SupabaseClient,
    user_id: str,
    now_utc: datetime | None = None,
) -> None:
    """영어회화 발송 성공 후 last_english_sent_at upsert."""
    ts = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    row = {
        "user_id": user_id,
        "last_english_sent_at": ts.isoformat().replace("+00:00", "Z"),
    }
    try:
        client.upsert("user_preferences", row, on_conflict="user_id")
    except SupabaseError as exc:
        logger.warning("mark_english_sent upsert failed: %s", exc)


def mark_weekly_sent(
    client: SupabaseClient,
    user_id: str,
    now_utc: datetime | None = None,
) -> None:
    """weekly 발송 성공 후 user_preferences.last_weekly_sent_at 을 UTC 로 기록.

    실패 시 경고만 남기고 raise 하지 않음 — 다음 주 중복 발송 방지는 보조 장치.
    """
    ts = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    row = {
        "user_id": user_id,
        "last_weekly_sent_at": ts.isoformat().replace("+00:00", "Z"),
    }
    try:
        client.upsert("user_preferences", row, on_conflict="user_id")
    except SupabaseError as exc:
        logger.warning("mark_weekly_sent upsert failed: %s", exc)


# ============================================================
# §6  Reminder override — user_preferences.reminder_times 를 Schedules 에 merge
# ============================================================
def resolve_reminder_schedules(
    schedule: UserSchedule,
    default_schedules: Schedules,
) -> Schedules:
    """user_preferences.reminder_plans · reminder_times 로 schedules.yaml 의 target 을 오버라이드.

    우선순위 (metric 당 독립적):
      1) reminder_plans[code] 있으면 → active + days + times 로 MetricSchedule 재구성
         - preset="off" 또는 active=False → metric 전체 비활성 (decide_reminders 가 skip)
         - 복수 times → 복수 target, delay_min=0
      2) reminder_times[code] 있으면 → (legacy) 단일 time 으로 targets 치환
      3) 둘 다 없으면 → schedules.yaml 기본값 유지 (하위 호환)
    """
    if not schedule.reminder_plans and not schedule.reminder_times:
        return default_schedules

    merged: dict[str, MetricSchedule] = {}
    for code, metric in default_schedules.metrics.items():
        # 1) reminder_plans 우선
        plan = schedule.reminder_plans.get(code)
        if plan is not None:
            if not plan.active:
                merged[code] = replace(metric, active=False)
                continue
            merged[code] = replace(
                metric,
                active=True,
                days=plan.days,
                targets=tuple(
                    MetricTarget(time_of_day=t, delay_min=0) for t in plan.times
                ),
                context=plan.context,
            )
            continue
        # 2) legacy reminder_times 오버라이드
        override = schedule.reminder_times.get(code)
        if override is not None:
            merged[code] = replace(
                metric,
                targets=(MetricTarget(time_of_day=override, delay_min=0),),
            )
            continue
        # 3) yaml 기본값 유지
        merged[code] = metric

    return replace(default_schedules, metrics=merged)


# ============================================================
# §7  Chat ID resolver — user_preferences 우선, 없으면 env fallback
# ============================================================
def resolve_chat_id(schedule: UserSchedule, env_fallback: int | None = None) -> int:
    """schedule.telegram_chat_id 가 있으면 그것, 아니면 env_fallback 또는 TELEGRAM_CHAT_ID.

    env_fallback 도 없으면 ValueError — cron 을 멈추게 하기보다는 호출 측이 판단하도록.
    """
    if schedule.telegram_chat_id is not None:
        return schedule.telegram_chat_id
    if env_fallback is not None:
        return env_fallback
    raw = os.environ.get("TELEGRAM_CHAT_ID")
    if raw:
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"TELEGRAM_CHAT_ID must be int, got {raw!r}") from exc
    raise ValueError(
        f"chat_id unresolved — user_preferences.telegram_chat_id NULL and "
        f"TELEGRAM_CHAT_ID env missing (user_id={schedule.user_id})"
    )


__all__ = [
    "ReminderPlan",
    "UserSchedule",
    "fetch_user_schedule",
    "should_send_briefing",
    "should_send_weekly",
    "mark_weekly_sent",
    "resolve_reminder_schedules",
    "resolve_chat_id",
]
