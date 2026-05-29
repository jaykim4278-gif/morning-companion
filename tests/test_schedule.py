# Layer 4 Sensor — PR I (cron 재설계) schedule gate 검증.
# 순수 함수(parse/merge/should_*) 는 mock client 로, DB-touching 경로는 httpx MockTransport 로 계약 검증.
from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from execution.github_actions._schedule import (
    ReminderPlan,
    UserSchedule,
    fetch_user_schedule,
    mark_weekly_sent,
    resolve_chat_id,
    resolve_reminder_schedules,
    should_send_briefing,
    should_send_weekly,
)
from execution.github_actions._supabase import SupabaseClient
from src.config.schedules import (
    BriefingSchedule,
    MetricSchedule,
    MetricTarget,
    RateLimits,
    Schedules,
)

SUPA_URL = "https://fake.supabase.co"
KEY = "SR_KEY"
UID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
CT = ZoneInfo("America/Chicago")


def _mock(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0)


def _mk_client(handler: Callable[[httpx.Request], httpx.Response]) -> SupabaseClient:
    return SupabaseClient(SUPA_URL, KEY, client=_mock(handler))


def _default_schedules() -> Schedules:
    """schedules.yaml 과 동일한 기본값 구조 — 테스트 결정론."""
    metrics = {
        "bp": MetricSchedule(
            code="bp", label="혈압", active=True,
            targets=(
                MetricTarget(time_of_day=time(8, 0), delay_min=30),
                MetricTarget(time_of_day=time(20, 0), delay_min=30),
            ),
        ),
        "weight": MetricSchedule(
            code="weight", label="체중", active=True,
            targets=(MetricTarget(time_of_day=time(7, 30), delay_min=90),),
        ),
        "glucose": MetricSchedule(
            code="glucose", label="혈당", active=True,
            targets=(MetricTarget(time_of_day=time(7, 0), delay_min=60),),
        ),
    }
    return Schedules(
        version="test",
        timezone="America/Chicago",
        metrics=metrics,
        morning_briefing=BriefingSchedule(
            time_of_day=time(6, 0), weekdays=(0, 1, 2, 3, 4, 5, 6),
        ),
        weekly_report=BriefingSchedule(
            time_of_day=time(21, 0), weekdays=(6,), weekday=6,
        ),
        rate_limits=RateLimits(
            max_reminders_per_day_per_metric=1,
            reminder_window_start=time(6, 0),
            reminder_window_end=time(22, 0),
        ),
    )


def _make_schedule(**overrides) -> UserSchedule:
    base = {
        "user_id": UID,
        "briefing_time": time(6, 0),
        "reminder_times": {},
        "reminder_plans": {},
        "weekly_day": 6,
        "weekly_time": time(21, 0),
        "timezone": CT,
        "telegram_chat_id": None,
        "last_weekly_sent_at": None,
    }
    base.update(overrides)
    return UserSchedule(**base)


# ============================================================
# §1  fetch_user_schedule — 행 없음 / 있음 / DB 오류
# ============================================================
class TestFetchUserSchedule:
    def test_row_missing_returns_defaults(self) -> None:
        def handler(req: httpx.Request) -> httpx.Response:
            assert "/rest/v1/user_preferences" in str(req.url)
            return httpx.Response(200, json=[])

        sched = fetch_user_schedule(_mk_client(handler), UID)
        assert sched.briefing_time == time(6, 0)
        assert sched.reminder_times == {}
        assert sched.weekly_day == 6
        assert sched.weekly_time == time(21, 0)
        assert sched.timezone.key == "America/Chicago"
        assert sched.telegram_chat_id is None
        assert sched.last_weekly_sent_at is None

    def test_row_present_parsed(self) -> None:
        row = {
            "user_id": UID,
            "briefing_time": "07:15:00",
            "reminder_times": {"bp": "09:00", "weight": "07:45"},
            "weekly_day": 5,
            "weekly_time": "20:30:00",
            "timezone": "Asia/Seoul",
            "telegram_chat_id": 12345,
            "last_weekly_sent_at": "2026-04-12T02:00:00Z",
        }

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[row])

        sched = fetch_user_schedule(_mk_client(handler), UID)
        assert sched.briefing_time == time(7, 15)
        assert sched.reminder_times == {"bp": time(9, 0), "weight": time(7, 45)}
        assert sched.weekly_day == 5
        assert sched.weekly_time == time(20, 30)
        assert sched.timezone.key == "Asia/Seoul"
        assert sched.telegram_chat_id == 12345
        assert sched.last_weekly_sent_at is not None
        assert sched.last_weekly_sent_at.tzinfo is not None

    def test_invalid_timezone_fallback(self) -> None:
        row = {
            "user_id": UID,
            "briefing_time": "06:00",
            "reminder_times": {},
            "weekly_day": 6,
            "weekly_time": "21:00",
            "timezone": "Not/A_Real_Tz",
            "telegram_chat_id": None,
            "last_weekly_sent_at": None,
        }

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[row])

        sched = fetch_user_schedule(_mk_client(handler), UID)
        assert sched.timezone.key == "America/Chicago"

    def test_db_error_returns_defaults(self) -> None:
        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="internal error")

        sched = fetch_user_schedule(_mk_client(handler), UID)
        # cron 을 막지 않음 — defaults 로 동작 계속
        assert sched.briefing_time == time(6, 0)
        assert sched.user_id == UID

    def test_invalid_weekly_day_fallback(self) -> None:
        row = {
            "user_id": UID, "briefing_time": "06:00", "reminder_times": {},
            "weekly_day": 99, "weekly_time": "21:00", "timezone": "America/Chicago",
            "telegram_chat_id": None, "last_weekly_sent_at": None,
        }

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[row])

        sched = fetch_user_schedule(_mk_client(handler), UID)
        assert sched.weekly_day == 6


# ============================================================
# §2  should_send_briefing
# ============================================================
class TestShouldSendBriefing:
    def _make_client_with_summary(
        self, rows: list[dict], expected_today_iso: str | None = None,
    ) -> SupabaseClient:
        def handler(req: httpx.Request) -> httpx.Response:
            # 쿼리 params 검증 (sent_at is not null 필터 명시)
            qs = str(req.url.query)
            assert "sent_at=not.is.null" in qs
            if expected_today_iso:
                assert f"summary_date=eq.{expected_today_iso}" in qs
            return httpx.Response(200, json=rows)

        return _mk_client(handler)

    def test_before_target_time_returns_false(self) -> None:
        sched = _make_schedule(briefing_time=time(6, 0))
        # 05:30 CT = 10:30 UTC (CDT)
        now_utc = datetime(2026, 4, 15, 10, 30, tzinfo=timezone.utc)
        # 로컬 05:30 < 06:00 → False (DB 조회 없음)
        client = self._make_client_with_summary([])  # 호출돼도 문제는 없음
        assert should_send_briefing(sched, client, UID, now_utc) is False

    def test_after_target_and_not_sent_returns_true(self) -> None:
        sched = _make_schedule(briefing_time=time(6, 0))
        # 06:05 CT 4/15 = 11:05 UTC (CDT) → 아침 브리핑은 전날(4/14) 요약
        now_utc = datetime(2026, 4, 15, 11, 5, tzinfo=timezone.utc)
        client = self._make_client_with_summary([], expected_today_iso="2026-04-14")
        assert should_send_briefing(sched, client, UID, now_utc) is True

    def test_after_target_but_already_sent_returns_false(self) -> None:
        sched = _make_schedule(briefing_time=time(6, 0))
        now_utc = datetime(2026, 4, 15, 14, 0, tzinfo=timezone.utc)  # 09:00 CT → yesterday=4/14
        client = self._make_client_with_summary([{"summary_date": "2026-04-14"}])
        assert should_send_briefing(sched, client, UID, now_utc) is False

    def test_db_error_returns_false_safely(self) -> None:
        sched = _make_schedule(briefing_time=time(6, 0))
        now_utc = datetime(2026, 4, 15, 11, 30, tzinfo=timezone.utc)

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        assert should_send_briefing(sched, _mk_client(handler), UID, now_utc) is False

    def test_custom_timezone_asia_seoul(self) -> None:
        """briefing_time=07:00 Asia/Seoul — 22:00 UTC 전날이면 False, 22:00 이후면 True."""
        sched = _make_schedule(
            briefing_time=time(7, 0), timezone=ZoneInfo("Asia/Seoul"),
        )
        # 2026-04-14 21:30 UTC = 2026-04-15 06:30 KST → False
        now_pre = datetime(2026, 4, 14, 21, 30, tzinfo=timezone.utc)
        client_pre = self._make_client_with_summary([])
        assert should_send_briefing(sched, client_pre, UID, now_pre) is False

        # 2026-04-14 22:30 UTC = 2026-04-15 07:30 KST → True (전날 4/14 미발송 조건)
        now_post = datetime(2026, 4, 14, 22, 30, tzinfo=timezone.utc)
        client_post = self._make_client_with_summary([], expected_today_iso="2026-04-14")
        assert should_send_briefing(sched, client_post, UID, now_post) is True


# ============================================================
# §3  should_send_weekly
# ============================================================
class TestShouldSendWeekly:
    def test_wrong_weekday_returns_false(self) -> None:
        sched = _make_schedule(weekly_day=6, weekly_time=time(21, 0))
        # 2026-04-15 수요일 02:00 UTC → 21:00 CDT 전날 화요일
        now_utc = datetime(2026, 4, 15, 2, 0, tzinfo=timezone.utc)
        assert should_send_weekly(sched, now_utc) is False

    def test_before_target_time_returns_false(self) -> None:
        sched = _make_schedule(weekly_day=6, weekly_time=time(21, 0))
        # 2026-04-19 일요일 18:00 UTC = 13:00 CDT (21:00 이전)
        now_utc = datetime(2026, 4, 19, 18, 0, tzinfo=timezone.utc)
        assert should_send_weekly(sched, now_utc) is False

    def test_correct_day_and_time_not_sent_this_week_returns_true(self) -> None:
        sched = _make_schedule(
            weekly_day=6, weekly_time=time(21, 0), last_weekly_sent_at=None,
        )
        # 2026-04-20 월요일 02:00 UTC = 21:00 CDT 일요일 2026-04-19
        now_utc = datetime(2026, 4, 20, 2, 0, tzinfo=timezone.utc)
        assert should_send_weekly(sched, now_utc) is True

    def test_already_sent_this_week_returns_false(self) -> None:
        # 이번 주 월요일 2026-04-13 00:00 CDT = 2026-04-13 05:00 UTC
        # 발송 기록은 2026-04-19 02:30 UTC (이번 주 일요일 21:30 CDT)
        sched = _make_schedule(
            weekly_day=6, weekly_time=time(21, 0),
            last_weekly_sent_at=datetime(2026, 4, 20, 2, 30, tzinfo=timezone.utc),
        )
        # 재시도 runner 30분 후 (2026-04-20 03:00 UTC = 일요일 22:00 CDT)
        now_utc = datetime(2026, 4, 20, 3, 0, tzinfo=timezone.utc)
        assert should_send_weekly(sched, now_utc) is False

    def test_last_sent_in_prior_week_returns_true(self) -> None:
        # last_weekly_sent_at = 지난 주 일요일 2026-04-13 02:30 UTC
        sched = _make_schedule(
            weekly_day=6, weekly_time=time(21, 0),
            last_weekly_sent_at=datetime(2026, 4, 13, 2, 30, tzinfo=timezone.utc),
        )
        # 이번 주 일요일 21:00 CDT = 2026-04-20 02:00 UTC
        now_utc = datetime(2026, 4, 20, 2, 0, tzinfo=timezone.utc)
        assert should_send_weekly(sched, now_utc) is True


class TestMarkWeeklySent:
    def test_upsert_sends_correct_payload(self) -> None:
        captured: list[bytes] = []

        def handler(req: httpx.Request) -> httpx.Response:
            assert req.method == "POST"
            assert "/rest/v1/user_preferences" in str(req.url)
            assert "on_conflict=user_id" in str(req.url.query)
            captured.append(req.read())
            return httpx.Response(204)

        mark_weekly_sent(_mk_client(handler), UID,
                         now_utc=datetime(2026, 4, 20, 2, 0, tzinfo=timezone.utc))
        # 요청 본문에 user_id + last_weekly_sent_at 포함
        assert len(captured) == 1
        body_str = captured[0].decode("utf-8")
        assert UID in body_str
        assert "last_weekly_sent_at" in body_str
        assert "2026-04-20T02:00:00Z" in body_str

    def test_upsert_network_error_does_not_raise(self) -> None:
        """DB 실패 시 cron 을 막지 않음 — 다음 주 중복 발송 위험은 수용."""
        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        # raise 하지 않아야 함
        mark_weekly_sent(_mk_client(handler), UID,
                         now_utc=datetime(2026, 4, 20, 2, 0, tzinfo=timezone.utc))


# ============================================================
# §4  resolve_reminder_schedules — override merge
# ============================================================
class TestResolveReminderSchedules:
    def test_no_overrides_returns_same_schedules(self) -> None:
        defaults = _default_schedules()
        sched = _make_schedule(reminder_times={})
        merged = resolve_reminder_schedules(sched, defaults)
        assert merged is defaults  # 짧은 경로 — 같은 참조

    def test_partial_override_only_affects_named_metric(self) -> None:
        defaults = _default_schedules()
        sched = _make_schedule(reminder_times={"bp": time(9, 30)})
        merged = resolve_reminder_schedules(sched, defaults)

        # bp 는 9:30 단일 target (delay 0) 로 변경
        bp = merged.metrics["bp"]
        assert len(bp.targets) == 1
        assert bp.targets[0].time_of_day == time(9, 30)
        assert bp.targets[0].delay_min == 0
        # weight / glucose 는 원본 유지
        assert merged.metrics["weight"].targets == defaults.metrics["weight"].targets
        assert merged.metrics["glucose"].targets == defaults.metrics["glucose"].targets

    def test_full_override_multiple_metrics(self) -> None:
        defaults = _default_schedules()
        sched = _make_schedule(
            reminder_times={"bp": time(8, 15), "weight": time(7, 0), "glucose": time(6, 30)},
        )
        merged = resolve_reminder_schedules(sched, defaults)
        assert merged.metrics["bp"].targets[0].time_of_day == time(8, 15)
        assert merged.metrics["weight"].targets[0].time_of_day == time(7, 0)
        assert merged.metrics["glucose"].targets[0].time_of_day == time(6, 30)

    def test_rate_limits_preserved(self) -> None:
        defaults = _default_schedules()
        sched = _make_schedule(reminder_times={"bp": time(9, 0)})
        merged = resolve_reminder_schedules(sched, defaults)
        assert merged.rate_limits == defaults.rate_limits
        assert merged.morning_briefing == defaults.morning_briefing


# ============================================================
# §5b  PR J — reminder_plans 프리셋 확장
# ============================================================
class TestReminderPlansResolve:
    def test_plan_with_multiple_times_produces_multiple_targets(self) -> None:
        """혈압 아침+저녁 프리셋 — targets 2개 생성."""
        defaults = _default_schedules()
        sched = _make_schedule(reminder_plans={
            "bp": ReminderPlan(
                active=True,
                days=(0, 1, 2, 3, 4, 5, 6),
                times=(time(8, 0), time(20, 0)),
                preset="morning_evening",
            ),
        })
        merged = resolve_reminder_schedules(sched, defaults)
        bp = merged.metrics["bp"]
        assert bp.active is True
        assert bp.days == (0, 1, 2, 3, 4, 5, 6)
        assert len(bp.targets) == 2
        assert bp.targets[0].time_of_day == time(8, 0)
        assert bp.targets[1].time_of_day == time(20, 0)
        assert bp.targets[0].delay_min == 0

    def test_plan_with_specific_weekdays(self) -> None:
        """혈당 주 2회 (화·금) 프리셋."""
        defaults = _default_schedules()
        sched = _make_schedule(reminder_plans={
            "glucose": ReminderPlan(
                active=True, days=(1, 4), times=(time(7, 0),),
                preset="twice_weekly_fasting",
            ),
        })
        merged = resolve_reminder_schedules(sched, defaults)
        g = merged.metrics["glucose"]
        assert g.active is True
        assert g.days == (1, 4)
        assert g.targets[0].time_of_day == time(7, 0)

    def test_off_preset_deactivates_metric(self) -> None:
        defaults = _default_schedules()
        sched = _make_schedule(reminder_plans={
            "bp": ReminderPlan(active=False, days=(), times=(), preset="off"),
        })
        merged = resolve_reminder_schedules(sched, defaults)
        assert merged.metrics["bp"].active is False

    def test_plans_take_priority_over_reminder_times(self) -> None:
        """동일 metric 에 plans + times 양쪽 있으면 plans 우선."""
        defaults = _default_schedules()
        sched = _make_schedule(
            reminder_times={"bp": time(9, 0)},
            reminder_plans={
                "bp": ReminderPlan(
                    active=True, days=(0, 1, 2, 3, 4, 5, 6),
                    times=(time(8, 0),), preset="morning_only",
                ),
            },
        )
        merged = resolve_reminder_schedules(sched, defaults)
        assert merged.metrics["bp"].targets[0].time_of_day == time(8, 0)

    def test_metric_without_plan_falls_back_to_legacy_then_yaml(self) -> None:
        defaults = _default_schedules()
        # bp 는 plan 있음 → 프리셋 적용
        # weight 는 plan 없고 legacy reminder_times 있음 → legacy 적용
        # glucose 는 둘 다 없음 → yaml 기본값 유지
        sched = _make_schedule(
            reminder_times={"weight": time(7, 0)},
            reminder_plans={
                "bp": ReminderPlan(
                    active=True, days=(0, 1, 2, 3, 4, 5, 6),
                    times=(time(8, 0),), preset="morning_only",
                ),
            },
        )
        merged = resolve_reminder_schedules(sched, defaults)
        assert merged.metrics["bp"].targets[0].time_of_day == time(8, 0)
        assert merged.metrics["weight"].targets[0].time_of_day == time(7, 0)
        assert merged.metrics["glucose"].targets == defaults.metrics["glucose"].targets


class TestFetchReminderPlans:
    def test_fetch_parses_reminder_plans_jsonb(self) -> None:
        row = {
            "user_id": UID,
            "briefing_time": "06:00",
            "reminder_times": {},
            "reminder_plans": {
                "bp": {
                    "preset": "morning_evening",
                    "days": [0, 1, 2, 3, 4, 5, 6],
                    "times": ["08:00", "20:00"],
                },
                "glucose": {
                    "preset": "twice_weekly_fasting",
                    "days": [1, 4],
                    "times": ["07:00"],
                    "context": "fasting",
                },
                "exercise": {"preset": "off"},
            },
            "weekly_day": 6,
            "weekly_time": "21:00",
            "timezone": "America/Chicago",
            "telegram_chat_id": None,
            "last_weekly_sent_at": None,
        }

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[row])

        sched = fetch_user_schedule(_mk_client(handler), UID)
        assert "bp" in sched.reminder_plans
        assert sched.reminder_plans["bp"].active is True
        assert sched.reminder_plans["bp"].days == (0, 1, 2, 3, 4, 5, 6)
        assert sched.reminder_plans["bp"].times == (time(8, 0), time(20, 0))
        assert sched.reminder_plans["glucose"].context == "fasting"
        assert sched.reminder_plans["exercise"].active is False

    def test_fetch_handles_missing_reminder_plans(self) -> None:
        row = {
            "user_id": UID, "briefing_time": "06:00", "reminder_times": {},
            "reminder_plans": None,
            "weekly_day": 6, "weekly_time": "21:00",
            "timezone": "America/Chicago",
            "telegram_chat_id": None, "last_weekly_sent_at": None,
        }

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[row])

        sched = fetch_user_schedule(_mk_client(handler), UID)
        assert sched.reminder_plans == {}

    def test_fetch_rejects_invalid_days_and_times(self) -> None:
        """잘못된 days/times 값은 필터링 — cron 을 망가뜨리지 않음."""
        row = {
            "user_id": UID, "briefing_time": "06:00", "reminder_times": {},
            "reminder_plans": {
                "bp": {
                    "preset": "custom",
                    "days": [0, 99, "not-a-number", 3],   # 99 와 문자열 제외
                    "times": ["08:00", "invalid", 1234],  # 'invalid'/숫자 제외
                },
            },
            "weekly_day": 6, "weekly_time": "21:00",
            "timezone": "America/Chicago",
            "telegram_chat_id": None, "last_weekly_sent_at": None,
        }

        def handler(_r: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=[row])

        sched = fetch_user_schedule(_mk_client(handler), UID)
        plan = sched.reminder_plans["bp"]
        assert plan.days == (0, 3)
        assert plan.times == (time(8, 0),)
        assert plan.active is True   # days + times 둘 다 비어있지 않으므로 active


# ============================================================
# §5  resolve_chat_id — prefs 우선, env fallback, 미해결 시 오류
# ============================================================
class TestResolveChatId:
    def test_prefs_chat_id_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
        sched = _make_schedule(telegram_chat_id=42)
        assert resolve_chat_id(sched) == 42

    def test_env_fallback_when_prefs_null(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "999")
        sched = _make_schedule(telegram_chat_id=None)
        assert resolve_chat_id(sched) == 999

    def test_explicit_env_fallback_param(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        sched = _make_schedule(telegram_chat_id=None)
        assert resolve_chat_id(sched, env_fallback=555) == 555

    def test_no_source_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        sched = _make_schedule(telegram_chat_id=None)
        with pytest.raises(ValueError, match="chat_id unresolved"):
            resolve_chat_id(sched)

    def test_env_bad_format_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "not-an-int")
        sched = _make_schedule(telegram_chat_id=None)
        with pytest.raises(ValueError, match="must be int"):
            resolve_chat_id(sched)
