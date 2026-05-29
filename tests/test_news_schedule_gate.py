# W8a PR #5 — should_send_news_briefing + mark_news_sent 테스트.
# _schedule.py 확장 커버리지.
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from execution.github_actions._schedule import (
    UserSchedule,
    should_send_news_briefing,
)


CT = ZoneInfo("America/Chicago")


def _make_schedule(
    briefing_time: time = time(6, 0),
    last_news_sent_at: datetime | None = None,
    news_briefing_time: time | None = None,
) -> UserSchedule:
    return UserSchedule(
        user_id="u1",
        briefing_time=briefing_time,
        reminder_times={},
        reminder_plans={},
        weekly_day=6,
        weekly_time=time(21, 0),
        timezone=CT,
        telegram_chat_id=None,
        last_weekly_sent_at=None,
        last_news_sent_at=last_news_sent_at,
        news_briefing_time=news_briefing_time,
    )


def _utc(local_hour: int, local_minute: int, local_date=(2026, 4, 20)) -> datetime:
    """CT 로 지정된 시각을 UTC datetime 으로 변환."""
    local = datetime(*local_date, local_hour, local_minute, tzinfo=CT)
    return local.astimezone(timezone.utc)


class TestScheduleGate:
    def test_before_briefing_time_returns_false(self) -> None:
        # 05:30 CT < 06:00 + 1 → 발송 안 함
        s = _make_schedule()
        assert should_send_news_briefing(s, _utc(5, 30)) is False

    def test_at_briefing_time_still_false_until_plus_1min(self) -> None:
        # 06:00 정각도 +1분 오프셋 때문에 false
        s = _make_schedule()
        assert should_send_news_briefing(s, _utc(6, 0)) is False

    def test_at_briefing_plus_1min_returns_true(self) -> None:
        s = _make_schedule()
        assert should_send_news_briefing(s, _utc(6, 1)) is True

    def test_later_same_day_still_true_if_not_sent(self) -> None:
        # 08:00 이어도 오늘 미발송이면 true.
        s = _make_schedule()
        assert should_send_news_briefing(s, _utc(8, 0)) is True

    def test_already_sent_today_returns_false(self) -> None:
        # 오늘 06:02 에 보냈고, 지금 08:00 → false
        sent = datetime(2026, 4, 20, 6, 2, tzinfo=CT).astimezone(timezone.utc)
        s = _make_schedule(last_news_sent_at=sent)
        assert should_send_news_briefing(s, _utc(8, 0)) is False

    def test_sent_yesterday_allows_today(self) -> None:
        # 어제 06:02 에 보냈고, 오늘 06:02 → true
        yesterday = datetime(2026, 4, 19, 6, 2, tzinfo=CT).astimezone(timezone.utc)
        s = _make_schedule(last_news_sent_at=yesterday)
        assert should_send_news_briefing(s, _utc(6, 2)) is True

    def test_custom_briefing_time(self) -> None:
        # 사용자가 briefing_time 을 09:00 으로 바꾸면 뉴스는 09:01+.
        s = _make_schedule(briefing_time=time(9, 0))
        assert should_send_news_briefing(s, _utc(8, 59)) is False
        assert should_send_news_briefing(s, _utc(9, 0)) is False
        assert should_send_news_briefing(s, _utc(9, 1)) is True


class TestIndependentNewsBriefingTime:
    """W8a hotfix 2026-04-21 — news_briefing_time 독립 설정."""

    def test_explicit_news_time_overrides_fallback(self) -> None:
        # briefing_time=05:00 이지만 news_briefing_time=07:30 → 07:30 발송.
        s = _make_schedule(briefing_time=time(5, 0), news_briefing_time=time(7, 30))
        assert should_send_news_briefing(s, _utc(5, 1)) is False   # 폴백 시각 지난 뒤지만 news_time 전
        assert should_send_news_briefing(s, _utc(7, 29)) is False
        assert should_send_news_briefing(s, _utc(7, 30)) is True
        assert should_send_news_briefing(s, _utc(8, 0)) is True

    def test_explicit_news_time_earlier_than_health(self) -> None:
        # 뉴스를 건강보다 먼저 — news 05:30, health 06:00.
        s = _make_schedule(briefing_time=time(6, 0), news_briefing_time=time(5, 30))
        assert should_send_news_briefing(s, _utc(5, 29)) is False
        assert should_send_news_briefing(s, _utc(5, 30)) is True

    def test_news_time_none_falls_back_to_briefing_plus_1(self) -> None:
        # 명시 없으면 기존 동작 (briefing + 1분).
        s = _make_schedule(briefing_time=time(6, 0), news_briefing_time=None)
        assert should_send_news_briefing(s, _utc(6, 0)) is False
        assert should_send_news_briefing(s, _utc(6, 1)) is True
