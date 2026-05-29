# W9 PR #93 — should_send_english_phrase + mark_english_sent 테스트.
from __future__ import annotations

from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from execution.github_actions._schedule import (
    UserSchedule,
    should_send_english_phrase,
)


CT = ZoneInfo("America/Chicago")


def _schedule(
    *,
    english_phrase_time: time | None = None,
    last_english_sent_at: datetime | None = None,
) -> UserSchedule:
    return UserSchedule(
        user_id="u1",
        briefing_time=time(6, 0),
        reminder_times={},
        reminder_plans={},
        weekly_day=6,
        weekly_time=time(21, 0),
        timezone=CT,
        telegram_chat_id=None,
        last_weekly_sent_at=None,
        last_news_sent_at=None,
        news_briefing_time=None,
        english_phrase_time=english_phrase_time,
        last_english_sent_at=last_english_sent_at,
    )


def _utc_local(h: int, m: int, *, date=(2026, 5, 3)) -> datetime:
    return datetime(*date, h, m, tzinfo=CT).astimezone(timezone.utc)


class TestShouldSendEnglishPhrase:
    def test_default_disabled_when_time_none(self) -> None:
        # english_phrase_time NULL → 기능 자체 비활성 (default 비활성).
        s = _schedule(english_phrase_time=None)
        assert should_send_english_phrase(s, _utc_local(8, 0)) is False

    def test_before_target_time_returns_false(self) -> None:
        s = _schedule(english_phrase_time=time(6, 2))
        assert should_send_english_phrase(s, _utc_local(6, 1)) is False

    def test_at_target_time_returns_true(self) -> None:
        s = _schedule(english_phrase_time=time(6, 2))
        assert should_send_english_phrase(s, _utc_local(6, 2)) is True

    def test_already_sent_today_returns_false(self) -> None:
        sent = datetime(2026, 5, 3, 6, 3, tzinfo=CT).astimezone(timezone.utc)
        s = _schedule(english_phrase_time=time(6, 2), last_english_sent_at=sent)
        assert should_send_english_phrase(s, _utc_local(8, 0)) is False

    def test_sent_yesterday_allows_today(self) -> None:
        yesterday = datetime(2026, 5, 2, 6, 3, tzinfo=CT).astimezone(timezone.utc)
        s = _schedule(english_phrase_time=time(6, 2), last_english_sent_at=yesterday)
        assert should_send_english_phrase(s, _utc_local(6, 3)) is True
