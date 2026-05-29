# Layer 4 Sensor — schedules.yaml 로더 무결성.
from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pytest
import yaml

from src.config.schedules import (
    ScheduleConfigError,
    is_briefing_day,
    load_schedules,
)


# ==================== 기본 YAML 로딩 ====================

def test_default_file_loads_and_has_version() -> None:
    s = load_schedules()
    assert s.version == "2026-05-26"
    assert s.timezone == "America/Chicago"


def test_all_expected_metrics_present() -> None:
    s = load_schedules()
    for code in ("bp", "weight", "glucose", "exercise", "sleep"):
        assert code in s.metrics


def test_bp_has_single_morning_target() -> None:
    """2026-05-26 PR A — 저녁 target 제거, delay_min 으로 cutoff 를 10:00 으로 조정."""
    s = load_schedules()
    bp = s.metrics["bp"]
    assert bp.active is True
    assert len(bp.targets) == 1                     # 저녁 target (20:00) 제거됨
    assert bp.targets[0].time_of_day == time(8, 0)
    assert bp.targets[0].delay_min == 120           # 08:00 + 120m = 10:00 cutoff


def test_weight_glucose_cutoff_aligned_to_ten() -> None:
    """2026-05-26 PR A — weight/glucose 도 10:00 cutoff 정렬."""
    s = load_schedules()
    w = s.metrics["weight"]
    g = s.metrics["glucose"]
    assert w.active is True
    assert w.targets[0].time_of_day == time(7, 30)
    assert w.targets[0].delay_min == 150            # 07:30 + 150m = 10:00
    assert g.active is True
    assert g.targets[0].time_of_day == time(7, 0)
    assert g.targets[0].delay_min == 180            # 07:00 + 180m = 10:00


def test_exercise_is_inactive() -> None:
    """2026-05-26 PR A — 저녁 알림 제거 정책으로 운동 리마인더 비활성."""
    s = load_schedules()
    ex = s.metrics["exercise"]
    assert ex.active is False


def test_sleep_is_inactive_and_empty_targets() -> None:
    s = load_schedules()
    sleep = s.metrics["sleep"]
    assert sleep.active is False
    assert sleep.targets == ()


def test_briefing_morning_is_daily() -> None:
    s = load_schedules()
    assert s.morning_briefing.time_of_day == time(6, 0)
    assert set(s.morning_briefing.weekdays) == set(range(7))


def test_weekly_report_is_sunday_21() -> None:
    s = load_schedules()
    assert s.weekly_report.time_of_day == time(21, 0)
    assert s.weekly_report.weekday == 6


def test_rate_limits_window_narrowed_to_ten_am() -> None:
    """2026-05-26 PR A — reminder_window 를 10:00~11:30 으로 좁힘 (UTC 15+16 cron 양쪽 포함)."""
    s = load_schedules()
    assert s.rate_limits.max_reminders_per_day_per_metric == 1
    assert s.rate_limits.reminder_window_start == time(10, 0)
    assert s.rate_limits.reminder_window_end == time(11, 30)


# ==================== is_briefing_day ====================

def test_is_briefing_day_morning_is_every_day() -> None:
    s = load_schedules()
    for d in (date(2026, 4, 13), date(2026, 4, 14), date(2026, 4, 19)):
        assert is_briefing_day(s.morning_briefing, d) is True


def test_is_briefing_day_weekly_only_sunday() -> None:
    s = load_schedules()
    assert is_briefing_day(s.weekly_report, date(2026, 4, 19)) is True    # 일요일
    assert is_briefing_day(s.weekly_report, date(2026, 4, 13)) is False   # 월요일


# ==================== 에러 경로 ====================

def test_missing_required_field_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("metrics: {}\nbriefing: {morning: {time: '06:00', weekdays: [0]}}\n", encoding="utf-8")
    with pytest.raises(ScheduleConfigError):
        load_schedules(bad)


def test_bad_time_format_raises(tmp_path: Path) -> None:
    raw = yaml.safe_dump({
        "metrics": {"bp": {"label": "x", "targets": [{"time": "25:99", "delay_min": 10}]}},
        "briefing": {
            "morning": {"time": "06:00", "weekdays": [0, 1, 2, 3, 4, 5, 6]},
            "weekly":  {"time": "21:00", "weekday": 6},
        },
        "rate_limits": {
            "max_reminders_per_day_per_metric": 1,
            "reminder_window_start": "06:00",
            "reminder_window_end": "22:00",
        },
    })
    bad = tmp_path / "bad.yaml"
    bad.write_text(raw, encoding="utf-8")
    with pytest.raises(ScheduleConfigError):
        load_schedules(bad)


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ScheduleConfigError):
        load_schedules(bad)
