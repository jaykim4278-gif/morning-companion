# schedules.yaml 로더 — blueprint §2 W4 (리마인더 발송 판정) 의 유일한 구성 진입점.
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time
from functools import lru_cache
from pathlib import Path

import yaml

_DEFAULT_PATH = Path(__file__).resolve().parent / "schedules.yaml"


class ScheduleConfigError(ValueError):
    """schedules.yaml 이 필수 필드 누락·타입 불일치일 때."""


@dataclass(frozen=True)
class MetricTarget:
    time_of_day: time
    delay_min: int


@dataclass(frozen=True)
class MetricSchedule:
    code: str                       # 'bp', 'weight', 'glucose', 'exercise', 'sleep'
    label: str
    targets: tuple[MetricTarget, ...]
    active: bool
    note: str = ""
    # PR J — 주 N회 리마인더 지원. 0=월 ~ 6=일. 기본 매일 (하위 호환).
    days: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    # Stage 2 — context-aware 메시지용 (예: "fasting", "post_meal").
    context: str | None = None


@dataclass(frozen=True)
class BriefingSchedule:
    time_of_day: time
    weekdays: tuple[int, ...]       # 0=월 ~ 6=일
    weekday: int | None = None      # 주간 리포트용 (단일 요일)


@dataclass(frozen=True)
class RateLimits:
    max_reminders_per_day_per_metric: int
    reminder_window_start: time
    reminder_window_end: time


@dataclass(frozen=True)
class Schedules:
    version: str
    timezone: str
    metrics: dict[str, MetricSchedule]
    morning_briefing: BriefingSchedule
    weekly_report: BriefingSchedule
    rate_limits: RateLimits


def _parse_time(raw: str, field: str) -> time:
    try:
        hh, mm = raw.split(":")
        return time(int(hh), int(mm))
    except (ValueError, AttributeError) as e:
        raise ScheduleConfigError(f"invalid time '{raw}' at {field}") from e


def _parse_metric(code: str, raw: dict) -> MetricSchedule:
    try:
        targets = tuple(
            MetricTarget(
                time_of_day=_parse_time(t["time"], f"metrics.{code}.targets[].time"),
                delay_min=int(t["delay_min"]),
            )
            for t in raw.get("targets", [])
        )
        return MetricSchedule(
            code=code,
            label=str(raw["label"]),
            targets=targets,
            active=bool(raw.get("active", True)),
            note=str(raw.get("note", "")),
        )
    except (KeyError, TypeError) as e:
        raise ScheduleConfigError(f"metric '{code}': {e}") from e


def _load(path: Path) -> Schedules:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ScheduleConfigError(f"{path} root must be a mapping")

    try:
        metrics_raw = data["metrics"]
        metrics = {code: _parse_metric(code, raw) for code, raw in metrics_raw.items()}

        briefing_raw = data["briefing"]
        morning = BriefingSchedule(
            time_of_day=_parse_time(briefing_raw["morning"]["time"], "briefing.morning.time"),
            weekdays=tuple(int(x) for x in briefing_raw["morning"]["weekdays"]),
        )
        weekly = BriefingSchedule(
            time_of_day=_parse_time(briefing_raw["weekly"]["time"], "briefing.weekly.time"),
            weekdays=(int(briefing_raw["weekly"]["weekday"]),),
            weekday=int(briefing_raw["weekly"]["weekday"]),
        )

        rl_raw = data["rate_limits"]
        limits = RateLimits(
            max_reminders_per_day_per_metric=int(rl_raw["max_reminders_per_day_per_metric"]),
            reminder_window_start=_parse_time(rl_raw["reminder_window_start"], "rate_limits.start"),
            reminder_window_end=_parse_time(rl_raw["reminder_window_end"], "rate_limits.end"),
        )
    except KeyError as e:
        raise ScheduleConfigError(f"missing required field: {e}") from e

    return Schedules(
        version=str(data.get("version", "unknown")),
        timezone=str(data.get("timezone", "America/Chicago")),
        metrics=metrics,
        morning_briefing=morning,
        weekly_report=weekly,
        rate_limits=limits,
    )


@lru_cache(maxsize=4)
def load_schedules(path: str | Path | None = None) -> Schedules:
    """기본 경로 또는 테스트용 커스텀 경로에서 스케줄 로드.  lru_cache 로 2회차 이후 O(1)."""
    p = Path(path) if path else _DEFAULT_PATH
    return _load(p)


def is_briefing_day(sched: BriefingSchedule, d: date) -> bool:
    return d.weekday() in sched.weekdays


__all__ = [
    "MetricTarget",
    "MetricSchedule",
    "BriefingSchedule",
    "RateLimits",
    "Schedules",
    "ScheduleConfigError",
    "load_schedules",
    "is_briefing_day",
]
