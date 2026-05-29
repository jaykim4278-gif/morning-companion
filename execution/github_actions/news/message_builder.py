# W8a PR #5 — Telegram MarkdownV2 메시지 조립.
# docs/design/w8a-news-briefing-design.md §7 근거.
#
# MarkdownV2 escape: _*[]()~`>#+-=|{}.! 전부 \ 로.
# 링크 URL 안에서는 ) 와 \ 만 escape.
from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timezone
from typing import Any

from execution.github_actions.news.dedup_url import normalize_url
from execution.github_actions.news.market_indices import IndexQuote
from execution.github_actions.news.rss_fetcher import RssItem
from execution.github_actions.news.sources import (
    CATEGORY_HEADERS,
    CATEGORY_ORDER,
    Category,
)
from execution.github_actions.news.weather import WeatherAlert, WeatherSnapshot

# Telegram MarkdownV2 특수문자 (escape 필수).
# 참조: https://core.telegram.org/bots/api#markdownv2-style
_MDV2_SPECIAL = "_*[]()~`>#+-=|{}.!"

# URL 내부에서만 escape.
_URL_SPECIAL = ")\\"

# 4096자 Telegram 메시지 상한 — 안전 여유 두고 4000자 cap.
_MAX_MESSAGE_LEN = 4000

# 제목 길이 상한 (너무 긴 제목 truncate).
_MAX_TITLE_LEN = 120

# 메시지에 노출하는 요약 길이 상한 (rss_fetcher 는 260 자까지 추출 → 여기서 더 줄임).
_MAX_SUMMARY_DISPLAY_LEN = 180


_WEEKDAY_KR = "월화수목금토일"


def escape_mdv2(text: str) -> str:
    """일반 텍스트 / 링크 텍스트용 escape."""
    return "".join(f"\\{c}" if c in _MDV2_SPECIAL else c for c in text)


def _flatten_inline(text: str) -> str:
    """multi-line 텍스트를 single-line 으로 평탄화 — italic/blockquote 짝맞춤 보호.

    2026-05-14 hotfix — 5/14 06:08 CT cron 실패 root cause.
    Telegram MarkdownV2 의 `>_text_` (blockquote italic) 구조에서 text 안에 `\\n`이
    있으면 line 1: blockquote+italic open, line 2: no `>` 라서 blockquote 가 끝나는데
    italic 은 아직 open 상태 → "Can't find end of Italic entity at byte offset N" 400.

    LLM 출력 (weather advisory / news summary 번역) 이 줄바꿈을 포함할 때 발현.
    NWS alert headline 도 가끔 \\n 포함.  inline 컨텐츠는 모두 평탄화로 보호.
    """
    return " ".join(text.split())


def escape_url(url: str) -> str:
    """링크 URL 내부용 escape — ) 와 \\ 만."""
    return "".join(f"\\{c}" if c in _URL_SPECIAL else c for c in url)


def link(text: str, url: str) -> str:
    """MarkdownV2 inline link [텍스트](URL)."""
    return f"[{escape_mdv2(text)}]({escape_url(url)})"


def _truncate(text: str, limit: int = _MAX_TITLE_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _fmt_change_arrow(pct: float) -> str:
    """+0.85 → '▲ \\+0\\.85' / -0.32 → '▼ \\-0\\.32'"""
    arrow = "▲" if pct >= 0 else "▼"
    sign = "+" if pct >= 0 else ""
    return f"{arrow} {escape_mdv2(f'{sign}{pct:.2f}')}"


def _fmt_index_line(q: IndexQuote) -> str:
    """• S&P 500: 5,123\\.45 ▲ \\+0\\.85% \\(YTD \\+8\\.2%\\)"""
    name = escape_mdv2(q.display_name)
    close_txt = escape_mdv2(f"{q.last_close:,.2f}")
    day_arrow = _fmt_change_arrow(q.day_change_pct)
    ytd_sign = "+" if q.ytd_change_pct >= 0 else ""

    if q.unit == "%":
        # yield 심볼 (^TNX) — 종가가 이미 %.  일일 변동은 %p.
        close_part = f"{close_txt}{escape_mdv2('%')}"
        ytd_part = escape_mdv2(f"({ytd_sign}{q.ytd_change_pct:.2f}%p YTD)")
        day_part = f"{day_arrow}{escape_mdv2('%p')}"
    else:
        close_part = close_txt
        ytd_part = escape_mdv2(f"(YTD {ytd_sign}{q.ytd_change_pct:.2f}%)")
        day_part = f"{day_arrow}{escape_mdv2('%')}"

    return f"• {name}: {close_part} {day_part} {ytd_part}"


def _fmt_item_line(item: RssItem) -> list[str]:
    """제목 굵게 + 매체명·자세히 메타 라인 + (선택) 요약 italic blockquote.

    2026-05-13 PR — 가독성 redesign (사용자 5/13 피드백 "파란색 글자라 아침에 읽기 너무 힘들어요"):
    - 이전: `• [Reuters] [title](url)` — title 이 link 라서 Telegram 이 파란색·밑줄 적용
    - 신규: title 을 bold (`*title*`) 만 적용 — Telegram 의 bold 는 테마 자동 적응
      (dark mode → 흰색 굵게, light mode → 검정 굵게 — 항상 고대비)
    - URL 은 "자세히 →" 단일 단어 링크로 분리 — 파란색은 그 단어 하나로 축소
    - 매체명은 italic + 📰 이모지로 시각적 종속 표현

    Example:
        *기사 제목 여기*
           📰 _Reuters_  ·  [자세히 →](URL)
           >_요약 italic_

    2026-05-03 정책 유지 — 영문 폴백 차단:
    - title 은 translator.translate_category 가 한국어만 보장 → title_ko 우선
    - summary 는 summary_ko 만 렌더, 없으면 라인 자체 생략 (영문 summary_en 폴백 금지)
    """
    from execution.github_actions.news.sources import by_slug

    src = by_slug(item.source)
    display_name = src.display_name if src else f"[{item.source}]"
    # 옛 디자인의 "[Reuters]" 대괄호는 redesign 에선 떼고 매체명만.
    clean_name = display_name.strip("[]")
    title_text = item.title_ko or item.title_en
    # 2026-05-14 — LLM 번역 출력이 가끔 \\n 포함 → bold/italic 멀티라인 안전망.
    title_text = _flatten_inline(title_text)
    title_text = _truncate(title_text, _MAX_TITLE_LEN)

    # 2026-05-14 hotfix — utm_* 등 트래킹 파라미터 포함 URL 의 raw `_` 가 Telegram
    # MarkdownV2 italic 파서를 트리거해 "Can't find end of Italic entity" 400 발생
    # (5/14 06:08 CT cron 실패).  Telegram docs 상 `(url)` 안 `_` 는 literal 이어야 하나
    # 실측에서 mismatch.  normalize_url 로 utm_*·fbclid·gclid 등 제거해 URL 내 `_` 최소화
    # (정상 link 동작 유지 — UTM 은 analytics 용, 서버 content 결정에 영향 없음).
    rendered_url = normalize_url(item.url)
    lines = [
        f"*{escape_mdv2(title_text)}*",
        f"  📰 _{escape_mdv2(clean_name)}_  ·  {link('자세히 →', rendered_url)}",
    ]
    # 요약 라인 — 한국어 번역본만 렌더.  영문 summary 는 사용자 노출 금지.
    summary_text = item.summary_ko or ""
    if summary_text:
        # 2026-05-14 hotfix — LLM 출력에 \\n 포함 시 `>_...\\n..._` 가 blockquote 끝나면서
        # italic 만 살아남아 파서 실패 → 평탄화로 single-line 강제.
        flat = _flatten_inline(summary_text)
        truncated = _truncate(flat, _MAX_SUMMARY_DISPLAY_LEN)
        # blockquote (`>`) 로 시각적 들여쓰기 + italic 톤.
        lines.append(f">_{escape_mdv2(truncated)}_")
    return lines


def _build_indices_section(
    indices: list[IndexQuote],
    *,
    indices_failed: bool,
) -> list[str]:
    lines: list[str] = []
    if not indices and indices_failed:
        lines.append(f"*{escape_mdv2(CATEGORY_HEADERS[Category.INDEX])}*")
        lines.append(escape_mdv2("• 지수 데이터 일시 불가"))
        lines.append("")
        return lines
    if not indices:
        return lines
    # 종가 기준일 — 가장 오래된 as_of 써서 안전한 표기 (일부 심볼 휴장 시 불일치 회피).
    as_of = min(q.as_of for q in indices)
    header = f"{CATEGORY_HEADERS[Category.INDEX]} ({as_of.month}월 {as_of.day}일 종가)"
    lines.append(f"*{escape_mdv2(header)}*")
    for q in indices:
        lines.append(_fmt_index_line(q))
    lines.append("")
    return lines


def _build_weather_section(
    weather: WeatherSnapshot | None,
    alerts: Sequence[WeatherAlert],
    advisory: str | None = None,
) -> list[str]:
    """⛅ 오늘 날씨 섹션 (advisory blockquote 포함, weather=None 이면 전체 생략)."""
    if weather is None and not alerts and not advisory:
        return []
    lines: list[str] = [f"*{escape_mdv2(CATEGORY_HEADERS[Category.WEATHER])}*"]
    if weather is not None:
        lines.append("• " + escape_mdv2(weather.temp_line))
        lines.append("• " + escape_mdv2(weather.range_line))
    # NWS 경보 blockquote (severity 최상위 1건).
    if alerts:
        top = alerts[0]
        severity_icon = {
            "Extreme": "🚨", "Severe": "⚠️",
            "Moderate": "⚠", "Minor": "ℹ️", "Unknown": "ℹ️",
        }.get(top.severity, "ℹ️")
        headline = top.headline or top.event
        text = f"{severity_icon} {top.event} · {headline}"
        # NWS API headline 에 \\n 포함 케이스 방어.
        lines.append(">_" + escape_mdv2(_flatten_inline(text)[:240]) + "_")
    # 날씨 조언 blockquote (LLM 또는 규칙 기반).
    if advisory:
        # 2026-05-14 hotfix — LLM advisory 에 \\n 포함 시 multi-line 으로 들어가
        # `>_<line1>\\n<line2>_` 구조가 되어 line 2 가 blockquote 밖 → italic 만 살아남아
        # 파서 실패 (5/14 06:08 CT cron 의 byte offset 408 root cause).
        lines.append(">_" + escape_mdv2(_flatten_inline(advisory)[:400]) + "_")
    lines.append("")
    return lines


def _build_category_section(cat: Category, items: Sequence[RssItem]) -> list[str]:
    if not items:
        return []
    lines: list[str] = [f"*{escape_mdv2(CATEGORY_HEADERS[cat])}*"]
    for it in items:
        lines.extend(_fmt_item_line(it))
    lines.append("")
    return lines


def build_message(
    indices: list[IndexQuote],
    selected: dict[Category, list[RssItem]],
    *,
    now: datetime | None = None,
    indices_failed: bool = False,
    weather: WeatherSnapshot | None = None,
    alerts: Sequence[WeatherAlert] = (),
    weather_advisory: str | None = None,
) -> str:
    """Telegram MarkdownV2 최종 메시지.

    Args:
        indices: yfinance 결과.  빈 리스트 + indices_failed=True → "일시 불가".
        selected: 카테고리별 최종 선정 아이템 (cluster_and_select + translate_category 결과).
        now: 메시지 발송 시각 (로컬 헤더 렌더용).  None → UTC now.
        indices_failed: 지수 fetch 전면 실패 여부 (빈 리스트와 구분).

    4000자 초과 시 CATEGORY_ORDER 역순으로 카테고리 drop (LOCAL → RESTAURANT → TECH → ...).
    """
    now = now or datetime.now(timezone.utc)
    weekday = _WEEKDAY_KR[now.weekday()]

    def _assemble(drop_cats: set[Category]) -> str:
        lines: list[str] = [
            f"📰 *{escape_mdv2(f'미국 뉴스 브리핑 — {now.month}월 {now.day}일 ({weekday})')}*",
            "",
        ]
        for cat in CATEGORY_ORDER:
            if cat in drop_cats:
                continue
            if cat == Category.INDEX:
                lines.extend(_build_indices_section(indices, indices_failed=indices_failed))
                continue
            if cat == Category.WEATHER:
                lines.extend(_build_weather_section(weather, alerts, weather_advisory))
                continue
            lines.extend(_build_category_section(cat, selected.get(cat, [])))
        # footer — 발송 시각 (로컬 타임존으로 now 가 들어온다고 가정 · naive 표기).
        # "Gemini 번역" 은 제목 아래 🇰🇷 로 묶어 사용자 혼란 최소.
        hour = now.hour
        minute = now.minute
        footer = f"🤖 {hour:02d}:{minute:02d} CT · 제목·요약 AI 번역 (LLM 실패 시 사전 폴백 — 영문 미노출)"
        lines.append(escape_mdv2(footer))
        return "\n".join(lines).rstrip()

    message = _assemble(set())
    if len(message) <= _MAX_MESSAGE_LEN:
        return message

    # 길이 초과 — 역순 drop.
    drop = set()
    for cat in reversed(CATEGORY_ORDER):
        if cat == Category.INDEX:
            continue  # 지수는 항상 유지 (짧음)
        drop.add(cat)
        message = _assemble(drop)
        if len(message) <= _MAX_MESSAGE_LEN:
            break
    return message


__all__ = [
    "escape_mdv2",
    "escape_url",
    "link",
    "build_message",
]
