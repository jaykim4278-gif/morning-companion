# 날씨 기반 한국어 코멘트·권고.
#
# 오늘 기온·강수·이번 주 흐름을 한국어 비서 톤으로 요약.
# 전략: LLM 1차 → 실패 시 규칙 기반 폴백 (안전 fallback, 메시지 발송 보장).
from __future__ import annotations

import logging
from collections.abc import Sequence

from execution.ai_client.fallback_chain import (
    AllProvidersFailed,
    ChainConfig,
    Provider,
    call_with_fallback,
)
from execution.github_actions.news.translator import contains_foreign_cjk
from execution.github_actions.news.weather import (
    DailyForecast,
    WeatherAlert,
    WeatherSnapshot,
    WeatherWeek,
)

logger = logging.getLogger(__name__)


ADVISORY_PROMPT = """당신은 사용자의 거주 지역 날씨를 안내하는 친절한 비서입니다.
오늘 날씨와 7일 예보를 보고 2~3문장의 실용적인 한국어 안내를 드리세요.

⚠️ 어투 (매우 중요):
- 반드시 존댓말.  "~입니다 / ~이십니다 / ~습니다 / ~하세요 / ~하시면 좋겠습니다 / ~준비하시기 바랍니다" 등
- 친절하고 정중한 비서 톤.  "~하는 것이 좋겠다 / ~이다 / ~한다" 같은 반말·평서문 절대 금지
- 담담하지만 살갑게 (과장·감탄 X, "오늘도" "편안한 하루 보내세요" 같은 자연스러운 인사 OK)

반드시 포함:
- 오늘 옷차림 권고 (긴팔·반팔·재킷 등, 실제 기온 반영)
- 강수/악천후 대비 (우산·장화·운전 주의 등)
- 이번 주 흐름 간단 요약 (계속 비 / 주말 맑음 등)
- 경보가 있으면 가장 눈에 띄게 먼저 언급 (토네이도·홍수 등)

금지:
- 영어 섞기 금지 (매체명·고유명사 제외)
- 의학·건강 조언 (본 메시지는 별도 건강 브리핑에서 다룸)
- 🚫 한자·일본어·중국어 문자 절대 금지 ('氏·人·物·雨·風' 등은 '씨·사람·물건·비·바람' 으로)
- 🚫 반말 금지: "좋겠다" → "좋으시겠습니다",  "한다" → "합니다",  "이다" → "입니다"

좋은 예시:
"오늘은 63°F 로 쌀쌀하오니 긴팔에 얇은 재킷을 준비하시면 좋겠습니다.
비가 하루 종일 이어지니 우산도 챙기시기 바랍니다.
이번 주는 목요일까지 흐림·비 예보이니 우비·장화를 두어 두시면 편하실 것 같습니다."

나쁜 예시 (반말·평서문):
"오늘은 긴팔 입는 것이 좋겠다. 비가 올 것으로 보이니 우산 준비해라.
이번 주는 계속 흐릴 것이다."

【오늘 날씨】
기온 {today_f:.0f}°F (한국식 라벨 '{today_k}')
최고 {high_f:.0f}°F / 최저 {low_f:.0f}°F, 강수확률 {precip_prob}%

【이번 주 7일 흐름】
{week_lines}

【기상경보】
{alert_lines}

출력 규칙:
- 2~3문장, 총 200자 이내
- 평문 (마크다운·따옴표·이모지 금지)
- 말미 마침표 O
- 존댓말 필수"""


def _format_week_lines(week: WeatherWeek) -> str:
    """주간 예보를 LLM 이 읽기 쉬운 포맷으로."""
    if not week.days:
        return "(주간 예보 없음)"
    return "\n".join(
        f"- {d.date_iso}: {d.weather_label}, {d.low_f:.0f}~{d.high_f:.0f}°F, 강수{d.precip_prob_pct}%"
        for d in week.days
    )


def _format_alert_lines(alerts: Sequence[WeatherAlert]) -> str:
    if not alerts:
        return "(없음)"
    return "\n".join(
        f"- [{a.severity}] {a.event}: {a.headline[:140]}"
        for a in alerts[:3]
    )


def generate_weather_advisory(
    snapshot: WeatherSnapshot | None,
    week: WeatherWeek | None,
    alerts: Sequence[WeatherAlert],
    providers: Sequence[Provider],
    *,
    config: ChainConfig | None = None,
) -> str | None:
    """LLM 1회 호출로 2~3문장 한국어 조언 생성.

    반환:
      - 성공: 조언 문자열 (말미 개행 없음)
      - 실패: 규칙 기반 폴백 문장 (`_heuristic_advisory`)
      - snapshot=None → None (호출자는 advisory 섹션 생략)
    """
    if snapshot is None:
        return None
    if not providers:
        return _heuristic_advisory(snapshot, week, alerts)

    cfg = config or ChainConfig(max_retries_per_provider=1, base_backoff_seconds=0.5)
    prompt = ADVISORY_PROMPT.format(
        today_f=snapshot.temp_f,
        today_k=snapshot.weather_label,
        high_f=snapshot.today_high_f,
        low_f=snapshot.today_low_f,
        precip_prob=snapshot.precip_prob_pct,
        week_lines=_format_week_lines(week or WeatherWeek(days=())),
        alert_lines=_format_alert_lines(alerts),
    )
    # anonymize_payload 화이트리스트 "news" 재사용 (PII 없음).
    payload = {"news": {"weather": snapshot.temp_line}}
    try:
        result = call_with_fallback(
            payload=payload,
            prompt=prompt,
            providers=list(providers),
            drug_map={},
            config=cfg,
        )
    except AllProvidersFailed as exc:
        logger.info("날씨 조언 LLM 실패 — 규칙 기반 폴백: %s", exc)
        return _heuristic_advisory(snapshot, week, alerts)

    text = (result.text or "").strip().strip("*_`\"'")
    if not text:
        return _heuristic_advisory(snapshot, week, alerts)

    # 2026-04-21 hotfix — CJK(한자·가나) 포함 시 stricter retry 1회, 그래도 CJK 면 heuristic.
    if contains_foreign_cjk(text):
        logger.info("advisory CJK 포함 — stricter retry: %s", text[:80])
        strict_prompt = (
            prompt
            + "\n\n🚫 매우 중요: 한자·일본어 절대 사용 금지. 순수 한글로 재작성하세요."
        )
        try:
            retry = call_with_fallback(
                payload=payload, prompt=strict_prompt, providers=list(providers),
                drug_map={}, config=cfg,
            )
            retry_text = (retry.text or "").strip().strip("*_`\"'")
            if retry_text and not contains_foreign_cjk(retry_text):
                return retry_text[:400]
        except AllProvidersFailed:
            pass
        logger.info("advisory CJK 재시도 후에도 한자 — heuristic 폴백")
        return _heuristic_advisory(snapshot, week, alerts)

    return text[:400]


# ============================================================
# 규칙 기반 폴백 — LLM 불가 / 키 없음 시.
# ============================================================
def _heuristic_advisory(
    snapshot: WeatherSnapshot,
    week: WeatherWeek | None,
    alerts: Sequence[WeatherAlert],
) -> str:
    """간단 if/else 로 1~2문장 생성.  LLM 성공률 낮을 때 최종 안전판.

    2026-04-21 hotfix — 비서 톤 존댓말로 통일.
    """
    parts: list[str] = []

    # 1) 경보 우선.
    if alerts:
        top = alerts[0]
        parts.append(
            f"현재 {top.event} 기상경보가 발령 중입니다. 외출 전 최신 정보 확인하시기 바랍니다."
        )

    # 2) 기온 구간별 옷차림.
    temp_f = snapshot.temp_f
    if temp_f <= 45:
        parts.append(
            f"오늘은 {temp_f:.0f}°F 로 추우니 두꺼운 외투와 목도리를 챙겨 나가시기 바랍니다."
        )
    elif temp_f <= 60:
        parts.append(
            f"오늘은 {temp_f:.0f}°F 로 쌀쌀하오니 긴팔에 가벼운 재킷을 준비하시면 좋겠습니다."
        )
    elif temp_f <= 75:
        parts.append(
            f"오늘은 {temp_f:.0f}°F 로 쾌적합니다. 얇은 긴팔 또는 반팔이 적당하시겠습니다."
        )
    elif temp_f <= 88:
        parts.append(
            f"오늘은 {temp_f:.0f}°F 로 따뜻합니다. 가벼운 옷차림에 수분도 충분히 챙기시기 바랍니다."
        )
    else:
        parts.append(
            f"오늘은 {temp_f:.0f}°F 로 더운 날씨입니다. 직사광선 피하시고 수분 자주 섭취하시기 바랍니다."
        )

    # 3) 강수 대비.
    if snapshot.precip_prob_pct >= 60:
        parts.append(
            f"강수 확률 {snapshot.precip_prob_pct}% 이니 우산을 꼭 챙기시기 바랍니다."
        )
    elif snapshot.precip_prob_pct >= 30:
        parts.append(
            f"강수 확률 {snapshot.precip_prob_pct}% 이니 우산을 챙기시면 안심하실 것 같습니다."
        )

    # 4) 주간 흐름.
    if week and len(week.days) >= 3:
        rain_days = sum(1 for d in week.days[1:] if d.precip_prob_pct >= 50)
        if rain_days >= 4:
            parts.append(
                "이번 주는 대부분 비 예보이니 장기간 우천에 대비해 두시면 좋겠습니다."
            )
        elif rain_days >= 2:
            parts.append(
                "이번 주 중반에 비 예보가 있으니 참고해 두시기 바랍니다."
            )
        else:
            weekend_codes = [d.weather_label for d in week.days[-2:]]
            parts.append(
                f"주말은 {' · '.join(weekend_codes)} 예보입니다."
            )

    return " ".join(parts)[:400]


__all__ = [
    "ADVISORY_PROMPT",
    "generate_weather_advisory",
]
