# 아침 영어회화 한 마디 진입점.
#
# 흐름:
#   1) Schedule gate (english_phrase_time + 오늘 미발송)
#   2) 4주 테마 로테이션 (day-of-year 기반 결정론)
#   3) 최근 28일 phrase set 조회 (LLM 프롬프트 중복 회피용 컨텍스트)
#   4) LLM 호출 (Gemini → Groq → OpenRouter 3단 폴백)
#   5) Telegram 발송
#   6) english_phrase_log + last_english_sent_at 기록
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Protocol

from execution.ai_client.fallback_chain import (
    AllProvidersFailed,
    ChainConfig,
    Provider,
    call_with_fallback,
)

logger = logging.getLogger(__name__)


class Theme(str, Enum):
    """4주 테마 로테이션 — 한 주 동안 같은 주제.  미국 일상 회화 universal 테마."""
    SMALL_TALK = "small_talk"     # 일상 잡담·인사 (이웃·계산원·바리스타·동료)
    SOFT_SKILLS = "soft_skills"   # 부탁·거절·사과 (정중 사회 표현)
    REACTIONS = "reactions"       # 감정·반응 (맞장구·공감·솔직한 감정)
    IDIOMS = "idioms"             # 미국식 관용구·자연 표현


# 일자 기준 로테이션 — 매주 월요일 새 테마 시작.
_THEME_ROTATION: tuple[Theme, ...] = (
    Theme.SMALL_TALK,
    Theme.SOFT_SKILLS,
    Theme.REACTIONS,
    Theme.IDIOMS,
)


def select_theme_for_date(target: date) -> Theme:
    """target 일자의 ISO week number → 4 modulo → 테마.  결정론·예측 가능.

    예: 2026-W18 → 18 % 4 = 2 → DAILY
    """
    iso_week = target.isocalendar().week
    return _THEME_ROTATION[iso_week % 4]


# ============================================================
# 데이터 구조
# ============================================================
@dataclass(frozen=True)
class EnglishPhrase:
    """LLM 결과 — phrase 영문 + 한국어 직역 + 설명·사용예·발음·어원."""
    theme: Theme
    phrase_en: str           # "Sorry to keep you waiting"
    body_md: str             # 전체 메시지 본문 (한국어 직역 + 설명 + 사용예 + 발음 + 어원)


@dataclass(frozen=True)
class EnglishPhraseInputs:
    user_id: str
    as_of: date
    theme: Theme
    recent_phrases: Sequence[str] = ()    # 최근 28일 phrase_en (LLM 중복 회피용)
    chat_id: int | None = None


@dataclass(frozen=True)
class EnglishPhraseResult:
    sent: bool
    skipped: bool
    reason: str = ""
    phrase: EnglishPhrase | None = None
    ai_provider: str = ""


# ============================================================
# Protocols
# ============================================================
class EnglishPhraseStore(Protocol):
    def fetch_recent_phrases(self, user_id: str, days: int) -> list[str]:
        """최근 N일 발송된 phrase_en 리스트 (LLM 중복 회피용)."""
        ...

    def record_sent(self, user_id: str, phrase: EnglishPhrase, sent_at: datetime) -> None:
        """발송 후 english_phrase_log 기록 + last_english_sent_at upsert."""
        ...


class TelegramSender(Protocol):
    def send(self, chat_id: int, text: str) -> None: ...


# ============================================================
# 유틸
# ============================================================
def hash_phrase(phrase_en: str) -> str:
    """SHA256(lower(trim(phrase_en))) — DB UNIQUE 키와 일치."""
    norm = (phrase_en or "").strip().lower()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


# ============================================================
# 메인 entrypoint (LLM 분리 — PR #91 mock 가능, PR #92 에서 실 LLM 호출 채움)
# ============================================================
def run_english_phrase(
    inp: EnglishPhraseInputs,
    *,
    store: EnglishPhraseStore,
    telegram: TelegramSender,
    providers: Sequence[Provider],
    chain_config: ChainConfig | None = None,
    generator: "EnglishPhraseGenerator | None" = None,
    now_utc: datetime | None = None,
) -> EnglishPhraseResult:
    """영어회화 한 마디 발송 진입점.

    generator: PR #92 에서 LLM 기반 EnglishPhraseGenerator 주입.  None 이면 ImportError —
    이 PR (#91) 는 schema + 데이터 구조 + entrypoint 골격만 머지하고 LLM 은 다음 PR.
    """
    now_utc = now_utc or datetime.now(timezone.utc)

    if generator is None:
        # PR #91 단계 — generator 미주입 시 skip (호출자가 처리).
        return EnglishPhraseResult(
            sent=False, skipped=True, reason="generator not provided (PR #92 to add)",
        )

    if inp.chat_id is None:
        return EnglishPhraseResult(sent=False, skipped=True, reason="chat_id missing")

    try:
        phrase = generator.generate(inp, providers, config=chain_config)
    except AllProvidersFailed as exc:
        logger.warning("영어회화 LLM 3단 폴백 모두 실패: %s", exc)
        return EnglishPhraseResult(
            sent=False, skipped=False, reason=f"llm_failed: {exc}",
        )

    try:
        telegram.send(inp.chat_id, phrase.body_md)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Telegram 발송 실패: %s", exc)
        return EnglishPhraseResult(
            sent=False, skipped=False, reason=f"telegram: {exc}", phrase=phrase,
        )

    store.record_sent(inp.user_id, phrase, now_utc)

    return EnglishPhraseResult(
        sent=True, skipped=False, phrase=phrase,
        ai_provider=getattr(phrase, "_provider", ""),
    )


# Generator Protocol — PR #92 에서 실 구현 추가.
class EnglishPhraseGenerator(Protocol):
    def generate(
        self,
        inp: EnglishPhraseInputs,
        providers: Sequence[Provider],
        *,
        config: ChainConfig | None = None,
    ) -> EnglishPhrase: ...


__all__ = [
    "Theme",
    "select_theme_for_date",
    "hash_phrase",
    "EnglishPhrase",
    "EnglishPhraseInputs",
    "EnglishPhraseResult",
    "EnglishPhraseStore",
    "EnglishPhraseGenerator",
    "TelegramSender",
    "run_english_phrase",
]
