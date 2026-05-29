# W9 PR — 영어회화 JSON → Telegram HTML 메시지 렌더링 + TTS 링크 임베드.
#
# 2026-05-14 사용자 요청 (PR B 의 핵심):
# - 모든 영문 문장 (phrase + scenarios + responses) 에 🔊 TTS 음성 링크 부착
# - "대답 예시" 신규 섹션 (responses)
#
# parse_mode 전환: plain text → HTML
#   · MarkdownV2 는 escape 사고 잦음 (PR #107 italic-blockquote bug 직전 학습)
#   · HTML 은 <, >, & 만 entity encode 하면 안전 — 본문 한국어 95% 가 escape 무관
#   · 하이퍼링크는 <a href="URL">text</a> 로 안전 임베드
#
# TTS 흐름:
#   1) renderer 가 영문 문장 리스트 수집 → tts_uploader.exists 로 캐시 hit 확인
#   2) miss 시 synthesize_mp3(text) → upload_or_get_url(text, audio)
#   3) hit 시 get_url(text) 만 호출 (합성 skip)
#   4) HTML <a href="url">🔊 듣기</a> 문자열 임베드
#
# 장애 정책 (사용자 승인):
#   - TTS 전부 실패 시 음성 링크 없이 메시지 발송 (graceful degradation)
#   - 개별 문장 실패 시 그 문장만 링크 생략
from __future__ import annotations

import logging
from typing import Protocol

from execution.github_actions.english_phrase_llm import (
    EnglishPhraseData,
    ResponseExample,
    Scenario,
)

logger = logging.getLogger(__name__)


# ============================================================
# Protocols — 의존성 주입.
# ============================================================
class TtsSynthesizer(Protocol):
    """edge-tts 합성 — tts.synthesize_mp3 시그니처 호환."""
    def __call__(self, text: str) -> bytes: ...


class TtsUploader(Protocol):
    """Supabase Storage 업로드/조회 — _tts_storage.TtsStorageUploader 호환."""
    def exists(self, text: str) -> bool: ...
    def upload_or_get_url(self, text: str, audio_bytes: bytes) -> str: ...
    def get_url(self, text: str) -> str: ...


# ============================================================
# HTML escape — Telegram parse_mode=HTML 용 (<, >, & 만).
# ============================================================
def escape_html(text: str) -> str:
    """Telegram HTML parse_mode 의 entity escape — 보수적으로 3종만.

    Telegram HTML mode 규칙 (https://core.telegram.org/bots/api#html-style):
      - <, >, & 는 &lt; &gt; &amp; 로 인코딩
      - 그 외 모든 문자는 literal
    """
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ============================================================
# 테마 라벨 (renderer 내 sync — english_phrase_llm._THEME_LABEL 과 일치 유지).
# ============================================================
from execution.github_actions.english_phrase import Theme

_THEME_LABEL: dict[Theme, str] = {
    Theme.SMALL_TALK: "💬 일상 잡담·인사 (어디서든 만나는 사람들과)",
    Theme.SOFT_SKILLS: "🤝 부탁·거절·사과 (정중하게 의사 전달)",
    Theme.REACTIONS: "😊 감정·반응 (맞장구·공감·솔직한 감정)",
    Theme.IDIOMS: "🎯 미국식 관용구 (자연스러운 표현·슬랭)",
}


# ============================================================
# TTS URL 해결 — 캐시 우선, miss 시 합성 후 업로드.
# ============================================================
def _resolve_tts_url(
    text: str,
    synthesizer: TtsSynthesizer | None,
    uploader: TtsUploader | None,
) -> str | None:
    """text 의 TTS public URL.  실패 시 None 반환 (호출자가 link 생략).

    경우의 수:
      - uploader=None → TTS 비활성 (PR B 통합 전 호환)
      - cache hit → URL 만 반환 (합성 skip, 비용 0)
      - cache miss + synthesizer → 합성·업로드 후 URL
      - cache miss + synthesizer=None → None (TTS 미사용 모드)
      - 어디서든 예외 → None (graceful degradation)
    """
    if uploader is None:
        return None
    try:
        if uploader.exists(text):
            return uploader.get_url(text)
        if synthesizer is None:
            return None
        audio_bytes = synthesizer(text)
        return uploader.upload_or_get_url(text, audio_bytes)
    except Exception as exc:  # pylint: disable=broad-except
        # graceful degradation — 개별 문장 실패는 메시지 발송 차단하지 않음.
        logger.warning("TTS resolve 실패 (text 일부: %r): %s", text[:40], exc)
        return None


def _tts_link_html(text: str, url: str | None) -> str:
    """TTS URL 이 있으면 \"🔊 <a href=...>듣기</a>\" 반환, 없으면 빈 문자열."""
    if not url:
        return ""
    return f"  🔊 <a href=\"{escape_html(url)}\">듣기</a>"


# ============================================================
# 메시지 조립.
# ============================================================
def render_message(
    data: EnglishPhraseData,
    *,
    synthesizer: TtsSynthesizer | None = None,
    uploader: TtsUploader | None = None,
) -> str:
    """EnglishPhraseData → Telegram HTML 메시지 (parse_mode=HTML 전제).

    모든 영문 문장에 🔊 듣기 링크 시도 — 실패 시 link 생략 (메시지 자체는 발송).
    """
    label = _THEME_LABEL.get(data.theme, data.theme.value)

    # 영문 문장별 TTS URL — 한 번씩만 resolve (효율).
    phrase_url = _resolve_tts_url(data.phrase_en, synthesizer, uploader)
    scenario_urls = [
        _resolve_tts_url(s.english, synthesizer, uploader) for s in data.scenarios
    ]
    response_urls = [
        _resolve_tts_url(r.english, synthesizer, uploader) for r in data.responses
    ]

    lines: list[str] = []

    # Header.
    lines.append(f"📚 <b>오늘의 영어 한 마디</b> — {escape_html(label)}")
    lines.append("")

    # Phrase (메인 영문 + TTS).
    lines.append(
        f"▶ <b>Phrase:</b> \"{escape_html(data.phrase_en)}\""
        + _tts_link_html(data.phrase_en, phrase_url)
    )
    lines.append("")

    # 직역.
    if data.phrase_ko:
        lines.append("🇰🇷 <b>직역</b>")
        lines.append(f"\"{escape_html(data.phrase_ko)}\"")
        lines.append("")

    # 미국 현장에서.
    if data.context:
        lines.append("💡 <b>미국 현장에서</b>")
        lines.append(escape_html(data.context))
        lines.append("")

    # 직접 써먹기.
    if data.scenarios:
        lines.append("🎯 <b>직접 써먹기</b>")
        for idx, sc in enumerate(data.scenarios, start=1):
            lines.append(f"• 시나리오 {idx} — {escape_html(sc.setup)}")
            lines.append(
                f"  \"{escape_html(sc.english)}\""
                + _tts_link_html(sc.english, scenario_urls[idx - 1])
            )
            # 2026-05-14 사용자 피드백 — 영문 아래 한국어 직역 (대답 예시와 동일 포맷).
            if sc.korean:
                lines.append(f"  ({escape_html(sc.korean)})")
        lines.append("")

    # 대답 예시 (신규).
    if data.responses:
        lines.append("💬 <b>대답 예시</b>")
        for idx, rsp in enumerate(data.responses, start=1):
            lines.append(
                f"• \"{escape_html(rsp.english)}\""
                + _tts_link_html(rsp.english, response_urls[idx - 1])
            )
            if rsp.korean:
                lines.append(f"  ({escape_html(rsp.korean)})")
        lines.append("")

    # 발음.
    if data.pronunciation:
        lines.append("🗣️ <b>발음</b>")
        lines.append(escape_html(data.pronunciation))
        lines.append("")

    # 어원.
    if data.etymology:
        lines.append("📌 <b>어원·왜 이렇게?</b>")
        lines.append(escape_html(data.etymology))

    # trailing whitespace 제거.
    return "\n".join(lines).rstrip()


__all__ = [
    "escape_html",
    "render_message",
    "TtsSynthesizer",
    "TtsUploader",
]
