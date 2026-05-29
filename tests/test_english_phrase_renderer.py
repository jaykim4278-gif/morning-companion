# W9 PR — english_phrase_renderer 테스트.
# JSON → HTML 메시지 조립 + TTS 링크 임베드 검증.
from __future__ import annotations

from typing import Any

import pytest

from execution.github_actions.english_phrase import Theme
from execution.github_actions.english_phrase_llm import (
    EnglishPhraseData,
    ResponseExample,
    Scenario,
)
from execution.github_actions.english_phrase_renderer import (
    escape_html,
    render_message,
)


def _sample_data() -> EnglishPhraseData:
    return EnglishPhraseData(
        theme=Theme.SMALL_TALK,
        phrase_en="What have you been up to?",
        phrase_ko="그동안 뭘 하고 지내셨어요?",
        context="이 표현은 오랜만에 만난 사람에게 안부를 묻거나 근황을 물을 때 쓰입니다.",
        scenarios=(
            Scenario(setup="동네 커피숍에서 우연히 만난 이웃에게",
                     english="Hello, Sarah! Long time no see. What have you been up to?",
                     korean="안녕하세요, 사라! 오랜만이에요. 그동안 뭘 하고 지내셨어요?"),
            Scenario(setup="마트 캐셔에게 (오랜만에 만났을 때)",
                     english="Hey, Mike! Good to see you. What have you been up to?",
                     korean="안녕, 마이크! 오랜만에 만나서 반가워. 그동안 뭐 하고 지냈어?"),
        ),
        responses=(
            ResponseExample(english="Not much, just working a lot. How about you?",
                            korean="별일 없어, 일이 많아. 너는 어때?"),
            ResponseExample(english="Oh, just enjoying the summer weather.",
                            korean="그냥 여름 날씨를 즐기고 있어."),
        ),
        pronunciation="와[강]트 해브 유 빈 업 투?",
        etymology="Up to 는 원래 '무엇을 하고 있는지' 를 나타내는 구동사입니다.",
    )


class _StubUploader:
    """TtsUploader Protocol mock — 미리 정의된 URL 패턴 반환."""

    def __init__(self, existing: set[str] = frozenset(), upload_fails: bool = False):
        self.existing = set(existing)
        self.upload_fails = upload_fails
        self.upload_calls: list[tuple[str, int]] = []  # (text, audio_bytes_len)
        self.exists_calls: list[str] = []

    def exists(self, text: str) -> bool:
        self.exists_calls.append(text)
        return text in self.existing

    def upload_or_get_url(self, text: str, audio_bytes: bytes) -> str:
        self.upload_calls.append((text, len(audio_bytes)))
        if self.upload_fails:
            raise RuntimeError("upload failed")
        return f"https://example.com/tts/{hash(text) & 0xFFFFFF:x}.mp3"

    def get_url(self, text: str) -> str:
        return f"https://example.com/tts/{hash(text) & 0xFFFFFF:x}.mp3"


def _fake_synthesizer(text: str) -> bytes:
    """edge-tts mock — 텍스트 길이 기반 더미 bytes."""
    return f"audio:{text}".encode("utf-8")


class TestEscapeHtml:
    def test_escapes_ampersand_first(self) -> None:
        # & 가 가장 먼저 escape — 이후 < > escape 가 &amp; 를 망가뜨리지 않음.
        assert escape_html("a & b") == "a &amp; b"

    def test_escapes_lt_gt(self) -> None:
        assert escape_html("a < b > c") == "a &lt; b &gt; c"

    def test_combined(self) -> None:
        assert escape_html("<a href=\"x\">") == "&lt;a href=\"x\"&gt;"

    def test_other_chars_literal(self) -> None:
        # Telegram HTML mode 는 _, *, ., : 등 escape 불필요.
        assert escape_html("Hello_World!") == "Hello_World!"
        assert escape_html("3.14") == "3.14"


class TestRenderMessageWithoutTts:
    """uploader=None 일 때 음성 링크 없이 렌더 — backward compat."""

    def test_includes_all_sections(self) -> None:
        msg = render_message(_sample_data())
        assert "오늘의 영어 한 마디" in msg
        assert "잡담" in msg  # theme label
        assert "What have you been up to?" in msg
        assert "그동안 뭘 하고 지내셨어요?" in msg
        assert "미국 현장에서" in msg
        assert "직접 써먹기" in msg
        assert "시나리오 1" in msg
        assert "시나리오 2" in msg
        assert "대답 예시" in msg  # 신규 섹션
        assert "발음" in msg
        assert "어원" in msg
        assert "와[강]트" in msg

    def test_no_tts_links_without_uploader(self) -> None:
        msg = render_message(_sample_data())
        # 🔊 듣기 링크 없어야 함.
        assert "🔊" not in msg
        assert "href=" not in msg

    def test_html_bold_tags_used(self) -> None:
        msg = render_message(_sample_data())
        # HTML parse_mode 용 <b> 태그 사용 (MarkdownV2 *...* 아님).
        assert "<b>" in msg
        assert "</b>" in msg
        assert "*Phrase:*" not in msg  # MarkdownV2 잔재 없음


class TestRenderMessageWithTts:
    """synthesizer + uploader 주입 시 음성 링크 임베드."""

    def test_includes_tts_links_for_all_english(self) -> None:
        uploader = _StubUploader()
        msg = render_message(
            _sample_data(), synthesizer=_fake_synthesizer, uploader=uploader,
        )
        # phrase + 2 scenarios + 2 responses = 5 개 영문 → 5 개 🔊 듣기.
        assert msg.count("🔊") == 5
        # 모든 링크가 듣기 텍스트 + href 형식.
        assert msg.count("듣기</a>") == 5

    def test_synthesizer_called_for_each_unique_text(self) -> None:
        # cache miss → synthesizer 호출.
        uploader = _StubUploader(existing=set())
        synth_calls: list[str] = []

        def synth(text: str) -> bytes:
            synth_calls.append(text)
            return b"audio"

        render_message(_sample_data(), synthesizer=synth, uploader=uploader)
        # 모든 5개 영문에 대해 합성 호출.
        assert len(synth_calls) == 5

    def test_cache_hit_skips_synthesis(self) -> None:
        # 모든 텍스트가 이미 storage 에 있음 → 합성 skip.
        all_english = {
            "What have you been up to?",
            "Hello, Sarah! Long time no see. What have you been up to?",
            "Hey, Mike! Good to see you. What have you been up to?",
            "Not much, just working a lot. How about you?",
            "Oh, just enjoying the summer weather.",
        }
        uploader = _StubUploader(existing=all_english)
        synth_calls: list[str] = []
        render_message(
            _sample_data(),
            synthesizer=lambda t: (synth_calls.append(t), b"x")[1],
            uploader=uploader,
        )
        # 합성 0회.
        assert synth_calls == []
        # 메시지에는 링크 5개 (URL 만 조회 — get_url 사용).
        # render 결과는 정상.

    def test_upload_failure_omits_link_but_keeps_text(self) -> None:
        # uploader.upload_or_get_url 가 예외 → 해당 라인 링크만 생략, 텍스트는 보존.
        uploader = _StubUploader(existing=set(), upload_fails=True)
        msg = render_message(
            _sample_data(),
            synthesizer=_fake_synthesizer,
            uploader=uploader,
        )
        # 영문 텍스트 자체는 모두 포함.
        assert "What have you been up to?" in msg
        assert "Hello, Sarah" in msg
        assert "Not much" in msg
        # 모든 링크 실패 → 🔊 없음.
        assert "🔊" not in msg

    def test_synthesizer_exception_graceful(self) -> None:
        # synth 가 예외 던져도 메시지 자체는 발송됨.
        uploader = _StubUploader()

        def bad_synth(text: str) -> bytes:
            raise RuntimeError("edge-tts down")

        msg = render_message(
            _sample_data(), synthesizer=bad_synth, uploader=uploader,
        )
        # 텍스트 보존 + 링크 없음.
        assert "What have you been up to?" in msg
        assert "🔊" not in msg


class TestScenarioKoreanTranslation:
    """2026-05-14 사용자 피드백 — 시나리오 영문 아래 한국어 직역 표시."""

    def test_scenario_korean_rendered_in_parentheses(self) -> None:
        msg = render_message(_sample_data())
        # 시나리오 한국어가 괄호로 표시.
        assert "(안녕하세요, 사라! 오랜만이에요. 그동안 뭘 하고 지내셨어요?)" in msg
        assert "(안녕, 마이크! 오랜만에 만나서 반가워. 그동안 뭐 하고 지냈어?)" in msg

    def test_empty_korean_omits_line(self) -> None:
        # 빈 korean (구 데이터·LLM 누락) 은 라인 자체 생략.
        data = EnglishPhraseData(
            theme=Theme.SMALL_TALK,
            phrase_en="Hi",
            phrase_ko="안녕",
            context="",
            scenarios=(
                Scenario(setup="상황 1", english="Hi there!", korean=""),
            ),
            responses=(),
            pronunciation="",
            etymology="",
        )
        msg = render_message(data)
        # 영문은 있지만 빈 korean 라인 없음.
        assert "Hi there!" in msg
        # 빈 괄호 없음.
        assert "()" not in msg


class TestHtmlSafetyInRendering:
    """사용자 컨텐츠 (LLM 출력) 가 HTML escape 되는지."""

    def test_lt_gt_in_phrase_escaped(self) -> None:
        # LLM 이 <script> 같은 텍스트 반환해도 entity escape.
        data = EnglishPhraseData(
            theme=Theme.SMALL_TALK,
            phrase_en="<script>alert(1)</script>",
            phrase_ko="<>",
            context="A & B < C > D",
            scenarios=(),
            responses=(),
            pronunciation="",
            etymology="",
        )
        msg = render_message(data)
        assert "<script>" not in msg
        assert "&lt;script&gt;" in msg
        assert "A &amp; B &lt; C &gt; D" in msg
