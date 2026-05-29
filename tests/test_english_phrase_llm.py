# W9 — 영어회화 LLM Generator + JSON 파싱 + 결정론 폴백 테스트.
# 2026-05-14 PR — 출력 형식 plain text → JSON 전환에 따라 테스트 재작성.
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest

from execution.ai_client.fallback_chain import (
    AllProvidersFailed,
    ChainConfig,
    FallbackError,
)
from execution.github_actions.english_phrase import (
    EnglishPhraseInputs,
    Theme,
)
from execution.github_actions.english_phrase_llm import (
    ENGLISH_PHRASE_SYSTEM,
    EnglishPhraseData,
    LlmEnglishPhraseGenerator,
    ResponseExample,
    Scenario,
    _FALLBACK_BY_THEME,
    dict_to_data,
    parse_phrase_data,
)


@dataclass
class FakeProvider:
    name: str
    responses: list[Any] = field(default_factory=list)
    seen_prompts: list[str] = field(default_factory=list)
    seen_payloads: list[dict] = field(default_factory=list)
    call_count: int = 0

    def generate(self, payload: dict, prompt: str) -> str:
        self.seen_payloads.append(payload)
        self.seen_prompts.append(prompt)
        idx = self.call_count
        self.call_count += 1
        if idx >= len(self.responses):
            return "default"
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        return item


FAST = ChainConfig(max_retries_per_provider=0, base_backoff_seconds=0.0)


def _valid_json_payload(phrase_en: str = "What have you been up to?") -> str:
    """LLM 이 반환할 만한 유효 JSON 문자열."""
    return json.dumps({
        "phrase_en": phrase_en,
        "phrase_ko": "그동안 뭘 하고 지내셨어요?",
        "context": "오랜만에 만난 사람에게...",
        "scenarios": [
            {"setup": "커피숍에서", "english": "Hello, what have you been up to?",
             "korean": "안녕하세요, 그동안 뭘 하셨어요?"},
            {"setup": "마트에서", "english": "Hey, what have you been up to?",
             "korean": "안녕, 뭐 하고 지냈어?"},
        ],
        "responses": [
            {"english": "Not much.", "korean": "별일 없어."},
            {"english": "Just working.", "korean": "그냥 일해."},
        ],
        "pronunciation": "와[강]트 해브 유 빈 업 투",
        "etymology": "Up to 는 ...",
    }, ensure_ascii=False)


# ============================================================
class TestParsePhraseData:
    def test_parses_valid_json(self) -> None:
        raw = _valid_json_payload("Test Phrase")
        data = parse_phrase_data(raw, Theme.SMALL_TALK)
        assert data is not None
        assert data.phrase_en == "Test Phrase"
        assert data.theme == Theme.SMALL_TALK
        assert len(data.scenarios) == 2
        assert len(data.responses) == 2
        assert data.scenarios[0].english.startswith("Hello")
        # 2026-05-14 사용자 피드백 — 시나리오에 korean 직역 포함.
        assert data.scenarios[0].korean.startswith("안녕")
        assert data.responses[0].korean == "별일 없어."

    def test_scenario_korean_missing_tolerated(self) -> None:
        # 구 데이터 / LLM 누락 케이스 — korean 필드 없으면 빈 문자열로.
        raw = json.dumps({
            "phrase_en": "Hi",
            "phrase_ko": "안녕",
            "scenarios": [
                {"setup": "장면", "english": "Hi there!"},  # korean 누락
            ],
            "responses": [],
        })
        data = parse_phrase_data(raw, Theme.SMALL_TALK)
        assert data is not None
        assert data.scenarios[0].korean == ""

    def test_extracts_json_from_code_fence(self) -> None:
        # LLM 이 ```json ... ``` 으로 감싼 경우 호환.
        raw = "여기 JSON 입니다:\n```json\n" + _valid_json_payload() + "\n```\n끝."
        data = parse_phrase_data(raw, Theme.IDIOMS)
        assert data is not None
        assert data.phrase_en == "What have you been up to?"

    def test_extracts_first_object_greedy(self) -> None:
        # 앞뒤 텍스트 + JSON 1개.  greedy { ... } 매칭.
        raw = "Sure!\n" + _valid_json_payload() + "\nLet me know if you need more."
        data = parse_phrase_data(raw, Theme.REACTIONS)
        assert data is not None
        assert data.theme == Theme.REACTIONS

    def test_invalid_json_returns_none(self) -> None:
        assert parse_phrase_data("not json", Theme.SMALL_TALK) is None
        assert parse_phrase_data("", Theme.SMALL_TALK) is None

    def test_missing_required_field_returns_none(self) -> None:
        # phrase_en 누락 → None.
        raw = json.dumps({"phrase_ko": "한국어만"})
        assert parse_phrase_data(raw, Theme.SMALL_TALK) is None

    def test_empty_optional_fields_tolerated(self) -> None:
        # 선택 필드 (context·pronunciation 등) 빈 값 허용.
        raw = json.dumps({
            "phrase_en": "Hello",
            "phrase_ko": "안녕",
            "context": "",
            "scenarios": [],
            "responses": [],
            "pronunciation": "",
            "etymology": "",
        })
        data = parse_phrase_data(raw, Theme.SMALL_TALK)
        assert data is not None
        assert data.scenarios == ()
        assert data.responses == ()


# ============================================================
class TestDictToData:
    def test_round_trip(self) -> None:
        data = EnglishPhraseData(
            theme=Theme.SMALL_TALK,
            phrase_en="Hi",
            phrase_ko="안녕",
            context="컨텍스트",
            scenarios=(Scenario("상황 1", "Hi there!"),),
            responses=(ResponseExample("Hi back", "안녕 다시"),),
            pronunciation="하이",
            etymology="OE greeting",
        )
        d = {
            "theme": data.theme.value,
            "phrase_en": data.phrase_en,
            "phrase_ko": data.phrase_ko,
            "context": data.context,
            "scenarios": [{"setup": s.setup, "english": s.english} for s in data.scenarios],
            "responses": [{"english": r.english, "korean": r.korean} for r in data.responses],
            "pronunciation": data.pronunciation,
            "etymology": data.etymology,
        }
        restored = dict_to_data(d)
        assert restored == data


# ============================================================
class TestLlmGeneratorSuccess:
    def test_returns_phrase_with_html_body(self) -> None:
        provider = FakeProvider("gemini", [_valid_json_payload("Right this way")])
        gen = LlmEnglishPhraseGenerator()
        inp = EnglishPhraseInputs(
            user_id="u", as_of=date(2026, 5, 3), theme=Theme.SMALL_TALK,
        )
        result = gen.generate(inp, [provider], config=FAST)
        assert result.theme == Theme.SMALL_TALK
        assert result.phrase_en == "Right this way"
        # body_md 는 HTML 렌더 결과여야 함.
        assert "<b>" in result.body_md
        assert "Right this way" in result.body_md

    def test_recent_phrases_included_in_prompt(self) -> None:
        provider = FakeProvider("gemini", [_valid_json_payload()])
        gen = LlmEnglishPhraseGenerator()
        inp = EnglishPhraseInputs(
            user_id="u", as_of=date(2026, 5, 3), theme=Theme.SMALL_TALK,
            recent_phrases=("Sorry to keep you waiting", "Right this way"),
        )
        gen.generate(inp, [provider], config=FAST)
        prompt = provider.seen_prompts[0]
        assert "Sorry to keep you waiting" in prompt
        assert "Right this way" in prompt
        assert "첫 발송" not in prompt

    def test_first_call_recent_empty_message(self) -> None:
        provider = FakeProvider("gemini", [_valid_json_payload()])
        gen = LlmEnglishPhraseGenerator()
        inp = EnglishPhraseInputs(
            user_id="u", as_of=date(2026, 5, 3), theme=Theme.REACTIONS,
        )
        gen.generate(inp, [provider], config=FAST)
        assert "첫 발송" in provider.seen_prompts[0]

    def test_theme_label_in_prompt(self) -> None:
        provider = FakeProvider("gemini", [_valid_json_payload()])
        gen = LlmEnglishPhraseGenerator()
        for theme in Theme:
            provider.seen_prompts = []
            provider.responses = [_valid_json_payload()] * 4
            provider.call_count = 0
            inp = EnglishPhraseInputs(user_id="u", as_of=date(2026, 5, 3), theme=theme)
            gen.generate(inp, [provider], config=FAST)
            label_keyword = {
                Theme.SMALL_TALK: "잡담",
                Theme.SOFT_SKILLS: "부탁",
                Theme.REACTIONS: "감정",
                Theme.IDIOMS: "관용구",
            }[theme]
            assert label_keyword in provider.seen_prompts[0]

    def test_payload_no_pii(self) -> None:
        provider = FakeProvider("gemini", [_valid_json_payload()])
        gen = LlmEnglishPhraseGenerator()
        gen.generate(
            EnglishPhraseInputs(user_id="u", as_of=date(2026, 5, 3), theme=Theme.SMALL_TALK),
            [provider], config=FAST,
        )
        payload = provider.seen_payloads[0]
        assert set(payload.keys()) == {"english_phrase"}
        assert "user_id" not in payload
        assert set(payload["english_phrase"].keys()) == {"theme"}


# ============================================================
class TestLlmFallback:
    def test_all_providers_failed_uses_deterministic_template(self) -> None:
        gen = LlmEnglishPhraseGenerator()
        provider = FakeProvider(
            "gemini", [FallbackError("quota") for _ in range(10)],
        )
        inp = EnglishPhraseInputs(
            user_id="u", as_of=date(2026, 5, 3), theme=Theme.SMALL_TALK,
        )
        result = gen.generate(inp, [provider], config=FAST)
        # 폴백 phrase_en 일치.
        assert result.phrase_en == _FALLBACK_BY_THEME[Theme.SMALL_TALK].phrase_en
        assert result.theme == Theme.SMALL_TALK
        # HTML 렌더 결과 포함.
        assert "<b>" in result.body_md

    def test_invalid_json_uses_deterministic_fallback(self) -> None:
        # LLM 이 응답은 했지만 JSON 아니면 결정론 폴백.
        gen = LlmEnglishPhraseGenerator()
        provider = FakeProvider("gemini", ["Hello! This is not JSON."])
        inp = EnglishPhraseInputs(
            user_id="u", as_of=date(2026, 5, 3), theme=Theme.IDIOMS,
        )
        result = gen.generate(inp, [provider], config=FAST)
        # 폴백 사용 확인.
        assert result.phrase_en == _FALLBACK_BY_THEME[Theme.IDIOMS].phrase_en

    def test_fallback_data_for_each_theme(self) -> None:
        # 4 테마 모두 폴백 데이터 정의되어 있음.
        for theme in Theme:
            assert theme in _FALLBACK_BY_THEME
            fb = _FALLBACK_BY_THEME[theme]
            assert fb.theme == theme
            assert len(fb.phrase_en) > 0
            assert len(fb.context) > 20  # 충분한 설명
            assert len(fb.scenarios) >= 1
            assert len(fb.responses) >= 1  # 응답 예시 포함 (2026-05-14 신규)


# ============================================================
class TestSystemPromptInvariant:
    def test_prompt_is_generic_and_has_json_format(self) -> None:
        # 시스템 프롬프트는 generic 학습자용이며 JSON 출력 형식이 강제되어야 한다.
        sys = ENGLISH_PHRASE_SYSTEM
        # 학습자 프로필은 template 자리표시자로 주입 — hardcoded 가 아니어야 한다.
        assert "{learner_profile}" in sys
        # 구체 나이 (두 자리 숫자 + '세') 가 prompt 본문에 박혀있지 않아야 한다.
        import re
        assert re.search(r"\d{2}\s*세", sys) is None, "구체 나이 hardcoded"
        # JSON 출력 형식 강제 키워드.
        assert "JSON" in sys
        assert "phrase_en" in sys
        assert "phrase_ko" in sys
        assert "scenarios" in sys
        assert "responses" in sys
        # template placeholder 들이 존재 (format() 호출 호환).
        assert "{recent_phrases}" in sys
