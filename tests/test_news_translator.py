# W8a PR #3 — translator 9중 retry + voucher 폴백 + 번역 실패 0 contract.
# docs/design/w8a-news-briefing-design.md §4 · §10.1.
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pytest

from execution.ai_client.fallback_chain import (
    ChainConfig,
    FallbackError,
    RetryableError,
)
from execution.github_actions.news.rss_fetcher import RssItem
from execution.github_actions.news.sources import Category
from execution.github_actions.news.translator import (
    DEFAULT_CHAIN_CONFIG,
    _deterministic_summary_placeholder,
    _deterministic_translate,
    contains_foreign_cjk,
    translate_category,
    translate_one,
    translate_summary_one,
    translate_with_voucher,
)


@dataclass
class FakeProvider:
    name: str
    responses: list[Any] = field(default_factory=list)
    call_count: int = 0
    seen_prompts: list[str] = field(default_factory=list)
    seen_payloads: list[dict] = field(default_factory=list)

    def generate(self, payload: dict, prompt: str) -> str:
        self.seen_payloads.append(payload)
        self.seen_prompts.append(prompt)
        idx = self.call_count
        self.call_count += 1
        if idx >= len(self.responses):
            return f"{self.name}-default"
        item = self.responses[idx]
        if isinstance(item, BaseException):
            raise item
        return item


def _make_item(url: str = "https://r.com/1", title: str = "Fed signals rate cut") -> RssItem:
    return RssItem(
        source="reuters",
        url=url,
        url_hash="abc",
        title_en=title,
        pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
        category=Category.TOP,
    )


# zero-sleep config — 테스트 즉시 실행.
FAST_CONFIG = ChainConfig(
    max_retries_per_provider=2,
    base_backoff_seconds=0.0,
    max_backoff_seconds=0.0,
)

# 2026-05-13 PR — translate_category 의 inter-item sleep 도 테스트에서 0 으로.
_NO_INTERVAL = {"item_interval_sec": 0.0}


# ============================================================
class TestTranslateOne:
    def test_gemini_success_populates_title_ko(self) -> None:
        g = FakeProvider("gemini", ["Fed, 금리 인하 시그널"])
        item = _make_item()
        result = translate_one(item, [g], config=FAST_CONFIG)
        assert result is not None
        assert result.title_ko == "Fed, 금리 인하 시그널"

    def test_provider_strips_whitespace(self) -> None:
        g = FakeProvider("gemini", ["  Fed, 금리 인하 시그널  \n"])
        result = translate_one(_make_item(), [g], config=FAST_CONFIG)
        assert result.title_ko == "Fed, 금리 인하 시그널"

    def test_empty_response_treated_as_failure(self) -> None:
        # LLM 이 공백만 반환 = 실패 취급.
        g = FakeProvider("gemini", [""])
        result = translate_one(_make_item(), [g], config=FAST_CONFIG)
        assert result is None

    def test_all_providers_fail_returns_none(self) -> None:
        g = FakeProvider("gemini", [FallbackError("429")])
        q = FakeProvider("groq", [FallbackError("429")])
        o = FakeProvider("openrouter", [FallbackError("429")])
        result = translate_one(_make_item(), [g, q, o], config=FAST_CONFIG)
        assert result is None

    def test_gemini_retry_count_matches_9_attempts(self) -> None:
        # 9 attempt contract: 3 providers × (max_retries_per_provider+1) = 9.
        # 전부 RetryableError → 각 provider 3회 시도 후 다음.
        g = FakeProvider("gemini", [RetryableError("x"), RetryableError("x"), RetryableError("x")])
        q = FakeProvider("groq", [RetryableError("x"), RetryableError("x"), RetryableError("x")])
        o = FakeProvider("openrouter",
                          [RetryableError("x"), RetryableError("x"), RetryableError("x")])
        result = translate_one(_make_item(), [g, q, o], config=FAST_CONFIG)
        assert result is None
        assert g.call_count == 3
        assert q.call_count == 3
        assert o.call_count == 3
        assert g.call_count + q.call_count + o.call_count == 9

    def test_payload_contains_no_pii(self) -> None:
        # 뉴스 제목 번역 payload = {"news": {title, source}} — user_id/drug 없음.
        g = FakeProvider("gemini", ["번역"])
        translate_one(_make_item(), [g], config=FAST_CONFIG)
        sent = g.seen_payloads[0]
        assert set(sent.keys()) == {"news"}
        assert set(sent["news"].keys()) == {"title", "source"}
        assert "user_id" not in sent


# ============================================================
class TestTranslateWithVoucher:
    def test_primary_success_no_voucher_call(self) -> None:
        g = FakeProvider("gemini", ["primary 번역"])
        primary = _make_item("https://r.com/1", "primary")
        v1 = _make_item("https://r.com/2", "voucher1")
        result = translate_with_voucher(primary, [v1], [g], config=FAST_CONFIG)
        assert result is not None
        assert result.url == "https://r.com/1"
        assert g.call_count == 1  # primary 만

    def test_voucher_used_when_primary_exhausts(self) -> None:
        # primary 모두 실패 (3 retry), voucher1 은 성공 → 4번째 호출에서 성공.
        g = FakeProvider("gemini", [
            # primary 3 attempts
            RetryableError("1"), RetryableError("2"), RetryableError("3"),
            # voucher1 첫 시도 성공
            "voucher1 번역",
        ])
        # 단일 provider 만 제공 — groq/openrouter 없으므로 primary 는 3 attempt 로 AllProvidersFailed.
        primary = _make_item("https://r.com/1", "primary")
        v1 = _make_item("https://r.com/2", "voucher1")
        result = translate_with_voucher(primary, [v1], [g], config=FAST_CONFIG)
        assert result is not None
        assert result.url == "https://r.com/2"
        assert result.title_ko == "voucher1 번역"

    def test_all_candidates_fail_returns_none(self) -> None:
        # 모든 아이템 × 모든 provider 실패.
        g = FakeProvider("gemini", [FallbackError("a")] * 10)
        primary = _make_item("https://r.com/1")
        v1 = _make_item("https://r.com/2")
        result = translate_with_voucher(primary, [v1], [g], config=FAST_CONFIG)
        assert result is None


# ============================================================
class TestTranslateCategory:
    def test_returns_target_count_when_all_succeed(self) -> None:
        g = FakeProvider("gemini", ["t1", "t2"])
        cands = [_make_item(f"https://r.com/{i}", f"c{i}") for i in range(3)]
        result = translate_category(
            cands, target_count=2, providers=[g], config=FAST_CONFIG, **_NO_INTERVAL,
        )
        assert len(result) == 2
        assert all(it.title_ko is not None for it in result)

    def test_voucher_fills_when_primary_fails(self) -> None:
        # 2026-05-04: drop 폐기 — primary 실패 시 결정론 폴백으로 한국어 출력 → 첫 후보가 0번 url 로 통과.
        g = FakeProvider("gemini", [
            RetryableError("a"), RetryableError("b"), RetryableError("c"),
            # 1단계 9중 retry 실패 → 2단계 stricter retry 도 실패하도록 fail 채움.
            FallbackError("strict-1"), FallbackError("strict-2"), FallbackError("strict-3"),
            "voucher 번역",
        ])
        cands = [_make_item(f"https://r.com/{i}", f"c{i}") for i in range(2)]
        result = translate_category(
            cands, target_count=1, providers=[g], config=FAST_CONFIG, **_NO_INTERVAL,
        )
        assert len(result) == 1
        # 새 정책: 첫 candidate 도 결정론 폴백으로 통과.
        assert result[0].url == "https://r.com/0"
        assert result[0].title_ko is not None

    def test_zero_failure_contract_nonnull_title_ko(self) -> None:
        # §4.4 핵심 contract: 반환된 모든 아이템은 title_ko is not None.
        g = FakeProvider("gemini", ["a번역", "b번역"])
        cands = [_make_item(f"https://r.com/{i}", f"c{i}") for i in range(2)]
        result = translate_category(
            cands, target_count=2, providers=[g], config=FAST_CONFIG, **_NO_INTERVAL,
        )
        assert all(it.title_ko is not None and it.title_ko.strip() for it in result)

    def test_empty_candidates_returns_empty(self) -> None:
        assert translate_category(
            [], target_count=2, providers=[FakeProvider("g")], **_NO_INTERVAL,
        ) == []

    def test_deterministic_fallback_when_all_llm_fails(self) -> None:
        # 2026-05-04 정책 재전환: 번역 의무화. LLM 모두 실패해도 사전 기반 결정론 폴백.
        # 사용자 강력 재요구 — "기사 번역실패시 제외가 아니라 무조건 번역하게."
        g = FakeProvider("gemini", [FallbackError("quota 429")] * 30)
        cands = [
            _make_item("https://r.com/0", "Trump signals new tariffs on tech imports"),
            _make_item("https://r.com/1", "Fed announces rate cut"),
        ]
        result = translate_category(
            cands, target_count=2, providers=[g], config=FAST_CONFIG, **_NO_INTERVAL,
        )
        # drop 없음 — 모두 한국어 출력.
        assert len(result) == 2
        for it in result:
            assert it.title_ko is not None
            # 한글이 최소 한 글자 이상 포함되어야 함 (사전 매핑 적중).
            assert any("가" <= c <= "힯" for c in it.title_ko), f"한글 없음: {it.title_ko!r}"
        # 첫 항목은 트럼프·관세·테크 매핑.
        assert "트럼프" in result[0].title_ko
        # 둘째 항목은 연준·금리·인하 매핑.
        assert "연준" in result[1].title_ko or "인하" in result[1].title_ko

    def test_cursor_advances_skip_dead_provider(self) -> None:
        # 2026-05-13 PR: sticky cursor — 1번째 항목 Gemini 죽으면 2번째 항목은 Gemini 건너뛰고 Groq 직행.
        # 검증: Gemini call_count == 1 (1번째 항목만), Groq 가 두 항목 모두 처리.
        # 항목은 summary 없음 (_make_item) → title 호출만 발생.
        g = FakeProvider("gemini", [FallbackError("quota 429")])  # 1번 호출 후 죽음
        q = FakeProvider("groq", ["1번째 한국어", "2번째 한국어"])
        cands = [
            _make_item("https://r.com/0", "first"),
            _make_item("https://r.com/1", "second"),
        ]
        result = translate_category(
            cands, target_count=2, providers=[g, q], config=FAST_CONFIG, **_NO_INTERVAL,
        )
        assert len(result) == 2
        # Gemini 는 1번째 항목 1회만 호출 (이후 cursor 가 Groq 로 advance — 죽은 Gemini 재시도 X).
        assert g.call_count == 1, f"Gemini 가 {g.call_count}회 호출됨 — cursor 미작동 (1회여야 함)"
        # Groq 는 두 항목 모두 처리 (2회 호출).
        assert q.call_count == 2
        # 두 항목 다 Groq 로 번역 성공.
        assert result[0].title_ko == "1번째 한국어"
        assert result[1].title_ko == "2번째 한국어"
        assert all(not it.translation_failed for it in result)

    def test_inter_item_sleep_called_between_items(self) -> None:
        # 2026-05-13 PR: RPM 한도 존중 — N 아이템 발송 시 N-1 회 sleep.
        # 항목은 summary 없음 — title 호출만 (3 응답으로 충분).
        g = FakeProvider("gemini", ["a", "b", "c"])
        cands = [_make_item(f"https://r.com/{i}", f"c{i}") for i in range(3)]
        sleeps: list[float] = []
        translate_category(
            cands, target_count=3, providers=[g], config=FAST_CONFIG,
            item_interval_sec=2.5,
            sleep=lambda s: sleeps.append(s),
        )
        # 3 아이템 → 2회 sleep (1↔2, 2↔3 사이).
        assert sleeps == [2.5, 2.5], f"예상 [2.5, 2.5], 실제 {sleeps}"

    def test_all_dead_subsequent_items_skip_llm(self) -> None:
        # 2026-05-13 PR: cursor sentinel — provider 다 죽었으면 이후 항목은 LLM call 0.
        # FallbackError 1회 후 추가 호출이 발생하면 안 됨 (cursor=sentinel).
        g = FakeProvider("gemini", [FallbackError("dead")])
        cands = [_make_item(f"https://r.com/{i}", f"c{i}") for i in range(3)]
        result = translate_category(
            cands, target_count=3, providers=[g], config=FAST_CONFIG, **_NO_INTERVAL,
        )
        assert len(result) == 3
        # Gemini 호출 1회 (첫 항목만) — 이후 항목은 deterministic 직행.
        assert g.call_count == 1
        # 모두 deterministic — translation_failed=True.
        assert all(it.translation_failed for it in result)


# ============================================================
# ============================================================
# hotfix 2026-04-21 — summary best-effort + title English fallback.
# ============================================================
class TestTranslateSummary:
    def _item(self, summary_en: str = "Fed hints at a rate cut at the May FOMC.") -> RssItem:
        return RssItem(
            source="reuters",
            url="https://r.com/1",
            url_hash="abc",
            title_en="Fed signals rate cut",
            pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc),
            category=Category.TOP,
            summary_en=summary_en,
        )

    def test_success_populates_summary_ko(self) -> None:
        g = FakeProvider("gemini", ["Fed, 5월 FOMC 금리 인하 시사."])
        item = self._item()
        translate_summary_one(item, [g], config=FAST_CONFIG)
        assert item.summary_ko == "Fed, 5월 FOMC 금리 인하 시사."

    def test_failure_leaves_summary_ko_none(self) -> None:
        # Gemini 할당량 등 실패 시 조용히 None 유지 (메시지 발송은 영문 summary 로 진행).
        g = FakeProvider("gemini", [FallbackError("quota")] * 20)
        item = self._item()
        translate_summary_one(item, [g], config=FAST_CONFIG)
        assert item.summary_ko is None

    def test_empty_summary_skips_api_call(self) -> None:
        g = FakeProvider("gemini", ["should-not-be-used"])
        item = self._item(summary_en="")
        translate_summary_one(item, [g], config=FAST_CONFIG)
        assert g.call_count == 0
        assert item.summary_ko is None


class TestTranslateCategoryAlsoTranslatesSummary:
    def test_summary_translated_on_title_success(self) -> None:
        # 번역 성공 시 summary 번역도 함께 시도되어야 함.
        g = FakeProvider("gemini", ["한국어 제목", "한국어 요약"])
        item = RssItem(
            source="reuters", url="https://r.com/1", url_hash="h",
            title_en="English Title", summary_en="English summary sentence.",
            pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc), category=Category.TOP,
        )
        result = translate_category(
            [item], target_count=1, providers=[g], config=FAST_CONFIG, **_NO_INTERVAL,
        )
        assert len(result) == 1
        assert result[0].title_ko == "한국어 제목"
        assert result[0].summary_ko == "한국어 요약"


class TestCjkDetection:
    def test_pure_hangul_passes(self) -> None:
        assert not contains_foreign_cjk("연준 금리 인하")
        assert not contains_foreign_cjk("Apple 차기 CEO")
        assert not contains_foreign_cjk("")

    def test_chinese_ideograph_detected(self) -> None:
        assert contains_foreign_cjk("주요 장애物은")
        assert contains_foreign_cjk("Justin Ballard 氏")
        assert contains_foreign_cjk("휴戰 연장")

    def test_japanese_kana_detected(self) -> None:
        assert contains_foreign_cjk("今日のニュース")
        assert contains_foreign_cjk("テスト")


class TestCjkRetryInTranslateOne:
    def test_cjk_response_triggers_retry_and_accepts_clean(self) -> None:
        # 첫 응답 한자 포함 → 두번째 응답 순수 한글 → 수용.
        g = FakeProvider("gemini", ["Justin Ballard 氏가 전해준다", "Justin Ballard 가 전해준다"])
        item = _make_item()
        out = translate_one(item, [g], config=FAST_CONFIG)
        assert out is not None
        assert "氏" not in out.title_ko
        assert "Justin Ballard" in out.title_ko

    def test_cjk_twice_returns_none_for_english_fallback(self) -> None:
        # 두 번 다 한자 → None (호출자는 영문 폴백).
        g = FakeProvider("gemini", ["주요 장애物은", "주요 障礙은"])
        item = _make_item()
        out = translate_one(item, [g], config=FAST_CONFIG)
        assert out is None


class TestCjkRetryInSummary:
    def test_cjk_summary_retry_success(self) -> None:
        g = FakeProvider("gemini", ["今月의 실적", "이번 달의 실적입니다."])
        item = RssItem(
            source="cnbc", url="https://x.com", url_hash="h",
            title_en="x", summary_en="This month's earnings.",
            pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc), category=Category.MARKETS,
        )
        translate_summary_one(item, [g], config=FAST_CONFIG)
        assert item.summary_ko is not None
        assert "今月" not in item.summary_ko

    def test_cjk_summary_twice_leaves_none(self) -> None:
        # Summary 는 영문 원문 폴백 허용 — summary_ko None 으로 유지.
        g = FakeProvider("gemini", ["今月 실적", "今年 실적"])
        item = RssItem(
            source="cnbc", url="https://x.com", url_hash="h",
            title_en="x", summary_en="earnings report",
            pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc), category=Category.MARKETS,
        )
        translate_summary_one(item, [g], config=FAST_CONFIG)
        assert item.summary_ko is None


class TestDefaults:
    def test_default_chain_config_gives_9_attempts_envelope(self) -> None:
        # (max_retries + 1) × 3 providers = 9 attempts.
        # DEFAULT_CHAIN_CONFIG.max_retries_per_provider=2 → 3 attempts/provider.
        assert DEFAULT_CHAIN_CONFIG.max_retries_per_provider == 2


# ============================================================
# PR #95 — 결정론 사전 기반 번역 폴백 (LLM 0개 시).
# ============================================================
class TestDeterministicTranslate:
    def test_maps_known_keywords_to_korean(self) -> None:
        out = _deterministic_translate("Trump signals new tariffs on tech imports")
        assert "트럼프" in out
        assert "관세" in out

    def test_long_phrase_priority(self) -> None:
        # "federal reserve" 가 "fed" 보다 우선 매핑.
        out = _deterministic_translate("Federal Reserve cuts rates")
        assert "연방준비제도" in out
        assert "인하" in out

    def test_no_mapping_returns_korean_placeholder(self) -> None:
        # 매핑 안 되는 영문은 "「원문」 자동 번역 실패" 라벨.
        out = _deterministic_translate("Gobbledygook xyzzy quux")
        assert "자동 번역 실패" in out

    def test_empty_input_returns_placeholder(self) -> None:
        assert "(제목" in _deterministic_translate("")

    def test_word_boundary_prevents_false_match(self) -> None:
        # "cat" 이 매핑됐다고 가정해도 "category" 안의 "cat" 은 매칭 X.
        # 현재 사전엔 "rate" 가 있어 "rates" 만 매칭, "ratepayer" 은 매칭 X 검증.
        out = _deterministic_translate("Ratepayer association meeting")
        # rate 단어 경계 미충족 → 매핑 0개 → placeholder.
        assert "자동 번역 실패" in out


class TestDeterministicSummaryPlaceholder:
    def test_returns_korean_label_with_length(self) -> None:
        from execution.github_actions.news.rss_fetcher import RssItem
        item = RssItem(
            source="reuters", url="https://r.com", url_hash="h",
            title_en="x", summary_en="A" * 120,
            pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc), category=Category.TOP,
        )
        out = _deterministic_summary_placeholder(item)
        assert "120자" in out
        assert "한국어" not in out  # 안내 문구 자체가 한국어
        assert "자동 번역 일시 불가" in out

    def test_empty_summary_label(self) -> None:
        from execution.github_actions.news.rss_fetcher import RssItem
        item = RssItem(
            source="reuters", url="https://r.com", url_hash="h",
            title_en="x", summary_en="",
            pub_date=datetime(2026, 4, 20, tzinfo=timezone.utc), category=Category.TOP,
        )
        assert _deterministic_summary_placeholder(item) == "(요약 없음)"
