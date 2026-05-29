# W9 PR #91 — 영어회화 한 마디 entrypoint 단위 테스트.
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from execution.ai_client.fallback_chain import AllProvidersFailed, ChainConfig
from execution.github_actions.english_phrase import (
    EnglishPhrase,
    EnglishPhraseInputs,
    Theme,
    hash_phrase,
    run_english_phrase,
    select_theme_for_date,
)


# ============================================================
# Fakes
# ============================================================
@dataclass
class FakeStore:
    recent: list[str] = field(default_factory=list)
    recorded: list[tuple[str, EnglishPhrase, datetime]] = field(default_factory=list)

    def fetch_recent_phrases(self, user_id: str, days: int) -> list[str]:
        return list(self.recent)

    def record_sent(self, user_id: str, phrase: EnglishPhrase, sent_at: datetime) -> None:
        self.recorded.append((user_id, phrase, sent_at))


@dataclass
class FakeTelegram:
    sent: list[tuple[int, str]] = field(default_factory=list)
    fail: bool = False

    def send(self, chat_id: int, text: str) -> None:
        if self.fail:
            raise RuntimeError("network")
        self.sent.append((chat_id, text))


@dataclass
class FakeGenerator:
    next_phrase: EnglishPhrase | None = None
    raise_all_failed: bool = False
    seen_inputs: list[EnglishPhraseInputs] = field(default_factory=list)

    def generate(self, inp, providers, *, config=None) -> EnglishPhrase:
        self.seen_inputs.append(inp)
        if self.raise_all_failed:
            raise AllProvidersFailed("3 providers down")
        if self.next_phrase is None:
            return EnglishPhrase(
                theme=inp.theme,
                phrase_en="Sorry to keep you waiting",
                body_md="🇺🇸 영어 한 마디\n\n*Sorry to keep you waiting*\n\n🇰🇷 ...",
            )
        return self.next_phrase


# ============================================================
class TestThemeRotation:
    def test_each_iso_week_picks_one_theme(self) -> None:
        # 2026-W17 = 17 % 4 = 1 → SUPPLIER
        # 2026-W18 = 18 % 4 = 2 → DAILY
        # 결정론적이라야 같은 주 내내 같은 테마.
        d_w17 = date(2026, 4, 27)   # 월요일 (W18 시작 — ISO calendar)
        d_w17_sun = date(2026, 5, 3)  # 일요일 (W18 마지막)
        # 같은 주 (W18) 내에서 같은 테마.
        assert select_theme_for_date(d_w17) == select_theme_for_date(d_w17_sun)
        # 다음 주는 다음 테마.
        d_w19 = date(2026, 5, 4)   # W19
        assert select_theme_for_date(d_w17) != select_theme_for_date(d_w19)


class TestHashPhrase:
    def test_same_phrase_same_hash(self) -> None:
        assert hash_phrase("Sorry to keep you waiting") == hash_phrase("Sorry to keep you waiting")

    def test_case_and_whitespace_normalized(self) -> None:
        # 동일 문장의 대소문자·앞뒤 공백 차이는 같은 hash.
        assert hash_phrase("SORRY to keep you waiting  ") == hash_phrase("sorry to keep you waiting")

    def test_different_phrases_different_hash(self) -> None:
        assert hash_phrase("a") != hash_phrase("b")


class TestRunEnglishPhrase:
    def _inputs(self, **kw) -> EnglishPhraseInputs:
        defaults = dict(
            user_id="user-1", as_of=date(2026, 5, 3),
            theme=Theme.SMALL_TALK, recent_phrases=(),
            chat_id=1234,
        )
        defaults.update(kw)
        return EnglishPhraseInputs(**defaults)

    def test_no_generator_returns_skip(self) -> None:
        # PR #91 단계 — generator 미주입 시 skip (PR #92 에서 실 LLM 추가 예정).
        result = run_english_phrase(
            self._inputs(),
            store=FakeStore(), telegram=FakeTelegram(), providers=[],
        )
        assert result.skipped is True
        assert result.sent is False
        assert "generator" in result.reason

    def test_chat_id_missing_returns_skip(self) -> None:
        result = run_english_phrase(
            EnglishPhraseInputs(
                user_id="u", as_of=date(2026, 5, 3),
                theme=Theme.SMALL_TALK, chat_id=None,
            ),
            store=FakeStore(), telegram=FakeTelegram(), providers=[],
            generator=FakeGenerator(),
        )
        assert result.skipped is True
        assert "chat_id" in result.reason

    def test_full_path_sends_and_records(self) -> None:
        store = FakeStore()
        tg = FakeTelegram()
        gen = FakeGenerator()
        result = run_english_phrase(
            self._inputs(), store=store, telegram=tg, providers=[],
            generator=gen,
        )
        assert result.sent is True
        assert result.phrase is not None
        # Telegram 1회 발송.
        assert len(tg.sent) == 1
        assert tg.sent[0][0] == 1234   # chat_id
        # store 에 기록.
        assert len(store.recorded) == 1
        assert store.recorded[0][0] == "user-1"

    def test_generator_failure_no_send_no_record(self) -> None:
        store = FakeStore()
        tg = FakeTelegram()
        gen = FakeGenerator(raise_all_failed=True)
        result = run_english_phrase(
            self._inputs(), store=store, telegram=tg, providers=[],
            generator=gen,
        )
        assert result.sent is False
        assert "llm_failed" in result.reason
        assert tg.sent == []
        assert store.recorded == []

    def test_telegram_failure_no_record(self) -> None:
        # Telegram 발송 실패 → store.record_sent 호출 안 됨 → 다음 cron 재시도 가능.
        store = FakeStore()
        tg = FakeTelegram(fail=True)
        gen = FakeGenerator()
        result = run_english_phrase(
            self._inputs(), store=store, telegram=tg, providers=[],
            generator=gen,
        )
        assert result.sent is False
        assert "telegram" in result.reason
        assert store.recorded == []

    def test_recent_phrases_passed_to_generator(self) -> None:
        gen = FakeGenerator()
        recent = ("Sorry to keep you waiting", "Right away")
        run_english_phrase(
            self._inputs(recent_phrases=recent),
            store=FakeStore(), telegram=FakeTelegram(), providers=[],
            generator=gen,
        )
        # generator 가 recent_phrases 받았는지.
        assert gen.seen_inputs[0].recent_phrases == recent
