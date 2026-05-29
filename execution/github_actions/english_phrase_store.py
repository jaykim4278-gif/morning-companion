# W9 PR #93 — Supabase 기반 EnglishPhraseStore 구현체.
# english_phrase.run_english_phrase 가 주입받아 사용.
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from execution.github_actions._schedule import mark_english_sent
from execution.github_actions._supabase import SupabaseClient, SupabaseError
from execution.github_actions.english_phrase import EnglishPhrase, hash_phrase

logger = logging.getLogger(__name__)


class SupabaseEnglishPhraseStore:
    """english_phrase_log + user_preferences.last_english_sent_at 양방향."""

    def __init__(self, client: SupabaseClient) -> None:
        self._client = client

    def fetch_recent_phrases(self, user_id: str, days: int = 28) -> list[str]:
        """최근 N일 발송된 phrase_en 리스트 (LLM 중복 회피용)."""
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
        # ISO timestamp 문자열로 변환.
        cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        try:
            rows = self._client.select(
                "english_phrase_log",
                filters=[
                    ("user_id", f"eq.{user_id}"),
                    ("sent_at", f"gte.{cutoff_iso}"),
                ],
                order="sent_at.desc",
                select_cols="phrase_en",
            )
        except SupabaseError as exc:
            logger.warning("fetch_recent_phrases 실패 — 빈 list 반환: %s", exc)
            return []
        return [r["phrase_en"] for r in rows if r.get("phrase_en")]

    def record_sent(self, user_id: str, phrase: EnglishPhrase, sent_at: datetime) -> None:
        """english_phrase_log INSERT + last_english_sent_at upsert.

        UNIQUE(user_id, phrase_hash) 위반 시 (4주 내 같은 phrase 재시도) — 23505 흡수,
        record_sent 만 skip 하고 mark_english_sent 는 진행 (사용자에겐 발송됐으므로).
        """
        sent_at_utc = sent_at.astimezone(timezone.utc)
        row = {
            "user_id": user_id,
            "sent_date": sent_at_utc.date().isoformat(),
            "theme": phrase.theme.value,
            "phrase_en": phrase.phrase_en,
            "phrase_hash": hash_phrase(phrase.phrase_en),
            "sent_at": sent_at_utc.isoformat().replace("+00:00", "Z"),
        }
        try:
            # ignore-duplicates: english_phrase_log 는 append-only (UNIQUE 충돌 시 무시).
            # 좁은 role 은 INSERT 권한만 → merge-duplicates(=UPDATE 필요) 는 403.
            self._client.upsert(
                "english_phrase_log",
                row,
                on_conflict="user_id,phrase_hash",
                ignore_duplicates=True,
            )
        except SupabaseError as exc:
            logger.warning("english_phrase_log insert 실패: %s", exc)
        # last_english_sent_at 마크 (다음 cron 중복 차단 핵심).
        mark_english_sent(self._client, user_id, sent_at_utc)


__all__ = ["SupabaseEnglishPhraseStore"]
