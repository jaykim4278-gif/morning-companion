# execution/github_actions/_supabase.py — upsert resolution 헤더 단위 테스트.
# 좁은 role 의 append-only 기록(ignore-duplicates)이 INSERT 권한만으로 동작하도록
# Prefer 헤더가 올바르게 설정되는지 검증 (english_phrase_log / news_sent_log 403 회귀 방지).
from __future__ import annotations

from typing import Any

import httpx

from execution.github_actions._supabase import SupabaseClient


class _RecordingClient:
    """httpx.Client mock — 마지막 POST 의 헤더를 기록."""

    def __init__(self) -> None:
        self.last_headers: dict[str, Any] = {}

    def post(self, url: str, *, headers: dict[str, Any] | None = None,
             params: Any = None, json: Any = None) -> httpx.Response:
        self.last_headers = headers or {}
        return httpx.Response(status_code=204, request=httpx.Request("POST", url))


def _client(rec: _RecordingClient) -> SupabaseClient:
    return SupabaseClient(
        "https://example.supabase.co",
        "jwt-token",
        anon_key="anon",
        client=rec,  # type: ignore[arg-type]
    )


def test_default_upsert_uses_merge_duplicates() -> None:
    # user_preferences 등 UPDATE 권한 있는 테이블은 기존 merge-duplicates 유지.
    rec = _RecordingClient()
    _client(rec).upsert("user_preferences", {"user_id": "u1"}, on_conflict="user_id")
    assert "resolution=merge-duplicates" in rec.last_headers["Prefer"]


def test_ignore_duplicates_uses_ignore_resolution() -> None:
    # english_phrase_log / news_sent_log append-only — INSERT 권한만으로 동작해야 함.
    rec = _RecordingClient()
    _client(rec).upsert(
        "english_phrase_log",
        {"user_id": "u1"},
        on_conflict="user_id,phrase_hash",
        ignore_duplicates=True,
    )
    assert "resolution=ignore-duplicates" in rec.last_headers["Prefer"]
