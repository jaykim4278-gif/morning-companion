# Supabase PostgREST 최소 어댑터.
# SELECT + UPSERT 만 지원하는 경량 클라이언트.
# 인증: 좁은 morning_companion_writer PostgreSQL role JWT.
# apikey 헤더는 공개 anon key, Authorization 헤더는 JWT.
from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class SupabaseError(RuntimeError):
    """Supabase PostgREST 응답이 성공 계약 밖일 때."""


class SupabaseClient:
    """SELECT + UPSERT 만 지원.  httpx.Client 는 DI 로 주입 가능 (테스트 용이).

      - `auth_jwt`: 좁은 role 로 발급된 JWT (Authorization 헤더)
      - `anon_key`: 공개 anon key (apikey 헤더) — 없으면 JWT 를 재사용
    """

    def __init__(
        self,
        url: str,
        auth_jwt: str,
        *,
        anon_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._base = url.rstrip("/") + "/rest/v1"
        # PostgREST 요건: apikey + Authorization.
        # morning-companion 은 apikey 에 공개 anon key, Authorization 에 좁은 JWT.
        self._headers = {
            "apikey": anon_key or auth_jwt,
            "Authorization": f"Bearer {auth_jwt}",
            "Content-Type": "application/json",
        }
        self._client = client or httpx.Client(timeout=timeout)

    def select(
        self,
        table: str,
        filters: Sequence[tuple[str, str]] = (),
        *,
        order: str | None = None,
        select_cols: str | None = None,
    ) -> list[dict[str, Any]]:
        params: list[tuple[str, str]] = list(filters)
        if order:
            params.append(("order", order))
        if select_cols:
            params.append(("select", select_cols))
        try:
            resp = self._client.get(f"{self._base}/{table}", headers=self._headers, params=params)
        except httpx.HTTPError as exc:
            raise SupabaseError(f"select {table}: network {exc}") from exc
        if resp.status_code != 200:
            raise SupabaseError(f"select {table}: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        if not isinstance(data, list):
            raise SupabaseError(f"select {table}: expected list, got {type(data).__name__}")
        return data

    def upsert(
        self,
        table: str,
        row: dict[str, Any],
        on_conflict: str,
        *,
        ignore_duplicates: bool = False,
    ) -> None:
        # ignore_duplicates=True → resolution=ignore-duplicates (ON CONFLICT DO NOTHING):
        #   INSERT 권한만으로 동작 — 좁은 role 이 UPDATE 없이 append-only 기록 가능.
        # False(기본) → merge-duplicates (ON CONFLICT DO UPDATE): UPDATE 권한 필요.
        resolution = "ignore-duplicates" if ignore_duplicates else "merge-duplicates"
        headers = {**self._headers, "Prefer": f"resolution={resolution},return=minimal"}
        try:
            resp = self._client.post(
                f"{self._base}/{table}",
                headers=headers,
                params=[("on_conflict", on_conflict)],
                json=[row],
            )
        except httpx.HTTPError as exc:
            raise SupabaseError(f"upsert {table}: network {exc}") from exc
        if resp.status_code not in (200, 201, 204):
            raise SupabaseError(f"upsert {table}: {resp.status_code} {resp.text[:200]}")


# ============================================================
# §2  헬퍼
# ============================================================
def _to_iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
