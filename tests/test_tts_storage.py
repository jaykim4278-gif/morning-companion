# execution/github_actions/_tts_storage.py — Supabase Storage TTS 캐시 단위 테스트.
# httpx Mock 으로 외부 HTTP 호출 없이 dedup·URL 생성·PUT·인증헤더 동작 검증.
from __future__ import annotations

from typing import Any

import httpx
import pytest

from execution.github_actions._tts_storage import (
    TtsStorageError,
    TtsStorageUploader,
    text_to_object_path,
)


class _MockClient:
    """httpx.Client mock — (method, url) → {"status": int, "text": str}."""

    def __init__(self, status_map: dict[tuple[str, str], dict[str, Any]] | None = None):
        self._status_map = status_map or {}
        self.request_log: list[tuple[str, str, dict[str, Any]]] = []  # (method, url, headers)

    def get(self, url: str, *, headers: dict[str, Any] | None = None) -> httpx.Response:
        return self._respond("GET", url, headers or {})

    def post(
        self,
        url: str,
        *,
        content: bytes | None = None,
        headers: dict[str, Any] | None = None,
    ) -> httpx.Response:
        return self._respond("POST", url, headers or {}, content=content)

    def _respond(
        self,
        method: str,
        url: str,
        headers: dict[str, Any],
        content: bytes | None = None,
    ) -> httpx.Response:
        self.request_log.append((method, url, headers))
        entry = self._status_map.get((method, url), {"status": 200, "text": ""})
        return httpx.Response(
            status_code=entry["status"],
            text=entry.get("text", ""),
            request=httpx.Request(method, url),
        )


class TestTextToObjectPath:
    def test_same_text_same_path(self) -> None:
        assert text_to_object_path("Hello world") == text_to_object_path("Hello world")

    def test_case_insensitive(self) -> None:
        assert text_to_object_path("Hello World") == text_to_object_path("hello world")

    def test_strips_whitespace(self) -> None:
        assert text_to_object_path("  Hello  ") == text_to_object_path("Hello")

    def test_different_text_different_path(self) -> None:
        assert text_to_object_path("Hello") != text_to_object_path("World")

    def test_path_format(self) -> None:
        p = text_to_object_path("Hello")
        assert p.startswith("v1/")
        assert p.endswith(".mp3")
        parts = p.split("/")
        assert len(parts) == 3
        assert len(parts[1]) == 2  # 2자 subdirectory


class TestTtsStorageUploader:
    def _make(
        self, status_map: dict[tuple[str, str], dict[str, Any]] | None = None
    ) -> tuple[TtsStorageUploader, _MockClient]:
        mock = _MockClient(status_map)
        uploader = TtsStorageUploader(
            url="https://example.supabase.co",
            auth_jwt="test-key",
            client=mock,  # type: ignore[arg-type]
        )
        return uploader, mock

    def test_cache_hit_skips_upload(self) -> None:
        uploader, mock = self._make({
            ("GET", "https://example.supabase.co/storage/v1/object/info/tts-cache/"
                    + text_to_object_path("Hello")): {"status": 200},
        })
        url = uploader.upload_or_get_url("Hello", b"audio-bytes")
        assert "https://example.supabase.co/storage/v1/object/public/tts-cache/" in url
        post_calls = [r for r in mock.request_log if r[0] == "POST"]
        assert post_calls == []

    def test_cache_miss_uploads(self) -> None:
        path = text_to_object_path("Hello")
        uploader, mock = self._make({
            ("GET", f"https://example.supabase.co/storage/v1/object/info/tts-cache/{path}"):
                {"status": 404},
            ("POST", f"https://example.supabase.co/storage/v1/object/tts-cache/{path}"):
                {"status": 200},
        })
        url = uploader.upload_or_get_url("Hello", b"audio-bytes")
        assert url == f"https://example.supabase.co/storage/v1/object/public/tts-cache/{path}"
        posts = [r for r in mock.request_log if r[0] == "POST"]
        assert len(posts) == 1

    def test_put_error_raises(self) -> None:
        path = text_to_object_path("Hello")
        uploader, _mock = self._make({
            ("GET", f"https://example.supabase.co/storage/v1/object/info/tts-cache/{path}"):
                {"status": 404},
            ("POST", f"https://example.supabase.co/storage/v1/object/tts-cache/{path}"):
                {"status": 500, "text": "internal error"},
        })
        with pytest.raises(TtsStorageError, match="HTTP 500"):
            uploader.upload_or_get_url("Hello", b"audio")

    def test_auth_headers_jwt_and_apikey(self) -> None:
        # JWT → Authorization, anon 미지정 시 JWT 가 apikey 로도 재사용.
        path = text_to_object_path("Hi")
        uploader, mock = self._make({
            ("GET", f"https://example.supabase.co/storage/v1/object/info/tts-cache/{path}"):
                {"status": 200},
        })
        uploader.upload_or_get_url("Hi", b"x")
        get_calls = [r for r in mock.request_log if r[0] == "GET"]
        assert any(
            call[2].get("authorization") == "Bearer test-key"
            and call[2].get("apikey") == "test-key"
            for call in get_calls
        )

    def test_anon_key_used_as_apikey(self) -> None:
        # anon_key 지정 시 apikey 헤더는 anon, Authorization 은 JWT.
        mock = _MockClient()
        uploader = TtsStorageUploader(
            url="https://example.supabase.co",
            auth_jwt="jwt-token",
            anon_key="anon-public",
            client=mock,  # type: ignore[arg-type]
        )
        uploader.exists("Hi")
        get_calls = [r for r in mock.request_log if r[0] == "GET"]
        assert any(
            call[2].get("authorization") == "Bearer jwt-token"
            and call[2].get("apikey") == "anon-public"
            for call in get_calls
        )

    def test_exists_check_failure_returns_false(self) -> None:
        class _FailingClient:
            def get(self, url: str, *, headers: dict[str, Any] | None = None) -> httpx.Response:
                raise httpx.RequestError("network down")

            def post(self, url: str, *, content: bytes | None = None,
                     headers: dict[str, Any] | None = None) -> httpx.Response:
                return httpx.Response(status_code=200, request=httpx.Request("POST", url))

        uploader = TtsStorageUploader(
            url="https://example.supabase.co",
            auth_jwt="test-key",
            client=_FailingClient(),  # type: ignore[arg-type]
        )
        url = uploader.upload_or_get_url("Hello", b"audio")
        assert "public/tts-cache/" in url

    def test_get_url_without_upload(self) -> None:
        uploader, mock = self._make()
        url = uploader.get_url("Hello")
        assert "https://example.supabase.co/storage/v1/object/public/tts-cache/" in url
        assert mock.request_log == []

    def test_exists_returns_true_on_200(self) -> None:
        path = text_to_object_path("Hi")
        uploader, _mock = self._make({
            ("GET", f"https://example.supabase.co/storage/v1/object/info/tts-cache/{path}"):
                {"status": 200},
        })
        assert uploader.exists("Hi") is True

    def test_exists_returns_false_on_404(self) -> None:
        path = text_to_object_path("Hi")
        uploader, _mock = self._make({
            ("GET", f"https://example.supabase.co/storage/v1/object/info/tts-cache/{path}"):
                {"status": 404},
        })
        assert uploader.exists("Hi") is False
