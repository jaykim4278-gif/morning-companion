# Supabase Storage TTS 캐시 + 업로드.
#
# 영어회화 음성 자료를 public bucket "tts-cache" 에 보관.
# SHA256(text) 해시 파일명 → 같은 영문 문장은 한 번만 합성·저장 (dedup).
# Telegram 메시지의 🔊 듣기 링크가 이 URL 을 가리킴 → 모바일 브라우저가 MP3 직접 재생.
#
# 인증 (좁은 role 패턴 — SupabaseClient 와 동일):
#   - Authorization: Bearer <좁은 role JWT>
#   - apikey: 공개 anon key (없으면 JWT 재사용)
#   - 업로드(PUT)·조회(HEAD) 는 storage.objects 의 INSERT/SELECT 정책이 좁은 role 을
#     허용해야 동작.  정책 미설정 시 403 → 호출자(renderer)가 graceful degradation 으로
#     링크 생략 (메시지 자체는 정상 발송).
#   - public read 는 bucket 의 public=true 정책으로 처리 (링크 클릭은 인증 불요).
#   - URL 에 user_id·PII 미포함 (해시만) — 공개 URL 도 정보 누출 없음.
#
# 정리:
#   - 파일은 기본 영구 보관 (같은 phrase 가 미래에도 재사용 → cache hit 가치 ↑).
#   - 1GB(무료 한도) 초과 시 cron 으로 오래된 파일 정리 (현 시점 불필요).
from __future__ import annotations

import hashlib
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# 버킷명은 Supabase 에 수동 생성 (public).
_BUCKET_NAME: str = "tts-cache"

# 경로 prefix — voice/format 변경 시 bump 로 cache busting.  현재 v1: Ava MP3.
_OBJECT_PREFIX: str = "v1"


class TtsStorageError(RuntimeError):
    """Supabase Storage 업로드·조회 실패."""


def text_to_object_path(text: str) -> str:
    """text 의 SHA256 해시 → object path.

    같은 영문 → 같은 path → 같은 public URL (cross-day 재사용).
    소문자·trim 정규화로 case-insensitive dedup.

    예시 경로: v1/a3/a3f5b8...d2.mp3
        - 2자 prefix subdirectory 로 한 폴더 내 파일 폭증 회피.
    """
    norm = (text or "").strip().lower()
    digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()
    return f"{_OBJECT_PREFIX}/{digest[:2]}/{digest}.mp3"


class TtsStorageUploader:
    """Supabase Storage TTS 캐시 — text → public URL.

    cache hit 시 업로드 생략 (네트워크·시간 절약).
    좁은 role JWT + 공개 anon key 로 인증 (service_role 미사용 — Public repo 안전).
    """

    def __init__(
        self,
        url: str,
        auth_jwt: str,
        *,
        anon_key: str | None = None,
        bucket: str = _BUCKET_NAME,
        client: httpx.Client | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._url = url.rstrip("/")
        self._jwt = auth_jwt
        self._apikey = anon_key or auth_jwt
        self._bucket = bucket
        self._client = client or httpx.Client(timeout=timeout)

    def upload_or_get_url(self, text: str, audio_bytes: bytes) -> str:
        """text 에 해당하는 mp3 public URL 반환.

        - 이미 storage 에 있으면 그대로 URL 반환 (audio_bytes 무시).
        - 없으면 PUT 후 URL 반환.
        """
        object_path = text_to_object_path(text)
        public_url = self._public_url(object_path)

        if self._exists(object_path):
            logger.debug("TTS cache hit — %s", object_path)
            return public_url

        self._put(object_path, audio_bytes)
        logger.info("TTS uploaded — %s (%d bytes)", object_path, len(audio_bytes))
        return public_url

    def exists(self, text: str) -> bool:
        """text 의 TTS 가 이미 storage 에 있는지 — 합성 비용 절약용 사전 체크."""
        return self._exists(text_to_object_path(text))

    def get_url(self, text: str) -> str:
        """업로드 없이 URL 만 계산 — 호출자가 exists 로 확인 후 사용."""
        return self._public_url(text_to_object_path(text))

    # ------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------
    def _public_url(self, object_path: str) -> str:
        return f"{self._url}/storage/v1/object/public/{self._bucket}/{object_path}"

    def _exists(self, object_path: str) -> bool:
        """HEAD-equivalent (info GET) 로 존재 확인.  실패·timeout 시 보수적으로 False."""
        endpoint = f"{self._url}/storage/v1/object/info/{self._bucket}/{object_path}"
        try:
            res = self._client.get(endpoint, headers=self._auth_headers())
        except httpx.RequestError as exc:
            logger.warning("TTS exists check 실패 — 보수적으로 재업로드: %s", exc)
            return False
        return res.status_code == 200

    def _put(self, object_path: str, audio_bytes: bytes) -> None:
        """Storage PUT — x-upsert: true 로 race condition 흡수."""
        endpoint = f"{self._url}/storage/v1/object/{self._bucket}/{object_path}"
        try:
            res = self._client.post(
                endpoint,
                content=audio_bytes,
                headers={
                    **self._auth_headers(),
                    "content-type": "audio/mpeg",
                    "x-upsert": "true",
                    "cache-control": "public, max-age=31536000, immutable",
                },
            )
        except httpx.RequestError as exc:
            raise TtsStorageError(f"storage PUT network error: {exc}") from exc

        if res.status_code >= 400:
            raise TtsStorageError(
                f"storage PUT {object_path} failed — HTTP {res.status_code}: {res.text[:200]}"
            )

    def _auth_headers(self) -> dict[str, Any]:
        return {
            "authorization": f"Bearer {self._jwt}",
            "apikey": self._apikey,
        }


__all__ = [
    "TtsStorageError",
    "TtsStorageUploader",
    "text_to_object_path",
]
