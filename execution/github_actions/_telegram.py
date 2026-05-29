# TelegramSender 실구현 — httpx 기반, TelegramSender Protocol 충족.
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class TelegramSendError(RuntimeError):
    """Telegram API 호출 실패."""


class HttpTelegramSender:
    """Bot API `sendMessage` 래퍼.  CLAUDE.md 컨벤션(httpx, timeout 명시) 준수."""

    def __init__(
        self,
        bot_token: str,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        base_url: str = "https://api.telegram.org",
    ) -> None:
        self._token = bot_token
        self._base_url = base_url
        self._client = client or httpx.Client(timeout=timeout)

    def send(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
        disable_web_page_preview: bool = False,
    ) -> None:
        """TelegramSender Protocol 구현.  실패 시 TelegramSendError.

        parse_mode: None (일반 텍스트) · "MarkdownV2" (뉴스 브리핑 등) · "HTML".
        disable_web_page_preview: True 면 첫 링크 자동 미리보기(Instant View 카드) 차단.
            뉴스 브리핑처럼 여러 링크를 병렬로 보여줄 때 혼란스러운 영문 카드 노출 방지용.
        """
        url = f"{self._base_url}/bot{self._token}/sendMessage"
        body: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            body["parse_mode"] = parse_mode
        if disable_web_page_preview:
            body["disable_web_page_preview"] = True
        try:
            resp = self._client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise TelegramSendError(f"network: {exc}") from exc

        if resp.status_code != 200:
            preview = resp.text[:200]
            raise TelegramSendError(f"status {resp.status_code}: {preview}")

        data = resp.json()
        if not data.get("ok"):
            raise TelegramSendError(f"bot api ok=false: {data}")

    def send_document(
        self,
        chat_id: int,
        file_bytes: bytes,
        filename: str,
        caption: str | None = None,
    ) -> None:
        """Bot API `sendDocument` 래퍼 — W7 진료 리포트 PDF 전송용.

        multipart/form-data.  PDF 바이트를 직접 업로드 (디스크 저장 없이).
        실패 시 TelegramSendError.
        """
        url = f"{self._base_url}/bot{self._token}/sendDocument"
        data: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption is not None:
            data["caption"] = caption
        files = {"document": (filename, file_bytes, "application/pdf")}
        try:
            resp = self._client.post(url, data=data, files=files)
        except httpx.HTTPError as exc:
            raise TelegramSendError(f"network: {exc}") from exc

        if resp.status_code != 200:
            preview = resp.text[:200]
            raise TelegramSendError(f"status {resp.status_code}: {preview}")

        body = resp.json()
        if not body.get("ok"):
            raise TelegramSendError(f"bot api ok=false: {body}")


__all__ = ["HttpTelegramSender", "TelegramSendError"]
