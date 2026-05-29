# Layer 4 Sensor — HttpTelegramSender 계약.
from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from execution.github_actions._telegram import (
    HttpTelegramSender,
    TelegramSendError,
)


def _mock(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=1.0)


def test_success_posts_to_send_message() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert "/bot" in str(req.url)
        assert str(req.url).endswith("/sendMessage")
        body = req.read()
        assert b"chat_id" in body
        seen["body"] = body
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})

    s = HttpTelegramSender(bot_token="SECRET", client=_mock(handler))
    s.send(12345, "hello")
    assert b"hello" in seen["body"]


def test_token_is_in_url_not_body() -> None:
    # 토큰 유출 방지 관점: URL 경로에 토큰, body 에는 미포함.
    def handler(req: httpx.Request) -> httpx.Response:
        assert "SECRET" in str(req.url)
        body = req.read()
        assert b"SECRET" not in body
        return httpx.Response(200, json={"ok": True})
    HttpTelegramSender(bot_token="SECRET", client=_mock(handler)).send(1, "x")


def test_non_200_raises() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"ok":false,"description":"bad chat_id"}')
    s = HttpTelegramSender(bot_token="t", client=_mock(handler))
    with pytest.raises(TelegramSendError, match="status 400"):
        s.send(0, "x")


def test_ok_false_raises() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "forbidden"})
    s = HttpTelegramSender(bot_token="t", client=_mock(handler))
    with pytest.raises(TelegramSendError, match="ok=false"):
        s.send(1, "x")


def test_network_error_wrapped() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns fail")
    s = HttpTelegramSender(bot_token="t", client=_mock(handler))
    with pytest.raises(TelegramSendError, match="network"):
        s.send(1, "x")


def test_disable_web_page_preview_omitted_by_default() -> None:
    # 기본 호출은 기존 페이로드 유지 (하위 호환).
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True})
    HttpTelegramSender(bot_token="t", client=_mock(handler)).send(1, "x")
    assert b"disable_web_page_preview" not in seen["body"]


def test_disable_web_page_preview_true_is_forwarded() -> None:
    # 뉴스 브리핑 호출처럼 True 로 넘기면 body 에 포함.
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["body"] = req.read()
        return httpx.Response(200, json={"ok": True})
    HttpTelegramSender(bot_token="t", client=_mock(handler)).send(
        1, "x", disable_web_page_preview=True,
    )
    assert b"disable_web_page_preview" in seen["body"]
    assert b"true" in seen["body"]


# ============================================================
# sendDocument (W7 진료 리포트 PDF)
# ============================================================
def test_send_document_posts_multipart_to_send_document() -> None:
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        assert "/bot" in str(req.url)
        assert str(req.url).endswith("/sendDocument")
        # multipart body 는 Content-Type 이 multipart/form-data; boundary=...
        assert req.headers["content-type"].startswith("multipart/form-data")
        body = req.read()
        seen["body"] = body
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 2}})

    s = HttpTelegramSender(bot_token="SECRET", client=_mock(handler))
    s.send_document(
        chat_id=555,
        file_bytes=b"%PDF-1.4\nfake pdf\n%%EOF",
        filename="report.pdf",
        caption="임상 경과 리포트",
    )
    body = seen["body"]
    # chat_id, caption, filename, application/pdf 모두 멀티파트 본문에 포함
    assert b"555" in body
    assert b"report.pdf" in body
    assert b"application/pdf" in body
    assert "임상 경과 리포트".encode("utf-8") in body
    assert b"%PDF-" in body


def test_send_document_no_caption_is_optional() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})
    s = HttpTelegramSender(bot_token="t", client=_mock(handler))
    # caption 미지정 — 예외 없이 진행
    s.send_document(1, b"%PDF-1.4", "r.pdf")


def test_send_document_non_200_raises() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(413, text="payload too large")
    s = HttpTelegramSender(bot_token="t", client=_mock(handler))
    with pytest.raises(TelegramSendError, match="status 413"):
        s.send_document(1, b"%PDF-", "r.pdf")


def test_send_document_ok_false_raises() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": False, "description": "chat not found"})
    s = HttpTelegramSender(bot_token="t", client=_mock(handler))
    with pytest.raises(TelegramSendError, match="ok=false"):
        s.send_document(1, b"%PDF-", "r.pdf")


def test_send_document_network_error_wrapped() -> None:
    def handler(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns fail")
    s = HttpTelegramSender(bot_token="t", client=_mock(handler))
    with pytest.raises(TelegramSendError, match="network"):
        s.send_document(1, b"%PDF-", "r.pdf")
