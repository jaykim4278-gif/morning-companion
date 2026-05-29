# execution/github_actions/tts.py — edge-tts MP3 합성 단위 테스트.
# edge-tts 자체는 네트워크 의존 — fake stream 으로 sync wrapper 동작 검증.
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import patch

import pytest

from execution.github_actions.tts import DEFAULT_VOICE, TtsError, synthesize_mp3


class _FakeCommunicate:
    """edge_tts.Communicate stub — 임의 chunk 시퀀스 반환."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self._chunks = chunks

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        for chunk in self._chunks:
            yield chunk


class TestSynthesizeMp3:
    def test_returns_audio_bytes_joined(self) -> None:
        # audio chunk 2개 + 무관한 metadata chunk 1개 — audio 만 join.
        fake = _FakeCommunicate([
            {"type": "audio", "data": b"abc"},
            {"type": "WordBoundary", "offset": 0},
            {"type": "audio", "data": b"def"},
        ])
        with patch("execution.github_actions.tts.edge_tts.Communicate", return_value=fake):
            audio = synthesize_mp3("Hello world")
        assert audio == b"abcdef"

    def test_empty_text_raises(self) -> None:
        with pytest.raises(TtsError, match="empty"):
            synthesize_mp3("")
        with pytest.raises(TtsError, match="empty"):
            synthesize_mp3("   ")

    def test_no_audio_chunks_raises(self) -> None:
        # 모든 chunk 가 audio 아니면 합성 실패.
        fake = _FakeCommunicate([{"type": "WordBoundary", "offset": 0}])
        with patch("execution.github_actions.tts.edge_tts.Communicate", return_value=fake):
            with pytest.raises(TtsError, match="no audio chunks"):
                synthesize_mp3("Hello")

    def test_network_exception_wrapped_as_tts_error(self) -> None:
        # aiohttp·기타 예외 모두 TtsError 로 통합.
        class _Failing:
            async def stream(self):  # type: ignore[no-untyped-def]
                raise ConnectionError("network unreachable")
                yield  # pragma: no cover  (unreachable, marker)
        with patch("execution.github_actions.tts.edge_tts.Communicate",
                   return_value=_Failing()):
            with pytest.raises(TtsError, match="network unreachable"):
                synthesize_mp3("Hello")

    def test_default_voice_constant(self) -> None:
        assert DEFAULT_VOICE == "en-US-AvaNeural"

    def test_voice_passed_to_communicate(self) -> None:
        fake = _FakeCommunicate([{"type": "audio", "data": b"x"}])
        with patch("execution.github_actions.tts.edge_tts.Communicate") as mock_cls:
            mock_cls.return_value = fake
            synthesize_mp3("Test", voice="en-US-AndrewNeural")
            mock_cls.assert_called_once_with("Test", "en-US-AndrewNeural")
