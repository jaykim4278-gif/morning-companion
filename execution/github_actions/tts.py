# edge-tts MP3 합성 — 무료 Microsoft Edge 온라인 TTS.
#
# 영어회화 학습자용 음성 자료 생성.  API 키 불필요, quota 없음 (합리적 사용 한).
# 음성: en-US-AvaNeural — 여성·차분하고 명료한 톤.
#
# edge-tts 는 asyncio 기반.  외부 호출자는 sync 환경이라 asyncio.run() 으로 브리지.
from __future__ import annotations

import asyncio
import logging

import edge_tts

logger = logging.getLogger(__name__)

# 차분하고 명료한 여성 음성, 일관 적용.
DEFAULT_VOICE: str = "en-US-AvaNeural"

# edge-tts 가 텍스트당 가끔 30+초 걸려 GH Actions timeout 위험 → 단일 호출 timeout.
_SYNTH_TIMEOUT_SEC: float = 20.0


class TtsError(RuntimeError):
    """TTS 합성 실패 (네트워크·서비스·empty audio)."""


def synthesize_mp3(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """text 를 MP3 bytes 로 합성.  실패·empty 시 TtsError.

    sync wrapper — edge-tts 의 async stream 을 asyncio.run() 으로 호출.
    timeout 초과 시 TtsError 로 변환 (호출자는 graceful degradation).
    """
    text = (text or "").strip()
    if not text:
        raise TtsError("empty text")
    try:
        return asyncio.run(_synthesize_async(text, voice))
    except TtsError:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        # edge-tts·aiohttp 의 다양한 예외를 단일 타입으로 통합.
        raise TtsError(f"edge-tts failed: {exc}") from exc


async def _synthesize_async(text: str, voice: str) -> bytes:
    """edge-tts stream → MP3 bytes 조립."""
    communicate = edge_tts.Communicate(text, voice)
    chunks: list[bytes] = []

    async def _collect() -> None:
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                chunks.append(chunk["data"])

    try:
        await asyncio.wait_for(_collect(), timeout=_SYNTH_TIMEOUT_SEC)
    except asyncio.TimeoutError as exc:
        raise TtsError(f"edge-tts timeout ({_SYNTH_TIMEOUT_SEC}s)") from exc

    if not chunks:
        raise TtsError("no audio chunks from edge-tts")
    return b"".join(chunks)


__all__ = ["DEFAULT_VOICE", "TtsError", "synthesize_mp3"]
