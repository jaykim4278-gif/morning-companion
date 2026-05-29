"""scripts/*.py 공용 유틸 — env 로딩 + 외부 어댑터 (Telegram/AI) + supabase env 확인."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable


# ============================================================
# stdio · dotenv
# ============================================================
def reconfigure_stdio_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            pass


def load_dotenv_if_present(*, announce: bool = True) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root_env = Path(__file__).resolve().parents[1] / ".env"
    if root_env.exists():
        load_dotenv(root_env, override=False)
        if announce:
            print(f"(loaded env from {root_env.name})", file=sys.stderr)


# ============================================================
# 공용 어댑터
# ============================================================
class ConsoleSender:
    def send(self, chat_id: int, text: str) -> None:
        print("\n" + "═" * 60)
        print(f"📱 Telegram (chat_id={chat_id}) — [DRY RUN, not actually sent]")
        print("─" * 60)
        print(text)
        print("═" * 60 + "\n")


class MockProvider:
    name = "mock"

    def __init__(self, responder: Callable[[dict[str, Any], str], str]) -> None:
        self._responder = responder

    def generate(self, payload: dict[str, Any], prompt: str) -> str:
        return self._responder(payload, prompt)


# ============================================================
# Provider 빌더 — Gemini → Groq → OpenRouter 폴백
# ============================================================
def build_real_providers() -> list[Any]:
    from execution.ai_client.gemini_client import GeminiClient
    from execution.ai_client.groq_client import GroqClient
    from execution.ai_client.openrouter_client import OpenRouterClient

    providers: list[Any] = []
    if k := os.environ.get("GEMINI_API_KEY"):
        providers.append(GeminiClient(api_key=k))
    else:
        print("⚠️  GEMINI_API_KEY 없음 — Gemini 스킵", file=sys.stderr)
    if k := os.environ.get("GROQ_API_KEY"):
        providers.append(GroqClient(api_key=k))
    if k := os.environ.get("OPENROUTER_API_KEY"):
        providers.append(OpenRouterClient(api_key=k))
    if not providers:
        print(
            "❌ AI 키가 하나도 없습니다. 최소 GEMINI_API_KEY 를 설정하세요.",
            file=sys.stderr,
        )
        sys.exit(2)
    return providers


def needs_real_telegram(mode: str) -> bool:
    return mode in ("telegram-real", "full")


def build_telegram_sender(mode: str) -> Any:
    if needs_real_telegram(mode):
        from execution.github_actions._telegram import HttpTelegramSender
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token:
            print("❌ TELEGRAM_BOT_TOKEN 필요", file=sys.stderr)
            sys.exit(2)
        return HttpTelegramSender(bot_token=token)
    return ConsoleSender()


def build_chat_id(mode: str) -> int:
    if needs_real_telegram(mode):
        raw = os.environ.get("TELEGRAM_CHAT_ID")
        if not raw:
            print("❌ TELEGRAM_CHAT_ID 필요 (정수)", file=sys.stderr)
            sys.exit(2)
        try:
            return int(raw)
        except ValueError:
            print("❌ TELEGRAM_CHAT_ID 는 정수여야 함", file=sys.stderr)
            sys.exit(2)
    return 0


# ============================================================
# Supabase env (morning-companion 좁은 JWT 패턴)
# ============================================================
def resolve_supabase_full_env() -> tuple[str, str, str, str | None]:
    """full 모드용 (SUPABASE_URL, MORNING_COMPANION_SUPABASE_JWT, OWNER_USER_ID, SUPABASE_ANON_KEY) 반환.

    보안 모델:
      - MORNING_COMPANION_SUPABASE_JWT = morning_companion_writer role 로 발급된 좁은 JWT
      - SUPABASE_ANON_KEY = 공개 anon key (apikey 헤더용, RLS 우회 불가)
        없으면 JWT 를 apikey 에도 재사용 (Supabase 가 허용).
    """
    url = os.environ.get("SUPABASE_URL")
    jwt = os.environ.get("MORNING_COMPANION_SUPABASE_JWT")
    user_id = os.environ.get("OWNER_USER_ID")
    anon = os.environ.get("SUPABASE_ANON_KEY")  # optional
    if not (url and jwt and user_id):
        print(
            "❌ full 모드는 SUPABASE_URL, MORNING_COMPANION_SUPABASE_JWT, OWNER_USER_ID 필요",
            file=sys.stderr,
        )
        sys.exit(2)
    return url, jwt, user_id, anon
