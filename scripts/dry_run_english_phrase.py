#!/usr/bin/env python3
"""영어회화 한 마디 dry-run / 실 cron entrypoint.

실행:
  PYTHONPATH=. python scripts/dry_run_english_phrase.py --mode preview
  PYTHONPATH=. python scripts/dry_run_english_phrase.py --mode full
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._common import (
    build_real_providers,
    load_dotenv_if_present,
    reconfigure_stdio_utf8,
    resolve_supabase_full_env,
)


def main() -> int:
    reconfigure_stdio_utf8()
    load_dotenv_if_present()

    parser = argparse.ArgumentParser(description="영어회화 한 마디 발송")
    parser.add_argument("--mode", choices=["preview", "full"], default="preview")
    parser.add_argument(
        "--theme",
        choices=["small_talk", "soft_skills", "reactions", "idioms"],
        help="테마 강제 (디버깅용 — 기본은 ISO week %% 4 자동 결정)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    url, jwt, user_id, anon = resolve_supabase_full_env()

    from execution.github_actions._schedule import (
        fetch_user_schedule,
        resolve_chat_id,
        should_send_english_phrase,
    )
    from execution.github_actions._supabase import SupabaseClient
    from execution.github_actions._telegram import HttpTelegramSender
    from execution.github_actions.english_phrase import (
        EnglishPhraseInputs,
        Theme,
        run_english_phrase,
        select_theme_for_date,
    )
    from execution.github_actions.english_phrase_llm import LlmEnglishPhraseGenerator
    from execution.github_actions.english_phrase_store import SupabaseEnglishPhraseStore

    client = SupabaseClient(url, jwt, anon_key=anon)
    schedule = fetch_user_schedule(client, user_id)
    now_utc = datetime.now(timezone.utc)

    skip_gate = os.environ.get("ENGLISH_SKIP_GATE", "").lower() in ("true", "1", "yes")
    if not skip_gate and not should_send_english_phrase(schedule, now_utc):
        print(
            f"schedule gate skip — english_phrase_time={schedule.english_phrase_time} "
            f"last_english_sent_at={schedule.last_english_sent_at}"
        )
        return 0

    local_today = now_utc.astimezone(schedule.timezone).date()
    theme = Theme(args.theme) if args.theme else select_theme_for_date(local_today)

    store = SupabaseEnglishPhraseStore(client)
    recent = store.fetch_recent_phrases(user_id, days=28)
    print(f"theme={theme.value} recent={len(recent)} phrases")

    chat_id = resolve_chat_id(
        schedule,
        env_fallback=int(os.environ["TELEGRAM_CHAT_ID"])
        if os.environ.get("TELEGRAM_CHAT_ID")
        else None,
    )

    inp = EnglishPhraseInputs(
        user_id=user_id,
        as_of=local_today,
        theme=theme,
        recent_phrases=tuple(recent),
        chat_id=chat_id,
    )

    # TTS 주입 — 영문 문장마다 🔊 듣기 링크 임베드.
    # Storage 정책 미설정·합성 실패 시 graceful degradation (음성 없이 텍스트만 발송).
    from execution.github_actions._tts_storage import TtsStorageUploader
    from execution.github_actions.tts import synthesize_mp3

    tts_uploader = TtsStorageUploader(url, jwt, anon_key=anon)
    generator = LlmEnglishPhraseGenerator(
        synthesizer=synthesize_mp3, uploader=tts_uploader,
    )

    if args.mode == "preview":
        from execution.ai_client.fallback_chain import AllProvidersFailed
        providers = build_real_providers()
        try:
            phrase = generator.generate(inp, providers)
        except AllProvidersFailed as exc:
            print(f"All providers failed: {exc}", file=sys.stderr)
            return 1
        print("=" * 60)
        print(f"theme: {phrase.theme.value}")
        print(f"phrase_en: {phrase.phrase_en}")
        print("-" * 60)
        print(phrase.body_md)
        print("=" * 60)
        return 0

    providers = build_real_providers()
    raw_telegram = HttpTelegramSender(bot_token=os.environ["TELEGRAM_BOT_TOKEN"])

    class _HtmlTelegram:
        def send(self, chat_id: int, text: str) -> None:
            raw_telegram.send(
                chat_id, text, parse_mode="HTML", disable_web_page_preview=True,
            )

    result = run_english_phrase(
        inp,
        store=store,
        telegram=_HtmlTelegram(),
        providers=providers,
        generator=generator,
        now_utc=now_utc,
    )
    print(f"sent={result.sent} skipped={result.skipped} reason={result.reason}")
    if result.phrase:
        print(f"phrase_en={result.phrase.phrase_en}")
    return 0 if (result.sent or result.skipped) else 1


if __name__ == "__main__":
    sys.exit(main())
