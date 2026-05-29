"""뉴스 브리핑 dry-run / 실 cron entrypoint.

사용법:
    PYTHONPATH=. python scripts/dry_run_news_briefing.py --mode full

mode:
    full         — 실 Supabase + 실 LLM + 실 Telegram (GitHub Actions cron 사용)
    ai-real      — 실 LLM + console 출력 (Telegram·DB 미호출)  [향후 확장]
    mock         — 로컬 개발용 (향후 확장)
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


def _build_full() -> int:
    from execution.github_actions._supabase import SupabaseClient
    from execution.github_actions._telegram import HttpTelegramSender
    from execution.github_actions.news_briefing import run_news_briefing

    url, jwt, user_id, anon = resolve_supabase_full_env()
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN 필요", file=sys.stderr)
        return 2

    supabase = SupabaseClient(url, jwt, anon_key=anon)
    telegram = HttpTelegramSender(bot_token)
    providers = build_real_providers()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    skip_gate = os.environ.get("NEWS_SKIP_GATE", "false").strip().lower() in ("1", "true", "yes")
    if skip_gate:
        print("(NEWS_SKIP_GATE=true — schedule gate 우회)")
    result = run_news_briefing(
        supabase=supabase,
        telegram=telegram,
        providers=providers,
        user_id=user_id,
        now_utc=datetime.now(timezone.utc),
        skip_gate=skip_gate,
    )
    if result.skipped:
        print(f"⏭ skipped — {result.reason}")
        return 0
    if result.sent:
        print(f"✅ sent — items={result.items_count} indices={result.indices_count}")
        return 0
    print(f"⚠ failed — {result.reason}")
    return 2


def main(argv: list[str] | None = None) -> int:
    reconfigure_stdio_utf8()
    load_dotenv_if_present()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "ai-real", "mock"], default="full")
    args = parser.parse_args(argv)

    if args.mode == "full":
        return _build_full()
    print(f"mode={args.mode} 은 현재 미구현")
    return 1


if __name__ == "__main__":
    sys.exit(main())
