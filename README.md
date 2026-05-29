# morning-companion

Daily English phrase + morning news/markets briefing — delivered to Telegram on a schedule.

Backed by a small Python pipeline running on GitHub Actions cron.  Uses only public
data sources (RSS, Yahoo Finance, public weather APIs) plus LLM APIs (Gemini / Groq /
OpenRouter with fallback).

## Features

- **English phrase of the day** — short coaching message with example scenarios and
  pronunciation, themed on a weekly rotation
- **Morning news/markets briefing** — market indices, today's weather, curated
  headlines across configurable categories (general, markets, tech, restaurant/industry,
  local), all translated and clustered to avoid duplicates

## Configuration (env / GitHub Secrets)

| Name | Purpose |
|---|---|
| `GEMINI_API_KEY` | LLM (primary) |
| `GROQ_API_KEY` | LLM (fallback) |
| `OPENROUTER_API_KEY` | LLM (fallback) |
| `TELEGRAM_BOT_TOKEN` | Bot for sending |
| `TELEGRAM_CHAT_ID` | Destination chat (integer) |
| `SUPABASE_URL` | PostgREST endpoint |
| `SUPABASE_ANON_KEY` | apikey header |
| `MORNING_COMPANION_SUPABASE_JWT` | Auth header (narrow role JWT) |
| `OWNER_USER_ID` | Owner UUID for RLS |
| `USER_CITY` | English display name for local section |
| `USER_CITY_KO` | Korean display name |
| `USER_LAT` | Latitude for weather API |
| `USER_LON` | Longitude for weather API |
| `USER_TZ` | IANA timezone |
| `USER_NWS_ZONE_PREFIX` | NWS area code prefix for active alerts |
| `USER_PERSONA_PROMPT` | Optional learner profile injected into English coaching prompt |
| `USER_INTERESTS` | Optional comma-separated keywords for news clustering hint |

Without `USER_*` location secrets, weather + local news sections are skipped.
Without `USER_PERSONA_PROMPT`, a generic intermediate learner profile is used.

## Local run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# create .env with the secrets above
PYTHONPATH=. python scripts/dry_run_english_phrase.py --mode preview
PYTHONPATH=. python scripts/dry_run_news_briefing.py --mode full
```

## License

MIT — see `LICENSE`.
