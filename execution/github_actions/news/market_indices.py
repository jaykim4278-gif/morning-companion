# W8a PR #2 — yfinance 지수 fetch (S&P 500 + 10Y 국채).
# docs/design/w8a-news-briefing-design.md §2 근거.
#
# 실패 허용 설계:
# - 심볼 단위 try/except → 하나 실패해도 다른 심볼은 정상 반환
# - 전체 실패 시 호출자는 IndexFetchError 대신 빈 리스트로 처리 가능
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexQuote:
    symbol: str              # '^GSPC', '^TNX'
    display_name: str        # 'S&P 500', '10Y 국채'
    last_close: float        # 전일 종가
    day_change_pct: float    # 일일 변동 % (signed)
    ytd_change_pct: float    # YTD 변동 % (signed)
    unit: str                # '' (지수) · '%' (수익률 symbol)
    as_of: date              # 종가 기준일 (UTC date)


# 심볼 카탈로그 — Plan §2.1 확정.
INDEX_CATALOG: tuple[tuple[str, str, str], ...] = (
    ("^GSPC", "S&P 500", ""),
    ("^TNX",  "10Y 국채", "%"),
)


class IndexFetchError(RuntimeError):
    """yfinance fetch 실패.  호출자는 단일 심볼 실패 허용하려면 try/except."""


def fetch_one(
    symbol: str,
    display_name: str,
    unit: str,
    *,
    yfinance_module: Any = None,
) -> IndexQuote:
    """단일 심볼 yfinance fetch.  실패 시 IndexFetchError 발생.

    Args:
        symbol: Yahoo Finance 심볼 ('^GSPC', '^TNX').
        display_name: 메시지 표시용 이름.
        unit: '' (지수) 또는 '%' (yield).
        yfinance_module: 테스트 주입용 (기본 실 yfinance).

    Returns:
        IndexQuote (종가, 일일 변동, YTD 변동).
    """
    if yfinance_module is None:
        import yfinance as yfinance_module  # type: ignore[no-redef]

    try:
        ticker = yfinance_module.Ticker(symbol)
        hist = ticker.history(period="ytd")
    except Exception as exc:  # pylint: disable=broad-except
        raise IndexFetchError(f"{symbol}: yfinance error {exc}") from exc

    if hist is None or len(hist) < 2:
        raise IndexFetchError(f"{symbol}: 거래일 2 이하")

    closes = hist["Close"].tolist()
    last_close = float(closes[-1])
    prev_close = float(closes[-2])
    ytd_start = float(closes[0])

    day_change_pct = ((last_close - prev_close) / prev_close) * 100.0
    ytd_change_pct = ((last_close - ytd_start) / ytd_start) * 100.0

    # 종가 날짜 — DatetimeIndex 의 마지막 값.
    idx = hist.index[-1]
    as_of = (
        idx.date() if hasattr(idx, "date")
        else (datetime.fromisoformat(str(idx)).date())
    )

    return IndexQuote(
        symbol=symbol,
        display_name=display_name,
        last_close=last_close,
        day_change_pct=day_change_pct,
        ytd_change_pct=ytd_change_pct,
        unit=unit,
        as_of=as_of,
    )


def fetch_indices(
    *,
    yfinance_module: Any = None,
    catalog: tuple[tuple[str, str, str], ...] = INDEX_CATALOG,
) -> list[IndexQuote]:
    """모든 지수 fetch.  개별 심볼 실패 허용 — 성공한 것만 반환.

    호출자가 빈 리스트 수신 시 '📈 시장 지수 데이터 일시 불가' 메시지로 대체.
    """
    results: list[IndexQuote] = []
    for symbol, display_name, unit in catalog:
        try:
            quote = fetch_one(symbol, display_name, unit, yfinance_module=yfinance_module)
            results.append(quote)
        except IndexFetchError as exc:
            logger.warning("지수 fetch 실패 %s — skip: %s", symbol, exc)
    return results


__all__ = [
    "IndexQuote",
    "INDEX_CATALOG",
    "IndexFetchError",
    "fetch_one",
    "fetch_indices",
]
