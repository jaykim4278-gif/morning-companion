# W8a PR #2 — market_indices 테스트 (yfinance mock + 부분 실패 허용).
# docs/design/w8a-news-briefing-design.md §2 · §10.1.
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from execution.github_actions.news.market_indices import (
    INDEX_CATALOG,
    IndexFetchError,
    fetch_indices,
    fetch_one,
)


class _FakeHistory:
    """pandas DataFrame 흉내 — closes + index 만 필요."""

    def __init__(self, closes: list[float], index_dates: list[datetime]):
        self._closes = closes
        self._idx = index_dates

    def __len__(self) -> int:
        return len(self._closes)

    def __getitem__(self, key: str) -> Any:
        if key == "Close":
            return SimpleNamespace(tolist=lambda: list(self._closes))
        raise KeyError(key)

    @property
    def index(self) -> list[Any]:
        # 마지막 원소 .date() 호출만 검증.
        return [SimpleNamespace(date=lambda d=d: d.date()) for d in self._idx]


class FakeYfinance:
    """Ticker(symbol).history(period='ytd') 체인 모킹."""

    def __init__(self, history_by_symbol: dict[str, Any]):
        self._history = history_by_symbol

    def Ticker(self, symbol: str):  # noqa: N802 — yfinance 의 클래스명
        h = self._history.get(symbol)

        class _T:
            def history(self_inner, period: str):  # noqa: N805
                if isinstance(h, Exception):
                    raise h
                return h

        return _T()


class TestFetchOne:
    def test_computes_day_and_ytd_change(self) -> None:
        # closes = [5000, 5050, 5100], ytd_start=5000, prev=5050, last=5100
        # day = (5100-5050)/5050 = +0.9901%
        # ytd = (5100-5000)/5000 = +2.0%
        hist = _FakeHistory(
            closes=[5000.0, 5050.0, 5100.0],
            index_dates=[datetime(2026, 1, 2), datetime(2026, 4, 19), datetime(2026, 4, 20)],
        )
        fy = FakeYfinance({"^GSPC": hist})
        q = fetch_one("^GSPC", "S&P 500", "", yfinance_module=fy)
        assert q.symbol == "^GSPC"
        assert q.last_close == 5100.0
        assert q.day_change_pct == pytest.approx(0.9901, abs=0.001)
        assert q.ytd_change_pct == pytest.approx(2.0, abs=0.001)
        assert q.as_of.isoformat() == "2026-04-20"

    def test_insufficient_history_raises(self) -> None:
        # 1거래일만 있으면 day_change 계산 불가 → IndexFetchError.
        hist = _FakeHistory([5000.0], [datetime(2026, 4, 20)])
        fy = FakeYfinance({"^GSPC": hist})
        with pytest.raises(IndexFetchError):
            fetch_one("^GSPC", "S&P 500", "", yfinance_module=fy)

    def test_yfinance_error_raises(self) -> None:
        fy = FakeYfinance({"^GSPC": RuntimeError("network down")})
        with pytest.raises(IndexFetchError):
            fetch_one("^GSPC", "S&P 500", "", yfinance_module=fy)

    def test_unit_pct_preserved(self) -> None:
        hist = _FakeHistory(
            closes=[4.0, 4.3, 4.32],
            index_dates=[datetime(2026, 1, 2), datetime(2026, 4, 19), datetime(2026, 4, 20)],
        )
        fy = FakeYfinance({"^TNX": hist})
        q = fetch_one("^TNX", "10Y 국채", "%", yfinance_module=fy)
        assert q.unit == "%"


class TestFetchIndices:
    def test_catalog_fully_loaded(self) -> None:
        # 두 심볼 모두 성공.
        h_gspc = _FakeHistory(
            [5000.0, 5050.0, 5100.0],
            [datetime(2026, 1, 2), datetime(2026, 4, 19), datetime(2026, 4, 20)],
        )
        h_tnx = _FakeHistory(
            [4.0, 4.3, 4.32],
            [datetime(2026, 1, 2), datetime(2026, 4, 19), datetime(2026, 4, 20)],
        )
        fy = FakeYfinance({"^GSPC": h_gspc, "^TNX": h_tnx})
        results = fetch_indices(yfinance_module=fy)
        assert [q.symbol for q in results] == ["^GSPC", "^TNX"]

    def test_partial_failure_only_returns_success(self) -> None:
        # S&P 성공, TNX 실패 → S&P 만 반환.
        h_gspc = _FakeHistory(
            [5000.0, 5050.0, 5100.0],
            [datetime(2026, 1, 2), datetime(2026, 4, 19), datetime(2026, 4, 20)],
        )
        fy = FakeYfinance({"^GSPC": h_gspc, "^TNX": RuntimeError("gone")})
        results = fetch_indices(yfinance_module=fy)
        assert [q.symbol for q in results] == ["^GSPC"]

    def test_all_failure_returns_empty(self) -> None:
        fy = FakeYfinance({"^GSPC": RuntimeError("a"), "^TNX": RuntimeError("b")})
        assert fetch_indices(yfinance_module=fy) == []


class TestCatalogContract:
    def test_catalog_has_gspc_and_tnx(self) -> None:
        symbols = [row[0] for row in INDEX_CATALOG]
        assert "^GSPC" in symbols
        assert "^TNX" in symbols
