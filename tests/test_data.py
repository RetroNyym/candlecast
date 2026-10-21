import pandas as pd
import pytest

from candlecast.data import _finalize, next_timestamps


def test_finalize_maps_aliases_and_adds_amount():
    raw = pd.DataFrame({
        "Date": ["2025-01-06", "2025-01-07"],
        "Open": [10.0, 11.0],
        "High": [12.0, 12.5],
        "Low": [9.5, 10.8],
        "Close/Last": [11.5, 12.0],
        "Volume": [1000, 1500],
    })
    out = _finalize(raw)
    assert list(out.columns[:5]) == ["open", "high", "low", "close", "volume"]
    assert "amount" in out.columns
    assert str(out.index.tz) == "UTC" or out.index.tz is None
    assert len(out) == 2


def test_finalize_dedupes_and_sorts():
    raw = pd.DataFrame({
        "timestamps": ["2025-01-07", "2025-01-06", "2025-01-07"],
        "open": [1.0, 2.0, 3.0],
        "high": [1.0, 2.0, 3.0],
        "low": [1.0, 2.0, 3.0],
        "close": [1.0, 2.0, 3.0],
        "volume": [1.0, 1.0, 1.0],
    })
    out = _finalize(raw)
    assert len(out) == 2
    assert out.index.is_monotonic_increasing
    assert out["open"].iloc[-1] == 3.0


def test_next_timestamps():
    ts = next_timestamps(pd.Timestamp("2025-01-06 09:00"), 3, "5min")
    assert list(ts.astype(str)) == [
        "2025-01-06 09:05:00", "2025-01-06 09:10:00", "2025-01-06 09:15:00"]


@pytest.mark.network
def test_fetch_ccxt_binance_btc():
    from candlecast.data import fetch_ccxt
    df = fetch_ccxt("binance", "BTC/USDT", "1h", limit=100)
    assert len(df) >= 90
    assert (df["high"] >= df["low"]).all()


@pytest.mark.network
def test_fetch_yfinance_bist():
    from candlecast.data import fetch_yfinance
    df = fetch_yfinance("THYAO.IS", period="1mo", interval="1d")
    assert len(df) > 10
    assert (df["high"] >= df["low"]).all()
