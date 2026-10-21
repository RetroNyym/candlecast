"""Free market-data connectors producing a unified OHLCV DataFrame.

All fetchers return a DataFrame indexed by a tz-naive UTC DatetimeIndex with
columns [open, high, low, close, volume] (+ amount when the venue provides it).

Sources (all free, no paid keys required):
  - yfinance : global equities, BIST (.IS), ETFs, indices, forex, futures
  - ccxt     : 100+ crypto exchanges (binance, bybit, okx, kraken, ...)
  - stooq    : daily OHLC for equities/indices/crypto, no key at all
  - csv      : your own files with forgiving column aliases
"""
import io

import numpy as np
import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]
OPTIONAL = ["amount"]

_COLUMN_ALIASES = {
    "open": "open", "high": "high", "low": "low",
    "close": "close", "last": "close", "adj close": "close",
    "adj. close": "close", "close/last": "close", "closing price": "close",
    "vol": "volume", "volumen": "volume", "volume": "volume",
    "amount": "amount", "turnover": "amount", "value": "amount",
    "open_price": "open", "high_price": "high",
    "low_price": "low", "close_price": "close",
}


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any raw OHLCV frame to the unified schema."""
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("timestamp", "timestamps", "date", "datetime", "time", "Date", "Datetime"):
            if col in df.columns:
                df = df.set_index(col)
                break
    idx = pd.to_datetime(df.index, utc=True, format="mixed")
    df = df.copy()
    df.index = idx.tz_convert("UTC").tz_localize(None)
    df.index.name = "timestamps"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    # flatten multiindex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _COLUMN_ALIASES:
            rename[col] = _COLUMN_ALIASES[key]
    df = df.rename(columns=rename)
    # 'Close' and 'Adj Close' can both alias to 'close' â€” keep the first
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing} (got {list(df.columns)})")

    out = df[OHLCV + [c for c in OPTIONAL if c in df.columns]].astype(np.float64)
    if "amount" not in out.columns:
        out["amount"] = out["close"] * out["volume"]
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=OHLCV)
    return out


def fetch_yfinance(symbol: str, period: str = "6mo", interval: str = "1h",
                   start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Yahoo Finance (free). BIST symbols use the .IS suffix, e.g. 'THYAO.IS'.

    Intervals: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo.
    Note: intraday history is limited (~730d for 1h, ~60d for 1m).
    """
    import yfinance as yf

    kwargs = dict(interval=interval, auto_adjust=False, progress=False, threads=False)
    if start or end:
        kwargs.update(start=start, end=end)
    else:
        kwargs["period"] = period
    raw = yf.download(symbol, **kwargs)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol!r}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return _finalize(raw)


def fetch_ccxt(exchange_id: str = "binance", symbol: str = "BTC/USDT",
               timeframe: str = "1h", limit: int = 500,
               since: int | None = None, paginate: int = 0) -> pd.DataFrame:
    """Crypto (and more) via ccxt â€” free public market data, rate-limit aware.

    Args:
        exchange_id: any ccxt exchange id ('binance', 'bybit', 'okx', 'kraken', ...).
        symbol: unified pair, e.g. 'BTC/USDT'.
        timeframe: '1m' '5m' '15m' '1h' '4h' '1d' ...
        limit: rows per request (venue max typically 1000-1500).
        since: epoch ms for the first candle (None = latest).
        paginate: if > 0, fetch that many pages (limit each) going back.
    """
    import ccxt

    klass = getattr(ccxt, exchange_id, None)
    if klass is None:
        raise ValueError(f"Unknown ccxt exchange: {exchange_id!r}")
    ex = klass({"enableRateLimit": True})

    rows = []
    cursor = since
    pages = max(1, paginate) if paginate else 1
    for _ in range(pages):
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + 1
        if len(batch) < limit:
            break
    if not rows:
        raise RuntimeError(f"ccxt {exchange_id} returned no data for {symbol!r}")

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return _finalize(df.set_index("timestamp"))


def list_ccxt_exchanges() -> list[str]:
    import ccxt
    return sorted(ccxt.exchanges)


def fetch_stooq(symbol: str = "btcusd") -> pd.DataFrame:
    """Stooq daily bars â€” zero key, zero auth. Examples: 'spy.us', 'btcusd', 'usdtry'."""
    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    raw = pd.read_csv(io.StringIO(_http_get_text(url)))
    if raw.empty or "Date" not in raw.columns:
        raise RuntimeError(f"stooq returned no data for {symbol!r} (check the symbol)")
    return _finalize(raw)


def _http_get_text(url: str) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "candlecast/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def load_csv(path: str, timestamp_col: str | None = None) -> pd.DataFrame:
    """Load a local CSV; columns are matched case-insensitively with aliases."""
    df = pd.read_csv(path)
    if timestamp_col and timestamp_col in df.columns:
        df = df.rename(columns={timestamp_col: "timestamps"})
    return _finalize(df)


def next_timestamps(last_ts: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    """Future timestamps for the prediction horizon (pure time, no data needed)."""
    return pd.date_range(start=pd.Timestamp(last_ts) + pd.tseries.frequencies.to_offset(freq),
                         periods=periods, freq=freq)
