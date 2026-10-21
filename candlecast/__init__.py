"""Candlecast — fast KV-cache inference + free market data for financial
forecasting. Vendored backbone is MIT-licensed; see LICENSE for notice."""
from .constraints import project_ohlc, project_returns, sanitize
from .data import (
    fetch_ccxt,
    fetch_stooq,
    fetch_yfinance,
    list_ccxt_exchanges,
    load_csv,
    next_timestamps,
)
from .engine import CandleEngine
from .predictor import CandlePredictor, load_hub
from .sampling import sample_from_logits

__version__ = "0.1.0"

__all__ = [
    "CandleEngine",
    "CandlePredictor",
    "load_hub",
    "fetch_yfinance",
    "fetch_ccxt",
    "fetch_stooq",
    "list_ccxt_exchanges",
    "load_csv",
    "next_timestamps",
    "project_ohlc",
    "project_returns",
    "sanitize",
    "sample_from_logits",
    "__version__",
]
