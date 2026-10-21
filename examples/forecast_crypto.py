"""Crypto forecast example: Binance BTC/USDT 1h data + KV-cache forecast engine.

    python examples/forecast_crypto.py [--model balanced] [--pred-len 24]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from candlecast import CandlePredictor, fetch_ccxt, next_timestamps

FREQ = "1h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="balanced", choices=["light", "balanced", "large"])
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--exchange", default="binance")
    ap.add_argument("--lookback", type=int, default=384)
    ap.add_argument("--pred-len", type=int, default=24)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    print(f"Fetching {args.symbol} from {args.exchange} ({FREQ})...")
    df = fetch_ccxt(args.exchange, args.symbol, FREQ, limit=max(args.lookback + args.pred_len, 500))
    print(f"  {len(df)} candles, last close {df['close'].iloc[-1]:.2f}")

    split = len(df) - args.pred_len
    hist = df.iloc[:split]
    truth = df.iloc[split:]

    x_ts = hist.index
    y_ts = next_timestamps(hist.index[-1], args.pred_len, FREQ)

    print(f"Loading model: {args.model} (first call downloads weights)...")
    pred = CandlePredictor.from_hub(args.model, project=True, continuity=0.3)

    print("Forecasting with KV-cache engine...")
    out = pred.predict(hist, x_ts, y_ts, args.pred_len,
                       T=1.0, top_p=0.9, sample_count=args.samples,
                       return_samples=True)

    m = out["mean"]
    print("\nForecast (mean):")
    print(m[["open", "high", "low", "close"]].head())
    err = (m["close"] - truth["close"]).abs()
    print(f"\nMAE(close) vs actual: {err.mean():.2f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    tail = hist["close"].iloc[-96:]
    ax.plot(tail.index, tail.values, label="history", color="#444")
    ax.plot(m.index, m["close"].values, label="forecast", color="#d62728")
    ax.fill_between(out["q10"].index, out["q10"]["close"], out["q90"]["close"],
                    color="#d62728", alpha=0.2, label="q10-q90")
    ax.plot(truth.index, truth["close"].values, label="actual", color="#1f77b4", ls="--")
    ax.legend()
    ax.set_title(f"{args.symbol} \u2014 Candlecast ({args.model}, {FREQ})")
    fig.tight_layout()
    out_path = Path(__file__).parent / f"crypto_{args.symbol.replace('/', '_')}.png"
    fig.savefig(out_path, dpi=110)
    print(f"Saved chart: {out_path}")


if __name__ == "__main__":
    main()
