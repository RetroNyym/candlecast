"""BIST equity forecast example via Yahoo Finance (free, .IS suffix).

    python examples/forecast_bist.py --symbol THYAO.IS
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from candlecast import CandlePredictor, fetch_yfinance, next_timestamps

FREQ = "1d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="THYAO.IS")
    ap.add_argument("--model", default="balanced", choices=["light", "balanced", "large"])
    ap.add_argument("--lookback", type=int, default=256)
    ap.add_argument("--pred-len", type=int, default=10)
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    print(f"Fetching {args.symbol} daily from Yahoo Finance...")
    df = fetch_yfinance(args.symbol, period="2y", interval=FREQ)
    print(f"  {len(df)} days, last close {df['close'].iloc[-1]:.2f}")

    split = len(df) - args.pred_len
    hist, truth = df.iloc[:split], df.iloc[split:]
    y_ts = next_timestamps(hist.index[-1], args.pred_len, FREQ)

    print(f"Loading model: {args.model}...")
    pred = CandlePredictor.from_hub(args.model, project=True, continuity=0.5)

    out = pred.predict(hist, hist.index, y_ts, args.pred_len,
                       T=0.8, top_p=0.9, sample_count=args.samples, return_samples=True)
    m = out["mean"]
    print("\nForecast (mean close):", [f"{v:.2f}" for v in m["close"]])
    print("Actual close         :", [f"{v:.2f}" for v in truth["close"]])

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist.index[-60:], hist["close"].iloc[-60:], label="history", color="#444")
    ax.plot(m.index, m["close"], label="forecast", color="#d62728")
    ax.fill_between(m.index, out["q10"]["close"], out["q90"]["close"],
                    color="#d62728", alpha=0.2, label="q10-q90")
    ax.plot(truth.index, truth["close"], label="actual", color="#1f77b4", ls="--")
    ax.legend()
    ax.set_title(f"{args.symbol} \u2014 Candlecast ({args.model})")
    fig.tight_layout()
    out_path = Path(__file__).parent / f"bist_{args.symbol.replace('.', '_')}.png"
    fig.savefig(out_path, dpi=110)
    print(f"Saved chart: {out_path}")


if __name__ == "__main__":
    main()
