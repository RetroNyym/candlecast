"""Benchmark: original full-recompute loop vs KV-cache engine.

Usage:
    python bench.py               # tiny synthetic model (no network)
    python bench.py --hub balanced   # real model from Hugging Face
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from candlecast import reference
from candlecast.backbone import CandleModel, CandleTokenizer
from candlecast.engine import CandleEngine
import candlecast.engine as engine_mod
import candlecast.reference as ref_mod


def _argmax(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    return torch.topk(logits, k=1, dim=-1).indices


def make_data(n0, pred_len, feats=6, seed=11):
    g = torch.Generator().manual_seed(seed)
    close = 100 + torch.cumsum(torch.randn(n0 + pred_len, generator=g) * 0.5, 0)
    o = close + torch.randn(n0 + pred_len, generator=g) * 0.1
    h = torch.maximum(o, close) + 0.2
    l = torch.minimum(o, close) - 0.2
    vol = torch.rand(n0 + pred_len, generator=g) * 1000 + 100
    x = torch.stack([o, h, l, close, vol, vol * close], -1)[:n0][None].float()
    stamps = pd.date_range("2025-01-06 09:00", periods=n0 + pred_len, freq="5min")
    arr = np.stack([stamps.minute, stamps.hour, stamps.weekday, stamps.day, stamps.month], 1).astype(np.float32)
    return x, torch.tensor(arr[:n0])[None], torch.tensor(arr[n0:])[None]


def build_tiny():
    torch.manual_seed(0)
    tok = CandleTokenizer(6, 256, 8, 512, 2, 2, 0.0, 0.0, 0.0,
                          8, 8, 0.25, 1.0, 1.0, 0.5, 8).eval()
    model = CandleModel(8, 8, 6, 256, 8, 512, 0.0, 0.0, 0.0, 0.0, True).eval()
    return model, tok, 512


def build_hub(name):
    from candlecast.predictor import load_hub
    model, tok, mc = load_hub(name)
    return model, tok, mc


def bench(model, tok, mc, n0, pred_len, sample_count=1, warmup=True):
    x, xs, ys = make_data(n0, pred_len)
    ref_mod.sample_from_logits = _argmax
    engine_mod.sample_from_logits = _argmax

    if warmup:
        reference.auto_regressive_inference(tok, model, x, xs, ys, mc, pred_len,
                                            T=1.0, top_p=1.0, sample_count=sample_count)

    t0 = time.perf_counter()
    ref = reference.auto_regressive_inference(tok, model, x.clone(), xs.clone(), ys.clone(),
                                              mc, pred_len, T=1.0, top_p=1.0,
                                              sample_count=sample_count)
    t_ref = time.perf_counter() - t0
    ref = ref[:, -pred_len:]

    eng = CandleEngine(model, tok, max_context=mc)
    if warmup:
        eng.generate(x, xs, ys, pred_len, T=1.0, top_p=1.0, sample_count=sample_count)
    t0 = time.perf_counter()
    fast = eng.generate(x.clone(), xs.clone(), ys.clone(), pred_len,
                        T=1.0, top_p=1.0, sample_count=sample_count)
    t_fast = time.perf_counter() - t0

    diff = float(np.abs(fast - ref).max())
    print(f"  context={n0}  pred_len={pred_len}  samples={sample_count}  cache={mc}")
    print(f"    original : {t_ref:8.3f}s")
    print(f"    kv-cache : {t_fast:8.3f}s   speedup {t_ref / max(t_fast, 1e-9):6.1f}x   max|diff| {diff:.2e}")
    return t_ref, t_fast, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", choices=["light", "balanced", "large"], default=None)
    args = ap.parse_args()

    if args.hub:
        print(f"Loading {args.hub} model from Hugging Face...")
        model, tok, mc = build_hub(args.hub)
        for n0, pl in [(128, 32), (256, 64), (512, 128)]:
            bench(model, tok, mc, min(n0, mc), pl, sample_count=1)
    else:
        print("Synthetic mid-size model (d=256, L=6):")
        model, tok, mc = build_tiny()
        for n0, pl in [(128, 32), (256, 64), (384, 128)]:
            bench(model, tok, mc, n0, pl, sample_count=1)
        print("sample_count=8 (uncertainty paths):")
        bench(model, tok, mc, 256, 32, sample_count=8)


if __name__ == "__main__":
    main()
