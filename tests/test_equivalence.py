import numpy as np
import pandas as pd
import pytest
import torch

from candlecast import reference
from candlecast.backbone import CandleModel, CandleTokenizer
from candlecast.engine import CandleEngine
from candlecast import engine as engine_mod

SEED = 123
FEATS = 6


def _argmax_sampler(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    return torch.topk(logits, k=1, dim=-1).indices


@pytest.fixture(autouse=True)
def _deterministic(monkeypatch):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    monkeypatch.setattr(reference, "sample_from_logits", _argmax_sampler)
    monkeypatch.setattr(engine_mod, "sample_from_logits", _argmax_sampler)


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(SEED)
    tokenizer = CandleTokenizer(
        d_in=FEATS, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        s1_bits=4, s2_bits=4, beta=0.25, gamma0=1.0, gamma=1.0, zeta=0.5, group_size=4,
    ).eval()
    model = CandleModel(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    ).eval()
    return model, tokenizer


def _make_case(n0, pred_len, seed=7):
    g = torch.Generator().manual_seed(seed)
    close = 100 + torch.cumsum(torch.randn(n0 + pred_len, generator=g) * 0.5, 0)
    o = close + torch.randn(n0 + pred_len, generator=g) * 0.1
    h = torch.maximum(o, close) + 0.2
    l = torch.minimum(o, close) - 0.2
    vol = torch.rand(n0 + pred_len, generator=g) * 1000 + 100
    amt = vol * close
    x = torch.stack([o, h, l, close, vol, amt], dim=-1)[:n0]

    base = pd.Timestamp("2025-01-06 09:00:00")
    stamps = pd.date_range(base, periods=n0 + pred_len, freq="5min")
    x_stamp = torch.tensor(
        np.stack([stamps.minute, stamps.hour, stamps.weekday, stamps.day, stamps.month], 1)[:n0],
        dtype=torch.float32)[None]
    y_stamp = torch.tensor(
        np.stack([stamps.minute, stamps.hour, stamps.weekday, stamps.day, stamps.month], 1)[n0:],
        dtype=torch.float32)[None]
    return x[None].float(), x_stamp, y_stamp


@pytest.mark.parametrize("n0,pred_len,mc", [
    (32, 16, 64),   # window never fills
    (64, 8, 64),    # warmup exactly fills the cache
    (80, 12, 64),   # warmup truncated to max_context
    (48, 32, 64),   # sliding starts mid-generation
])
def test_fast_matches_reference(pair, n0, pred_len, mc):
    model, tokenizer = pair
    x, x_stamp, y_stamp = _make_case(n0, pred_len)

    ref = reference.auto_regressive_inference(
        tokenizer, model, x.clone(), x_stamp.clone(), y_stamp.clone(),
        max_context=mc, pred_len=pred_len, clip=5, T=1.0, top_k=0, top_p=1.0,
        sample_count=1, verbose=False)[:, -pred_len:]

    eng = CandleEngine(model, tokenizer, max_context=mc, clip=5.0)
    fast = eng.generate(x.clone(), x_stamp.clone(), y_stamp.clone(), pred_len,
                        T=1.0, top_k=0, top_p=1.0, sample_count=1, verbose=False)

    assert fast.shape == ref.shape
    assert np.allclose(fast, ref, atol=2e-4), (
        f"max diff {np.abs(fast - ref).max():.2e} (n0={n0}, pred_len={pred_len}, mc={mc})")


def test_sample_count_batching(pair):
    model, tokenizer = pair
    x, x_stamp, y_stamp = _make_case(24, 6)
    ref = reference.auto_regressive_inference(
        tokenizer, model, x.clone(), x_stamp.clone(), y_stamp.clone(),
        max_context=32, pred_len=6, T=1.0, top_k=0, top_p=1.0, sample_count=3)[:, -6:]
    eng = CandleEngine(model, tokenizer, max_context=32)
    fast = eng.generate(x.clone(), x_stamp.clone(), y_stamp.clone(), 6,
                        T=1.0, top_k=0, top_p=1.0, sample_count=3)
    assert np.allclose(fast, ref, atol=2e-4)


def test_keep_samples_shape(pair):
    model, tokenizer = pair
    x, x_stamp, y_stamp = _make_case(16, 4)
    eng = CandleEngine(model, tokenizer, max_context=32)
    out = eng.generate(x, x_stamp, y_stamp, 4, sample_count=5, keep_samples=True)
    assert out.shape == (1, 5, 4, FEATS)
