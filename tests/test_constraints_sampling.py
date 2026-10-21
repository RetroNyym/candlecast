import numpy as np
import pytest
import torch

from candlecast.constraints import project_ohlc, project_returns, sanitize
from candlecast.sampling import sample_from_logits, top_k_top_p_filtering


def test_project_ohlc_validates_path():
    pred = np.array([
        [10.0, 9.0, 11.0, 12.0, -5.0, -1.0],   # high below open/close, vol negative
        [12.0, 13.0, 11.5, 12.5, 100.0, 1250.0],
    ])
    out = project_ohlc(pred)
    o, h, l, c, v, a = out[0]
    assert h >= max(o, c) and l <= min(o, c) and h >= l
    assert v >= 0 and a >= 0


def test_project_continuity_blends_first_open():
    pred = np.array([[10.0, 10.5, 9.5, 10.2, 1.0, 10.0]] * 2)
    out = project_ohlc(pred, last_close=12.0, continuity_weight=0.5)
    assert abs(out[0, 0] - 11.0) < 1e-9


def test_project_returns_reanchors_wild_jump():
    pred = np.array([[500.0, 501.0, 499.0, 500.5, 1.0, 500.0]] * 3)
    out = project_returns(pred, last_close=100.0)
    assert abs(out[0, 0] - 100.0) < 1e-6


def test_sanitize_full_pipeline():
    pred = np.array([[100.0, 101.0, 99.0, 100.5, 10.0, 1000.0]] * 5)
    out = sanitize(pred, last_close=100.0)
    assert (out[:, 1] >= out[:, 3]).all()
    assert (out[:, 2] <= out[:, 3]).all()


def test_sampling_none_kwargs_do_not_crash():
    logits = torch.randn(2, 16)
    a = sample_from_logits(logits.clone(), top_k=None, top_p=0.9, sample_logits=False)
    b = sample_from_logits(logits.clone(), top_k=0, top_p=None, sample_logits=False)
    assert a.shape == (2, 1) and b.shape == (2, 1)


def test_sampling_temperature_zero_is_greedy():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    out = sample_from_logits(logits, temperature=0.0)
    assert out.item() == 1


def test_top_p_filters_tiny_mass():
    logits = torch.tensor([[10.0, 0.0, -10.0]])
    filtered = top_k_top_p_filtering(logits.clone(), top_k=0, top_p=0.99)
    assert filtered[0, 0] > -1e30
    assert filtered[0, 2] == -float("inf")
