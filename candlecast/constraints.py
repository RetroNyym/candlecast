"""Post-prediction validity constraints for OHLCV paths.

The raw decoder output has no guarantees: high can sit below open, volume can
go negative, the first open can jump far from the last known close. These
projectors snap predictions back into a valid candle space.
"""
import numpy as np

EPS = 1e-8


def project_ohlc(pred, last_close=None, min_price=1e-6, continuity_weight=0.0):
    """Project an OHLCV array onto the valid candle manifold, in place-safe copy.

    Args:
        pred: array-like [T, 6+] with columns [open, high, low, close, volume, amount].
        last_close: optional float, last observed close. If given and
            continuity_weight > 0, the first open is blended toward it.
        min_price: floor for prices (keeps log/ratio math sane downstream).
        continuity_weight: 0..1 blend strength for the first open.

    Returns:
        np.ndarray with the same shape, valid OHLCV.
    """
    out = np.asarray(pred, dtype=np.float64).copy()
    o, h, l, c = out[:, 0], out[:, 1], out[:, 2], out[:, 3]

    o = np.maximum(o, min_price)
    c = np.maximum(c, min_price)

    if last_close is not None and continuity_weight > 0:
        o = o * (1 - continuity_weight) + float(last_close) * continuity_weight
        o = np.maximum(o, min_price)

    h = np.maximum.reduce([h, o, c])
    l = np.minimum.reduce([l, o, c])
    h = np.maximum(h, l)

    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = o, h, l, c
    out[:, 4] = np.maximum(out[:, 4], 0.0)  # volume
    out[:, 5] = np.maximum(out[:, 5], 0.0)  # amount
    return out


def project_returns(pred, last_close, min_price=1e-6):
    """Rescale a path so prices stay positive relative to the last close.

    If the decoder produced values that are not anchored to the real price
    level (common when mean/std normalization is applied over a short window),
    this softly re-anchors the first open to the previous close while keeping
    the predicted shape (returns) intact.
    """
    out = np.asarray(pred, dtype=np.float64).copy()
    gap = out[0, 0] / max(float(last_close), min_price)
    if gap <= 0 or not np.isfinite(gap) or abs(np.log(gap)) > 1.0:
        # Wild jump: re-anchor, preserving relative moves inside the path.
        shape = out / np.maximum(out[0, 0], min_price)
        out = shape * float(last_close)
    return out


def sanitize(pred, last_close=None, continuity_weight=0.0):
    """Full pipeline: optional re-anchor + OHLC projection. Returns [T, 6]."""
    out = project_returns(pred, last_close if last_close is not None else pred[0, 0])
    return project_ohlc(out, last_close=last_close, continuity_weight=continuity_weight)
