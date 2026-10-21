"""High-level predictor: normalization + fast engine + validity constraints."""
import numpy as np
import pandas as pd
import torch

from .constraints import sanitize
from .engine import CandleEngine

TIME_COLS = ["minute", "hour", "weekday", "day", "month"]
PRICE_COLS = ["open", "high", "low", "close"]

MODEL_ZOO = {
    "light":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k", 2048),
    "balanced": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base", 512),
    "large":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base", 512),
}


def _calc_time_stamps(ts: pd.Series) -> np.ndarray:
    ts = pd.DatetimeIndex(ts)
    return np.stack([ts.minute, ts.hour, ts.weekday, ts.day, ts.month], axis=1).astype(np.float32)


def _pick_device(device=None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hub(name_or_pair, device=None):
    """Load (model, tokenizer, max_context) for a zoo name or explicit (model, tok) pair."""
    from .backbone import CandleModel, CandleTokenizer

    if isinstance(name_or_pair, (tuple, list)) and len(name_or_pair) == 2:
        model_id, tok_id = name_or_pair
        mc = 512
    else:
        key = str(name_or_pair).lower()
        if key not in MODEL_ZOO:
            raise ValueError(f"Unknown model {name_or_pair!r}. Options: {list(MODEL_ZOO)}")
        model_id, tok_id, mc = MODEL_ZOO[key]

    tokenizer = CandleTokenizer.from_pretrained(tok_id)
    model = CandleModel.from_pretrained(model_id)
    dev = _pick_device(device)
    model, tokenizer = model.to(dev).eval(), tokenizer.to(dev)
    return model, tokenizer, mc


class CandlePredictor:
    """High-level predictor: normalization + KV-cache engine + validity constraints.

    Extra kwargs vs upstream:
        autocast: None | "bf16" | "fp16" (mixed precision on GPU/CPU-bf16)
        project: snap outputs to valid OHLCV (high>=max(o,c), low<=min(o,c), vol>=0)
        continuity: blend the first predicted open toward the last observed close
        return_samples: keep every sample path (for quantiles / uncertainty)
    """

    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5.0,
                 autocast=None, project=True, continuity=0.0):
        self.device = _pick_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device)
        self.engine = CandleEngine(self.model, self.tokenizer,
                                   max_context=max_context, clip=clip, autocast=autocast)
        self.max_context = max_context
        self.clip = clip
        self.project = project
        self.continuity = continuity
        self.price_cols = PRICE_COLS
        self.vol_col = "volume"
        self.amt_col = "amount"

    @classmethod
    def from_hub(cls, name="balanced", device=None, **kwargs):
        model, tokenizer, mc = load_hub(name, device=device)
        kwargs.setdefault("max_context", min(mc, 512))
        return cls(model, tokenizer, device=device, **kwargs)

    # ------------------------------------------------------------------ utils
    def _prepare_df(self, df, i=None):
        tag = f" at index {i}" if i is not None else ""
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Input{tag} must be a pandas DataFrame.")
        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found{tag}.")
        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_col] = 0.0
        if self.amt_col not in df.columns:
            df[self.amt_col] = df[self.vol_col] * df[self.price_cols].mean(axis=1)
        if df[self.price_cols + [self.vol_col, self.amt_col]].isnull().values.any():
            raise ValueError(f"DataFrame{tag} contains NaN values in price/volume columns.")
        return df

    def _norm(self, df, x_timestamp, y_timestamp):
        x_ts = pd.DatetimeIndex(x_timestamp)
        y_ts = pd.DatetimeIndex(y_timestamp)
        x = df[self.price_cols + [self.vol_col, self.amt_col]].values.astype(np.float32)
        x_stamp = _calc_time_stamps(x_ts)
        y_stamp = _calc_time_stamps(y_ts)
        if x.shape[0] != x_stamp.shape[0]:
            raise ValueError(f"Inconsistent lengths: x has {x.shape[0]} vs stamps {x_stamp.shape[0]}.")
        mean, std = x.mean(axis=0), x.std(axis=0)
        x_norm = np.clip((x - mean) / (std + 1e-5), -self.clip, self.clip)
        return x_norm, x_stamp, y_stamp, mean, std, x_ts, y_ts

    def _render(self, preds, mean, std, y_ts, x_ts, last_close):
        cols = self.price_cols + [self.vol_col, self.amt_col]
        if preds.ndim == 4:  # [B=1, S, T, F]
            out = []
            for j in range(preds.shape[1]):
                p = preds[0, j] * (std + 1e-5) + mean
                if self.project:
                    p = sanitize(p, last_close=last_close, continuity_weight=self.continuity)
                out.append(p)
            return np.stack(out, axis=0)  # [S, T, F]
        p = preds[0] * (std + 1e-5) + mean
        if self.project:
            p = sanitize(p, last_close=last_close, continuity_weight=self.continuity)
        return p  # [T, F]

    # ---------------------------------------------------------------- predict
    def predict(self, df, x_timestamp, y_timestamp, pred_len,
                T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False,
                return_samples=False):
        df = self._prepare_df(df)
        x_norm, x_stamp, y_stamp, mean, std, x_ts, y_ts = self._norm(df, x_timestamp, y_timestamp)
        if len(y_ts) != pred_len:
            raise ValueError(f"y_timestamp length {len(y_ts)} != pred_len {pred_len}.")
        last_close = float(df[self.price_cols].iloc[-1, 3])

        x = torch.from_numpy(x_norm[None]).to(self.device)
        xs = torch.from_numpy(x_stamp[None]).to(self.device)
        ys = torch.from_numpy(y_stamp[None]).to(self.device)

        preds = self.engine.generate(x, xs, ys, pred_len, T=T, top_k=top_k, top_p=top_p,
                                     sample_count=sample_count, verbose=verbose,
                                     keep_samples=return_samples)
        preds = preds[:, -pred_len:]
        rendered = self._render(preds, mean, std, y_ts, x_ts, last_close)
        cols = self.price_cols + [self.vol_col, self.amt_col]

        if return_samples:
            # rendered: [S, T, F] -> per-step quantiles across sample paths
            flat = rendered
            s = flat.shape[0]
            return {
                "mean": pd.DataFrame(flat.mean(axis=0), columns=cols, index=y_ts),
                "q10": pd.DataFrame(np.quantile(flat, 0.1, axis=0), columns=cols, index=y_ts),
                "q50": pd.DataFrame(np.quantile(flat, 0.5, axis=0), columns=cols, index=y_ts),
                "q90": pd.DataFrame(np.quantile(flat, 0.9, axis=0), columns=cols, index=y_ts),
                "samples": flat,
            }
        return pd.DataFrame(rendered.squeeze(0), columns=cols, index=y_ts)

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len,
                      T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=False):
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list / x_timestamp_list / y_timestamp_list length mismatch.")
        n = len(df_list)
        x_list, xs_list, ys_list, means, stds, closes = [], [], [], [], [], []
        for i in range(n):
            df = self._prepare_df(df_list[i], i=i)
            x_norm, x_stamp, y_stamp, mean, std, x_ts, y_ts = self._norm(
                df, x_timestamp_list[i], y_timestamp_list[i])
            if len(y_ts) != pred_len:
                raise ValueError(f"y_timestamp length {len(y_ts)} != pred_len {pred_len} (index {i}).")
            x_list.append(x_norm)
            xs_list.append(x_stamp)
            ys_list.append(y_stamp)
            means.append(mean)
            stds.append(std)
            closes.append(float(df[self.price_cols].iloc[-1, 3]))
        seq_lens = {a.shape[0] for a in x_list}
        if len(seq_lens) != 1:
            raise ValueError(f"Batch requires equal lookback lengths, got {sorted(seq_lens)}.")

        x = torch.from_numpy(np.stack(x_list)).to(self.device)
        xs = torch.from_numpy(np.stack(xs_list)).to(self.device)
        ys = torch.from_numpy(np.stack(ys_list)).to(self.device)
        preds = self.engine.generate(x, xs, ys, pred_len, T=T, top_k=top_k, top_p=top_p,
                                     sample_count=sample_count, verbose=verbose)
        preds = preds[:, -pred_len:]  # [n, T, F]

        cols = self.price_cols + [self.vol_col, self.amt_col]
        out = []
        for i in range(n):
            p = preds[i] * (stds[i] + 1e-5) + means[i]
            if self.project:
                p = sanitize(p, last_close=closes[i], continuity_weight=self.continuity)
            out.append(pd.DataFrame(p, columns=cols, index=pd.DatetimeIndex(y_timestamp_list[i])))
        return out
