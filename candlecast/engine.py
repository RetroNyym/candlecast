"""KV-cache autoregressive engine for the candlestick forecast model.

The upstream loop re-runs the full transformer over the whole context window at
every generation step (O(pred_len * context) work). This engine keeps a
per-layer KV cache plus a hidden-state cache for the dependency layer, so each
step only processes ONE token (O(pred_len) projections + O(context) attention).

RoPE note: upstream re-indexes the sliding window to positions 0..L-1 each step.
RoPE attention depends only on relative distance, which is preserved by that
re-indexing, so absolute positions + KV eviction produce identical attention
scores (verified by tests/test_equivalence.py).
"""
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import trange

from .sampling import sample_from_logits


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, inv_freq, positions):
    """x: [B, H, L, head_dim]; positions: [L] long tensor of absolute positions."""
    t = positions.to(dtype=inv_freq.dtype, device=x.device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class LayerCache:
    __slots__ = ("k", "v", "max_len")

    def __init__(self, max_len, batch, n_heads, head_dim, device, dtype):
        self.max_len = max_len
        self.k = torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype)

    def shift(self):
        self.k[..., :-1, :].copy_(self.k[..., 1:, :].clone())
        self.v[..., :-1, :].copy_(self.v[..., 1:, :].clone())


def _layer_step(layer, x, cache, write_start, rope_positions):
    """Run one TransformerBlock over x ([B, L, D]) with KV cache bookkeeping."""
    attn = layer.self_attn
    B, L, D = x.shape
    n_heads, head_dim = attn.n_heads, attn.head_dim

    residual = x
    x = layer.norm1(x)
    q = attn.q_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(B, L, n_heads, head_dim).transpose(1, 2)

    q = apply_rope(q, attn.rotary.inv_freq, rope_positions)
    k = apply_rope(k, attn.rotary.inv_freq, rope_positions)

    end = write_start + L
    cache.k[:, :, write_start:end] = k
    cache.v[:, :, write_start:end] = v

    out = torch.nn.functional.scaled_dot_product_attention(
        q, cache.k[:, :, :end], cache.v[:, :, :end],
        attn_mask=None, dropout_p=0.0, is_causal=(L > 1),
    )
    out = out.transpose(1, 2).contiguous().view(B, L, D)
    x = residual + attn.resid_dropout(attn.out_proj(out))

    residual = x
    x = layer.norm2(x)
    return residual + layer.ffn(x)


class CandleEngine:
    """Fast inference engine wrapping a (tokenizer, backbone model) pair."""

    def __init__(self, model, tokenizer, max_context=512, clip=5.0, autocast=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context = int(max_context)
        self.clip = clip
        # autocast: None | "bf16" | "fp16"
        self.autocast = autocast
        if next(model.parameters()).device.type == "cpu" and autocast == "fp16":
            self.autocast = None  # fp16 autocast is not supported on CPU

    def _autocast_ctx(self, device_type):
        if self.autocast is None:
            return nullcontext()
        dtype = torch.bfloat16 if self.autocast == "bf16" else torch.float16
        return torch.autocast(device_type=device_type, dtype=dtype)

    @torch.inference_mode()
    def generate(self, x, x_stamp, y_stamp, pred_len, T=1.0, top_k=0, top_p=0.9,
                 sample_count=1, verbose=False, keep_samples=False):
        """Autoregressive generation with KV cache.

        Args mirror the upstream `auto_regressive_inference` and return the same
        shape ([B, pred_len, F], mean over sample paths). With keep_samples=True
        returns ([B, sample_count, pred_len, F]) instead.
        """
        model, tok = self.model, self.tokenizer
        model.eval()
        mc = self.max_context
        device_type = next(model.parameters()).device.type

        x = torch.clip(x, -self.clip, self.clip)
        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        n0 = x.size(1)
        total = n0 + pred_len
        B = x.size(0)
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        param = next(model.parameters())
        dtype = param.dtype

        with self._autocast_ctx(device_type):
            pre_ids, post_ids = tok.encode(x, half=True)
            w0 = min(n0, mc)
            start = n0 - w0

            caches = [
                LayerCache(mc, B, model.n_heads, model.d_model // model.n_heads,
                           device, dtype)
                for _ in range(model.n_layers)
            ]
            h_cache = torch.zeros(B, mc, model.d_model, device=device, dtype=dtype)

            # Warmup: full forward over the initial window (positions 0..w0-1).
            h = model.embedding([pre_ids[:, start:], post_ids[:, start:]])
            h = h + model.time_emb(full_stamp[:, start:n0, :])
            rope_pos = torch.arange(w0, device=device)
            for li, layer in enumerate(model.transformer):
                h = _layer_step(layer, h, caches[li], write_start=0, rope_positions=rope_pos)
            h = model.norm(h)
            h_cache[:, :w0] = h  # normed hiddens for the dependency layer

            gen_pre = pre_ids.new_empty(B, pred_len)
            gen_post = post_ids.new_empty(B, pred_len)

            last_h = h[:, -1]  # [B, D] hidden of the newest cached position
            cur_len = w0

            ran = trange if verbose else range
            for i in ran(pred_len):
                s1_logits = model.head(last_h)  # [B, V1]
                sample_pre = sample_from_logits(
                    s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

                context = h_cache[:, :cur_len, :]
                s2_logits = model.decode_s2(context, sample_pre)[:, -1, :]
                sample_post = sample_from_logits(
                    s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

                gen_pre[:, i] = sample_pre.squeeze(-1)
                gen_post[:, i] = sample_post.squeeze(-1)

                if i == pred_len - 1:
                    break

                # Feed the newly sampled token as the next input position.
                abs_pos = n0 + i
                emb = model.embedding([sample_pre, sample_post])
                emb = emb + model.time_emb(full_stamp[:, abs_pos:abs_pos + 1, :])

                if cur_len >= mc:
                    for c in caches:
                        c.shift()
                    h_cache[:, :-1, :].copy_(h_cache[:, 1:, :].clone())
                    write_start = mc - 1
                else:
                    write_start = cur_len
                    cur_len += 1

                rp = torch.tensor([w0 + i], device=device)
                last_h = emb
                for li, layer in enumerate(model.transformer):
                    last_h = _layer_step(layer, last_h, caches[li],
                                         write_start=write_start, rope_positions=rp)
                last_h = model.norm(last_h)
                h_cache[:, write_start] = last_h[:, 0]
                last_h = last_h[:, 0]  # [B, D] for the next head() call

            full_pre = torch.cat([pre_ids, gen_pre], dim=1)
            full_post = torch.cat([post_ids, gen_post], dim=1)
            ctx_start = max(0, total - mc)
            z = tok.decode(
                [full_pre[:, ctx_start:total].contiguous(),
                 full_post[:, ctx_start:total].contiguous()],
                half=True,
            )

        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        z = z[:, :, -pred_len:, :]  # keep only the forecast horizon
        if keep_samples:
            return z.cpu().numpy()
        return np.mean(z.cpu().numpy(), axis=1)
