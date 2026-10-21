"""Robust top-k / nucleus sampling utilities (fixes the None-crash of the original)."""
import torch
import torch.nn.functional as F


def top_k_top_p_filtering(logits, top_k: int = 0, top_p: float = 1.0,
                          filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
    if top_k and top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    """Sample (or argmax) from logits with optional top-k / top-p filtering.

    Unlike the upstream implementation this tolerates None for top_k/top_p
    (previously `top_k=None` raised TypeError inside the filter condition).
    """
    if temperature is None:
        temperature = 1.0
    if temperature <= 0:
        sample_logits = False
        temperature = 1.0

    logits = logits / temperature

    tk = top_k or 0
    tp = 1.0 if top_p is None else top_p
    if tk > 0 or tp < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=tk, top_p=tp)

    probs = F.softmax(logits, dim=-1)
    if not sample_logits:
        return torch.topk(probs, k=1, dim=-1).indices
    return torch.multinomial(probs, num_samples=1)
