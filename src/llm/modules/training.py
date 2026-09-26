"""Data batching and a bounded-logit-memory language modeling loss."""

import random

import torch
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint


def tokenize_windows(tokenizer, texts: list[str], max_length: int) -> list[torch.Tensor]:
    """Tokenize once and cover long documents with one-token-overlapping windows.

    Each next-token target is used once. Documents remain separate so fast
    memory never crosses an unrelated document boundary.
    """
    windows = []
    for start in range(0, len(texts), 128):
        encoded = tokenizer(
            texts[start : start + 128],
            truncation=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        for ids in encoded:
            windows.extend(
                torch.tensor(ids[offset : offset + max_length], dtype=torch.long)
                for offset in range(0, len(ids) - 1, max_length - 1)
            )
    return windows


def token_batches(windows, batch_size: int, pad_token_id: int, seed: int | None):
    """Shuffle and bucket similar lengths; pad on the right regardless of tokenizer settings."""
    order = list(range(len(windows)))
    rng = random.Random(seed)
    if seed is not None:
        rng.shuffle(order)
    batches = []
    bucket_size = batch_size * 32
    for start in range(0, len(order), bucket_size):
        bucket = order[start : start + bucket_size]
        if seed is not None:
            bucket.sort(key=lambda i: len(windows[i]))
        batches.extend(bucket[i : i + batch_size] for i in range(0, len(bucket), batch_size))
    if seed is not None:
        rng.shuffle(batches)
    for indices in batches:
        samples = [windows[i] for i in indices]
        ids = pad_sequence(samples, batch_first=True, padding_value=pad_token_id)
        lengths = torch.tensor([len(sample) - 1 for sample in samples])
        valid = torch.arange(ids.shape[1] - 1).unsqueeze(0) < lengths.unsqueeze(1)
        # The last input token has no next-token target; avoid processing it.
        inputs = ids[:, :-1].contiguous()
        targets = ids[:, 1:].masked_fill(~valid, -100)
        yield inputs, valid, targets, int(lengths.sum())


def chunked_lm_loss(lm_head, hidden_states, targets, chunk_size: int = 128):
    """Sum CE without retaining [batch * sequence, vocabulary] logits for backward."""
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
    targets = targets.reshape(-1)

    def project_and_loss(states, labels):
        return functional.cross_entropy(
            lm_head(states).float(), labels, ignore_index=-100, reduction="sum"
        )

    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[0], chunk_size):
        states = hidden[start : start + chunk_size]
        labels = targets[start : start + chunk_size]
        if torch.is_grad_enabled() and states.requires_grad:
            part = checkpoint(
                project_and_loss, states, labels, use_reentrant=False, preserve_rng_state=False
            )
        else:
            part = project_and_loss(states, labels)
        loss = loss + part
    return loss
