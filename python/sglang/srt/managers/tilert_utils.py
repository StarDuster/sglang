from __future__ import annotations


def cached_len_after_external_prefill_tail(
    cached_len: int,
    prompt_len: int,
    tail_tokens: int,
    *,
    with_mtp: bool,
    mtp_seq_len: int,
) -> int:
    if tail_tokens <= 0 or cached_len <= 1:
        return cached_len

    target_cached_len = max(1, cached_len - tail_tokens)
    if with_mtp:
        mtp_seq_len = max(1, mtp_seq_len)
        target_cached_len = max(1, (target_cached_len // mtp_seq_len) * mtp_seq_len)

    return min(target_cached_len, prompt_len)
