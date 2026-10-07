"""
inverted_index.py

Builds and queries token -> positions inverted indexes.

Critical design choice: postings store NUMPY INTEGER POSITIONS (0..n-1
within a country partition), not entity_id strings. Storing millions of
Python string objects, duplicated across every token's posting list
they appear in, carries significant per-object memory overhead at this
scale. Integer positions in numpy int32 arrays are dramatically
cheaper, and entity_id strings are looked up only ONCE at the very end,
via a single numpy array indexing operation (ids_array[positions]) --
see blocking_pipeline.py.

Index-building is batched (explode + groupby PER BATCH, merged into a
running accumulator) rather than exploding the full multi-million-row
column in one shot, the same memory-bounding pattern used for token
frequency counting during the EDA phase.
"""

import gc
import time

import numpy as np
import pandas as pd


def build_inverted_index_batched(
    text_series: pd.Series,
    tokenizer_fn,
    batch_size: int,
    max_df_fraction: float,
    max_df_absolute: int,
    exclude_tokens: set,
    label: str,
) -> dict:
    """
    Builds a token -> np.ndarray[int32] positions index.

    tokenizer_fn: a Series -> Series-of-lists function (either
    tokenize_words_batch or tokenize_ngrams_batch from tokenizers.py).

    exclude_tokens: a stopword-style set (NAME_STOPWORDS or
    ADDRESS_STOPWORDS from config.py) removed regardless of measured
    document frequency, since these are already known to be
    near-universal from prior analysis.

    Returns only tokens that survive BOTH the exclude-set check and the
    document-frequency cap -- a token exceeding max_df_absolute or
    max_df_fraction * n (whichever is smaller) is dropped as too common
    to be a useful blocking signal.
    """
    n = len(text_series)
    if n == 0:
        return {}

    postings = {}  # token -> list of np arrays, one array per contributing batch
    n_batches = (n // batch_size) + 1
    print(f"    building {label} index: {n:,} rows in {n_batches} batch(es)")
    start_time = time.time()

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        batch_text = text_series.iloc[batch_start:batch_end]

        batch_tokens = tokenizer_fn(batch_text)

        batch_df = pd.DataFrame({
            "pos": np.arange(batch_start, batch_end, dtype=np.int32),
            "token": batch_tokens.values,
        })
        batch_df = batch_df.explode("token").dropna(subset=["token"])

        # Groupby is vectorized/C-level pandas -- the only Python-level
        # loop here is over DISTINCT tokens in this batch (typically
        # tens of thousands), not over exploded rows (hundreds of
        # thousands to millions), which is what keeps this fast.
        batch_grouped = batch_df.groupby("token")["pos"].apply(
            lambda s: s.to_numpy(dtype=np.int32)
        )
        for tok, arr in batch_grouped.items():
            if tok in exclude_tokens:
                continue
            postings.setdefault(tok, []).append(arr)

        # Frees this batch's intermediate data before the next batch
        # begins, so peak memory stays bounded to roughly one batch's
        # worth rather than accumulating across the whole column.
        del batch_text, batch_tokens, batch_df, batch_grouped
        gc.collect()

        elapsed = time.time() - start_time
        print(f"      batch {batch_start:,}-{batch_end:,} done ({elapsed:.1f}s elapsed)")

    max_df = min(max_df_absolute, int(max_df_fraction * n)) if n > 0 else max_df_absolute

    final_index = {}
    n_dropped_df = 0
    for tok, arr_list in postings.items():
        full_arr = arr_list[0] if len(arr_list) == 1 else np.concatenate(arr_list)
        if len(full_arr) > max_df:
            n_dropped_df += 1
            continue
        final_index[tok] = full_arr

    del postings
    gc.collect()

    print(f"    {label} index: {len(final_index):,} tokens kept, "
          f"{n_dropped_df:,} dropped for exceeding max_df={max_df:,}")
    return final_index


def get_top_k_rarest(tokens: list, index: dict, k: int) -> list:
    """
    Given an entity's token list and an index, returns the K tokens
    (among those actually present in the index) with the smallest
    posting lists -- the most discriminative available tokens. Tokens
    not present in the index (either genuinely absent from any S2/S3
    record, or filtered out as a stopword / too-common) are simply
    skipped, not treated as an error.
    """
    available = [(t, len(index[t])) for t in tokens if t in index]
    available.sort(key=lambda pair: pair[1])
    return [t for t, _ in available[:k]]


def query_positions(tokens: list, index: dict) -> np.ndarray:
    """
    Concatenates the posting arrays for the given tokens. Duplicates
    ARE preserved intentionally -- they are resolved later via
    np.unique(..., return_counts=True) in blocking_pipeline.py, where
    the count becomes a cheap "how many signals hit this candidate"
    strength score used for ranking during the optional trim step,
    never for final precision (that is the matching model's job in the
    next phase).
    """
    if not tokens:
        return np.array([], dtype=np.int32)
    arrays = [index[t] for t in tokens]
    return np.concatenate(arrays) if len(arrays) > 1 else arrays[0]