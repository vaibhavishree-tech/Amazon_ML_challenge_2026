"""
blocking_pipeline.py

Core orchestration: build per-country indexes from Source 2/3, then
generate candidates for every Source 1 entity in that country.

Country partitioning is mandatory, not optional: ground-truth topology
analysis during the EDA phase confirmed 0.00% of true-match pairs
disagree on country across the full training set, so restricting all
comparison work to within-country pairs loses zero recall while cutting
the search space several-fold.

Crash-safety design: each country's output is written to a .tmp file
during processing and renamed to its final name only after that
country completes without error. On a re-run, any country whose final
(non-.tmp) file already exists is skipped entirely. This bounds the
cost of an interrupted run to at most one country's worth of
reprocessing rather than the whole run, the same incremental-safety
principle used for per-file processing in the cleaning phase.
"""

import gc
import os
import time

import numpy as np
import pandas as pd

from config import (
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE,
    NAME_STOPWORDS, ADDRESS_STOPWORDS,
    MAX_DF_FRACTION, MAX_DF_ABSOLUTE, RARE_K, MAX_CANDIDATES_PER_ENTITY,
    INDEX_BUILD_BATCH_SIZE, QUERY_PROGRESS_BATCH_SIZE,
)
from tokenizers import (
    tokenize_words_batch, tokenize_ngrams_batch,
    tokenize_words_single, generate_char_ngrams,
)
from inverted_index import build_inverted_index_batched, get_top_k_rarest, query_positions


def build_country_indexes(s2s3_country_df: pd.DataFrame) -> dict:
    """
    Builds all four blocking signals for one country's combined
    Source 2 + Source 3 records:
      Signal A - name tokens (word-level inverted index)
      Signal B - name character n-grams (typo/transliteration-tolerant)
      Signal C - address tokens (word-level inverted index)
      Signal D - postal code exact match
    Returns a dict bundling the indexes plus the positional entity_id
    lookup array needed to translate index positions back to real IDs.
    """
    s2s3_country_df = s2s3_country_df.reset_index(drop=True)
    n = len(s2s3_country_df)
    ids = s2s3_country_df[COL_ENTITY_ID].to_numpy()

    name_token_index = build_inverted_index_batched(
        s2s3_country_df[COL_NAME], tokenize_words_batch,
        INDEX_BUILD_BATCH_SIZE, MAX_DF_FRACTION, MAX_DF_ABSOLUTE,
        NAME_STOPWORDS, label="name-token",
    )
    name_ngram_index = build_inverted_index_batched(
        s2s3_country_df[COL_NAME], tokenize_ngrams_batch,
        INDEX_BUILD_BATCH_SIZE, MAX_DF_FRACTION, MAX_DF_ABSOLUTE,
        exclude_tokens=set(), label="name-4gram",  # no stopword concept for n-grams; DF filtering alone handles commonality
    )
    address_token_index = build_inverted_index_batched(
        s2s3_country_df[COL_ADDRESS], tokenize_words_batch,
        INDEX_BUILD_BATCH_SIZE, MAX_DF_FRACTION, MAX_DF_ABSOLUTE,
        ADDRESS_STOPWORDS, label="address-token",
    )

    # Postal code index: simple exact-match dict, no DF filtering
    # needed at this scale since postal codes are sparse (present in
    # only a small minority of addresses per the EDA phase's keyword
    # frequency analysis) and each code covers a small geographic area,
    # unlikely to produce a posting list as large as a common word token
    # would.
    postal_index = {}
    postal_series = s2s3_country_df[COL_POSTAL_CODE]
    non_empty_mask = postal_series.str.len() > 0
    postal_df = pd.DataFrame({
        "pos": np.arange(n, dtype=np.int32)[non_empty_mask.to_numpy()],
        "code": postal_series[non_empty_mask].to_numpy(),
    })
    for code, group in postal_df.groupby("code")["pos"]:
        postal_index[code] = group.to_numpy(dtype=np.int32)
    del postal_df
    gc.collect()

    return {
        "ids": ids,
        "n": n,
        "name_token_index": name_token_index,
        "name_ngram_index": name_ngram_index,
        "address_token_index": address_token_index,
        "postal_index": postal_index,
    }


def generate_candidates_for_entity(name: str, address: str, postal_code: str, indexes: dict, k: int):
    """
    Computes candidate positions for ONE Source 1 entity by querying
    its rarest available tokens/n-grams against all four blocking
    signals and combining the results.

    Scoring design, two deliberate choices:

    1. RARITY-WEIGHTED SCORING instead of plain hit-count. Each
       token's contribution is weighted by 1/posting_list_size, so a
       token appearing in only a handful of records contributes far
       more "evidence" of a genuine match than one appearing in
       thousands of records. This makes the resulting score a more
       meaningful ranking signal than a simple count of how many
       signals fired.

    2. NO HARD EXCLUSION FLOOR. An earlier design required a candidate
       to be hit by at least two independent rare-token signals before
       being kept at all, reasoning that true matches typically share
       multiple tokens. In practice this discarded a meaningful number
       of true matches whose only surviving signal was a single rare
       token, turning recoverable candidates into permanent misses.
       The current design keeps every candidate the rarity-weighted
       index surfaces and lets trim_candidates() rank by weighted score
       when a cap is needed -- a weak candidate that reaches the
       matching model can still be correctly rejected there, whereas a
       candidate dropped here is unrecoverable by any later stage.
    """
    name_tokens = tokenize_words_single(name)
    name_ngrams = generate_char_ngrams(name)
    addr_tokens = tokenize_words_single(address)

    # Each entry: (positions_array, weight_per_hit)
    weighted_parts = []

    top_name_tokens = get_top_k_rarest(name_tokens, indexes["name_token_index"], k)
    for tok in top_name_tokens:
        positions = indexes["name_token_index"][tok]
        weight = 1.0 / len(positions)
        weighted_parts.append((positions, weight))

    top_name_ngrams = get_top_k_rarest(name_ngrams, indexes["name_ngram_index"], k)
    for tok in top_name_ngrams:
        positions = indexes["name_ngram_index"][tok]
        weight = 1.0 / len(positions)
        weighted_parts.append((positions, weight))

    top_addr_tokens = get_top_k_rarest(addr_tokens, indexes["address_token_index"], k)
    for tok in top_addr_tokens:
        positions = indexes["address_token_index"][tok]
        weight = 1.0 / len(positions)
        weighted_parts.append((positions, weight))

    if postal_code and postal_code in indexes["postal_index"]:
        positions = indexes["postal_index"][postal_code]
        # Postal code match is treated as strong, near-certain evidence
        # on its own -- given a fixed high weight rather than
        # 1/len(positions), so a postal-code hit alone contributes
        # meaningfully to a candidate's score without needing to rely
        # on a weak token-based signal as well.
        weighted_parts.append((positions, 2.0))

    if not weighted_parts:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float64)

    # Accumulate per-position: total hit count and total weighted score
    # (the latter is what ranking/trimming actually uses).
    all_positions = np.concatenate([p for p, _ in weighted_parts])
    all_weights = np.concatenate([
        np.full(len(p), w, dtype=np.float64) for p, w in weighted_parts
    ])

    unique_positions, inverse = np.unique(all_positions, return_inverse=True)
    hit_counts = np.bincount(inverse, minlength=len(unique_positions))
    weighted_scores = np.bincount(inverse, weights=all_weights, minlength=len(unique_positions))

    # All candidates surfaced by any signal are returned here, with
    # their weighted score attached -- see the function docstring above
    # for why no candidates are excluded at this stage. Ranking and any
    # necessary cap enforcement happens downstream, in trim_candidates().
    return unique_positions, weighted_scores


def trim_candidates(positions: np.ndarray, scores: np.ndarray, max_candidates: int):
    """
    Applies the soft cap only when needed. Ranks by the rarity-weighted
    score from generate_candidates_for_entity rather than plain hit
    count, so a candidate backed by several genuinely rare tokens
    outranks one backed by several common, barely-filtered-through
    tokens.
    """
    if len(positions) <= max_candidates:
        return positions
    order = np.argsort(-scores)[:max_candidates]
    return positions[order]


def process_country(
    country_label: str,
    s1_country_df: pd.DataFrame,
    s2s3_country_df: pd.DataFrame,
    output_dir,
    k: int = RARE_K,
    max_candidates: int = MAX_CANDIDATES_PER_ENTITY,
    query_progress_batch_size: int = QUERY_PROGRESS_BATCH_SIZE,
):
    """
    Full pipeline for one country: build indexes, generate candidates
    for every Source 1 entity in that country, write results
    incrementally. Returns the path to the finished (non-.tmp) output
    file for this country.
    """
    final_path = output_dir / f"candidates_{country_label}.tsv"
    if final_path.exists():
        print(f"  [{country_label}] already completed (found {final_path.name}) -- skipping.")
        return final_path

    tmp_path = output_dir / f"candidates_{country_label}.tsv.tmp"
    if tmp_path.exists():
        # Leftover from a previous interrupted run mid-country --
        # discard and restart this country cleanly, since a partial
        # .tmp file cannot be trusted to be complete or well-formed.
        os.remove(tmp_path)

    print(f"  [{country_label}] building indexes from {len(s2s3_country_df):,} Source 2+3 records...")
    indexes = build_country_indexes(s2s3_country_df)

    s1_country_df = s1_country_df.reset_index(drop=True)
    n = len(s1_country_df)
    name_list = s1_country_df[COL_NAME].tolist()
    addr_list = s1_country_df[COL_ADDRESS].tolist()
    postal_list = s1_country_df[COL_POSTAL_CODE].tolist()
    id_list = s1_country_df[COL_ENTITY_ID].tolist()

    print(f"  [{country_label}] generating candidates for {n:,} Source 1 entities...")
    start_time = time.time()

    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")

        for batch_start in range(0, n, query_progress_batch_size):
            batch_end = min(batch_start + query_progress_batch_size, n)

            for i in range(batch_start, batch_end):
                positions, counts = generate_candidates_for_entity(
                    name_list[i], addr_list[i], postal_list[i], indexes, k
                )
                positions = trim_candidates(positions, counts, max_candidates)
                candidate_ids = indexes["ids"][positions] if len(positions) > 0 else np.array([], dtype=object)
                candidate_str = ",".join(candidate_ids.tolist())
                f.write(f"{id_list[i]}\t{candidate_str}\n")

            elapsed = time.time() - start_time
            rate = batch_end / elapsed if elapsed > 0 else 0
            print(f"    [{country_label}] {batch_end:,}/{n:,} entities "
                  f"({elapsed:.1f}s elapsed, ~{rate:.0f} entities/sec)")
            gc.collect()

    os.rename(tmp_path, final_path)
    print(f"  [{country_label}] done -> {final_path.name}")

    del indexes
    gc.collect()
    return final_path