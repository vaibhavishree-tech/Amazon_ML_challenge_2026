"""
09_test_parallel_speedup.py

Builds filtered lookups, spins up a worker pool, and reproduces
07_measure_capped_performance.py's result (macro F_0.5 at cap=50,
locked threshold) using the parallelized scoring pipeline instead of
the single-threaded one.

This is a correctness-and-speed GATE, not just a speed test: if the
parallel result doesn't closely match the single-threaded result, the
parallel implementation has a bug that must be fixed before it is
trusted at full test-set scale, where a silent correctness bug would
be far more expensive to discover.

Memory strategy: reads eval_candidates_full.tsv once up front to
collect the exact set of entity IDs that will actually be needed
(respecting the same candidate cap used for scoring), then loads the
full cleaned/raw source files only in the main process, filters
lookups down to that referenced set, and releases the full dataframes
before creating the worker pool. Workers receive only these small,
filtered dicts -- never the full multi-million-row source lookups --
keeping total memory bounded regardless of worker count.

Windows multiprocessing requires this exact "if __name__ == '__main__':"
guard around the Pool creation -- without it, Windows' spawn-based
process creation causes each new process to re-execute this whole
script from the top, including recursively creating more pools. This
is a Windows-specific requirement, not a stylistic choice.
"""

import gc
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE3,
    RAW_TRAIN_SOURCE2, RAW_TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    EVAL_CANDIDATES_PATH, BASELINE_MODEL_PATH, BEST_THRESHOLD_PATH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    N_WORKERS, WORKER_CHUNK_SIZE, PARALLEL_TEST_MAX_CANDIDATES,
)
from io_utils import load_tsv
from parallel_worker import init_worker, score_batch


def explode_matched_ids(gt):
    """Same explode pattern used throughout this project."""
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"}).reset_index(drop=True)


def compute_f_beta(precision, recall, beta=0.5):
    """F-beta score with beta=0.5, matching the competition's F_0.5 metric."""
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def chunk_list(items, chunk_size):
    """Yields successive fixed-size chunks from a list, for dispatching work to the worker pool."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def main():
    print("Reading eval_candidates_full.tsv once to find referenced entity IDs...")
    all_lines = []
    referenced_s1_ids = set()
    referenced_other_ids = set()

    with open(EVAL_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            all_lines.append(stripped)
            parts = stripped.split("\t")
            s1_id = parts[0]
            referenced_s1_ids.add(s1_id)
            cand_str = parts[1] if len(parts) > 1 else ""
            if cand_str:
                # Apply the SAME cap here as scoring will use, so we
                # only build lookups for IDs that will actually be
                # touched -- no wasted memory on candidates beyond the cap.
                for cand_id in cand_str.split(",")[:PARALLEL_TEST_MAX_CANDIDATES]:
                    referenced_other_ids.add(cand_id)

    print(f"  {len(all_lines):,} lines, {len(referenced_s1_ids):,} S1 entities, "
          f"{len(referenced_other_ids):,} unique referenced candidates")

    print("\nLoading full cleaned source files (main process only)...")
    s1_df = load_tsv(CLEANED_TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s2_df = load_tsv(CLEANED_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s3_df = load_tsv(CLEANED_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])

    print("Filtering to referenced IDs only, before building lookups (memory control)...")
    s1_df = s1_df[s1_df[COL_ENTITY_ID].isin(referenced_s1_ids)]
    s2_df = s2_df[s2_df[COL_ENTITY_ID].isin(referenced_other_ids)]
    s3_df = s3_df[s3_df[COL_ENTITY_ID].isin(referenced_other_ids)]

    s1_lookup = dict(zip(s1_df[COL_ENTITY_ID], zip(s1_df[COL_NAME], s1_df[COL_ADDRESS], s1_df[COL_POSTAL_CODE])))
    other_lookup = {}
    other_lookup.update(dict(zip(s2_df[COL_ENTITY_ID], zip(s2_df[COL_NAME], s2_df[COL_ADDRESS], s2_df[COL_POSTAL_CODE]))))
    other_lookup.update(dict(zip(s3_df[COL_ENTITY_ID], zip(s3_df[COL_NAME], s3_df[COL_ADDRESS], s3_df[COL_POSTAL_CODE]))))
    del s1_df, s2_df, s3_df
    gc.collect()
    print(f"  filtered lookups built: {len(s1_lookup):,} S1, {len(other_lookup):,} S2/S3")

    print("\nLoading RAW source2/3 names, filtering to referenced IDs...")
    raw_s2_df = load_tsv(RAW_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s3_df = load_tsv(RAW_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s2_df = raw_s2_df[raw_s2_df[COL_ENTITY_ID].isin(referenced_other_ids)]
    raw_s3_df = raw_s3_df[raw_s3_df[COL_ENTITY_ID].isin(referenced_other_ids)]
    raw_name_lookup = {}
    raw_name_lookup.update(dict(zip(raw_s2_df[COL_ENTITY_ID], raw_s2_df[COL_NAME])))
    raw_name_lookup.update(dict(zip(raw_s3_df[COL_ENTITY_ID], raw_s3_df[COL_NAME])))
    del raw_s2_df, raw_s3_df
    gc.collect()
    print(f"  filtered raw name lookup: {len(raw_name_lookup):,} entries")

    print("\nLoading ground truth for the correctness check...")
    gt = load_tsv(TRAIN_GROUND_TRUTH)
    exploded = explode_matched_ids(gt)
    true_matches = defaultdict(set)
    for s1_id, matched_id in zip(exploded[COL_GT_SOURCE1_ID], exploded["matched_id"]):
        true_matches[s1_id].add(matched_id)
    del gt, exploded
    gc.collect()

    with open(BEST_THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())

    chunks = list(chunk_list(all_lines, WORKER_CHUNK_SIZE))
    print(f"\nStarting pool of {N_WORKERS} workers, {len(chunks)} chunks of up to "
          f"{WORKER_CHUNK_SIZE:,} lines each...")

    entity_predictions = defaultdict(set)  # s1_id -> set of predicted candidate_ids above threshold
    all_s1_ids_seen = set()

    start_time = time.time()
    n_pairs_scored = 0

    with mp.Pool(
        processes=N_WORKERS,
        initializer=init_worker,
        initargs=(s1_lookup, other_lookup, raw_name_lookup, str(BASELINE_MODEL_PATH), PARALLEL_TEST_MAX_CANDIDATES),
    ) as pool:
        for batch_idx, batch_results in enumerate(pool.imap_unordered(score_batch, chunks)):
            for s1_id, cand_id, prob in batch_results:
                all_s1_ids_seen.add(s1_id)
                if cand_id is None:
                    continue  # zero-candidate entity -- no prediction to add
                n_pairs_scored += 1
                if prob >= threshold:
                    entity_predictions[s1_id].add(cand_id)

            elapsed = time.time() - start_time
            print(f"    chunk {batch_idx + 1}/{len(chunks)} done "
                  f"({n_pairs_scored:,} pairs scored, {elapsed:.1f}s elapsed)")

    elapsed = time.time() - start_time
    rate = n_pairs_scored / elapsed if elapsed > 0 else 0
    print(f"\nParallel scoring done in {elapsed:.1f}s "
          f"({n_pairs_scored:,} pairs, ~{rate:.0f} pairs/sec across {N_WORKERS} workers)")

    print("\nComputing macro F_0.5 (should closely match the single-threaded result)...")
    f_scores = []
    for s1_id in all_s1_ids_seen:
        predicted = entity_predictions.get(s1_id, set())
        true_set = true_matches.get(s1_id, set())

        if not true_set and not predicted:
            f_scores.append(1.0)
        elif not predicted or not true_set:
            f_scores.append(0.0)
        else:
            overlap = predicted & true_set
            precision = len(overlap) / len(predicted)
            recall = len(overlap) / len(true_set)
            f_scores.append(compute_f_beta(precision, recall))

    macro_f0_5 = float(np.mean(f_scores))
    print(f"\n=== PARALLEL RESULT ===")
    print(f"  Entities evaluated: {len(f_scores):,}")
    print(f"  Macro F_0.5:        {macro_f0_5:.4f}")
    print(f"  (single-threaded reference result was: 0.8586)")

    diff = abs(macro_f0_5 - 0.8586)
    if diff < 0.002:
        print(f"\n  MATCH (diff={diff:.4f}) -- parallel implementation is correct. "
              f"Safe to use on the real test set.")
    else:
        print(f"\n  MISMATCH (diff={diff:.4f}) -- something differs from the single-threaded "
              f"version. Do NOT trust this on the full test set yet -- investigate before proceeding.")


if __name__ == "__main__":
    main()