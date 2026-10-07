"""
10_test_candidate_caps.py

Sweeps several candidate-per-entity caps on the same verified 50,000-
entity eval sample used in the parallel-pipeline correctness check,
reusing the exact same scoring pipeline (parallel_worker.py), so these
numbers are directly comparable to the already-confirmed cap=50 and
uncapped reference results -- the same methodology, swept across
multiple cap values.

This exists specifically so the final candidate-cap decision is based
on measured numbers across several real options, not interpolation
between only two data points.
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
    N_WORKERS, WORKER_CHUNK_SIZE, CAP_TEST_VALUES,
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


def run_one_cap(cap, all_lines, pool, true_matches, threshold):
    """
    Runs scoring for ONE cap value using an ALREADY-CREATED pool (passed
    in), rather than creating a new pool per cap -- pool creation itself
    has real overhead, so reusing one pool across the whole sweep keeps
    the comparison fast. The pool's workers were initialized with
    max(CAP_TEST_VALUES) as their candidate limit, so to test a SMALLER
    cap here, truncation is applied to the candidate lists on this side,
    before dispatching lines to workers, rather than inside the worker.
    """
    # Truncate candidate lists to THIS cap before sending to workers,
    # since the pool's workers were initialized with the largest cap.
    capped_lines = []
    for line in all_lines:
        parts = line.split("\t")
        s1_id = parts[0]
        cand_str = parts[1] if len(parts) > 1 else ""
        if cand_str:
            capped_cand_str = ",".join(cand_str.split(",")[:cap])
            capped_lines.append(f"{s1_id}\t{capped_cand_str}")
        else:
            capped_lines.append(f"{s1_id}\t")

    chunks = list(chunk_list(capped_lines, WORKER_CHUNK_SIZE))

    entity_predictions = defaultdict(set)
    all_s1_ids_seen = set()
    n_pairs_scored = 0
    start_time = time.time()

    for batch_results in pool.imap_unordered(score_batch, chunks):
        for s1_id, cand_id, prob in batch_results:
            all_s1_ids_seen.add(s1_id)
            if cand_id is None:
                continue
            n_pairs_scored += 1
            if prob >= threshold:
                entity_predictions[s1_id].add(cand_id)

    elapsed = time.time() - start_time

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
    return macro_f0_5, elapsed, n_pairs_scored


def main():
    print("Reading eval_candidates_full.tsv and finding referenced entity IDs "
          f"(using the largest cap in the sweep, {max(CAP_TEST_VALUES)}, so all "
          "smaller caps' needs are covered)...")
    max_cap = max(CAP_TEST_VALUES)
    all_lines = []
    referenced_s1_ids = set()
    referenced_other_ids = set()

    with open(EVAL_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        next(f)
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
                for cand_id in cand_str.split(",")[:max_cap]:
                    referenced_other_ids.add(cand_id)

    print(f"  {len(all_lines):,} lines, {len(referenced_s1_ids):,} S1 entities, "
          f"{len(referenced_other_ids):,} unique referenced candidates")

    print("\nLoading and filtering source files (main process only)...")
    s1_df = load_tsv(CLEANED_TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s2_df = load_tsv(CLEANED_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s3_df = load_tsv(CLEANED_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s1_df = s1_df[s1_df[COL_ENTITY_ID].isin(referenced_s1_ids)]
    s2_df = s2_df[s2_df[COL_ENTITY_ID].isin(referenced_other_ids)]
    s3_df = s3_df[s3_df[COL_ENTITY_ID].isin(referenced_other_ids)]

    s1_lookup = dict(zip(s1_df[COL_ENTITY_ID], zip(s1_df[COL_NAME], s1_df[COL_ADDRESS], s1_df[COL_POSTAL_CODE])))
    other_lookup = {}
    other_lookup.update(dict(zip(s2_df[COL_ENTITY_ID], zip(s2_df[COL_NAME], s2_df[COL_ADDRESS], s2_df[COL_POSTAL_CODE]))))
    other_lookup.update(dict(zip(s3_df[COL_ENTITY_ID], zip(s3_df[COL_NAME], s3_df[COL_ADDRESS], s3_df[COL_POSTAL_CODE]))))
    del s1_df, s2_df, s3_df
    gc.collect()

    raw_s2_df = load_tsv(RAW_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s3_df = load_tsv(RAW_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s2_df = raw_s2_df[raw_s2_df[COL_ENTITY_ID].isin(referenced_other_ids)]
    raw_s3_df = raw_s3_df[raw_s3_df[COL_ENTITY_ID].isin(referenced_other_ids)]
    raw_name_lookup = {}
    raw_name_lookup.update(dict(zip(raw_s2_df[COL_ENTITY_ID], raw_s2_df[COL_NAME])))
    raw_name_lookup.update(dict(zip(raw_s3_df[COL_ENTITY_ID], raw_s3_df[COL_NAME])))
    del raw_s2_df, raw_s3_df
    gc.collect()
    print("  lookups built.")

    print("\nLoading ground truth...")
    gt = load_tsv(TRAIN_GROUND_TRUTH)
    exploded = explode_matched_ids(gt)
    true_matches = defaultdict(set)
    for s1_id, matched_id in zip(exploded[COL_GT_SOURCE1_ID], exploded["matched_id"]):
        true_matches[s1_id].add(matched_id)
    del gt, exploded
    gc.collect()

    with open(BEST_THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())

    # Fewer workers than the project default, specifically for this
    # sweep: holding one pool open across several caps while repeatedly
    # dispatching full-sized chunks is more memory-sensitive than a
    # single scoring pass, so a smaller worker count is used here as a
    # safety margin.
    TEST_WORKERS = 2
    print(f"\nSweeping caps: {CAP_TEST_VALUES} (threshold={threshold}, {TEST_WORKERS} workers, "
          f"ONE pool reused across all caps to avoid repeated large pickling)\n")

    results = []
    with mp.Pool(
        processes=TEST_WORKERS,
        initializer=init_worker,
        initargs=(s1_lookup, other_lookup, raw_name_lookup, str(BASELINE_MODEL_PATH), max(CAP_TEST_VALUES)),
    ) as pool:
        for cap in CAP_TEST_VALUES:
            print(f"--- Testing cap={cap} ---")
            macro_f0_5, elapsed, n_pairs = run_one_cap(cap, all_lines, pool, true_matches, threshold)
            rate = n_pairs / elapsed if elapsed > 0 else 0
            print(f"  macro F_0.5 = {macro_f0_5:.4f}  |  {n_pairs:,} pairs in {elapsed:.1f}s  "
              f"(~{rate:.0f} pairs/sec)\n")
            results.append({"cap": cap, "macro_f0_5": macro_f0_5, "pairs": n_pairs, "seconds": elapsed, "pairs_per_sec": rate})

    print("=== SUMMARY ===\n")
    print(f"{'cap':>6}  {'macro_F0.5':>11}  {'pairs/sec':>10}  {'est. hours for ~865M pairs':>28}")
    for r in results:
        est_hours = 865_000_000 / r["pairs_per_sec"] / 3600 if r["pairs_per_sec"] > 0 else float("inf")
        print(f"{r['cap']:>6}  {r['macro_f0_5']:>11.4f}  {r['pairs_per_sec']:>10.0f}  {est_hours:>28.2f}")

    print("\nPick the cap that best balances macro_F0.5 against the estimated real "
          "test-set runtime -- note the 'pairs/sec' scales with the ACTUAL test set "
          "having far more total pairs than this 50,000-entity sample, so use the "
          "estimated-hours column as a planning guide, not an exact promise.")


if __name__ == "__main__":
    main()