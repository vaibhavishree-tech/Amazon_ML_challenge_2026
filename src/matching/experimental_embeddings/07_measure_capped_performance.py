"""
07_measure_capped_performance.py

Measures the ground-truth-backed F_0.5 impact of capping candidate
scoring to MAX_CANDIDATES_TO_SCORE=50 candidates per entity, matching
the cap used in an earlier version of test inference.

Reuses eval_candidates_full.tsv -- the same full-candidate-pool sample
used for threshold selection -- so no new blocking or extraction work
is needed. The only differences from the full threshold-sweep script:
  1. Each entity's candidate list is truncated to the first 50
     candidates BEFORE scoring, mirroring what the capped inference run
     does on the real test set.
  2. No threshold sweep -- scores ONLY at the already-locked threshold
     (read from best_threshold.txt), since that's the value actually
     used for inference.

This gives an honest, ground-truth-verified estimate of test-set
performance for US/India-like data, since train_ground_truth.tsv only
covers training entities. It cannot measure France, which has zero
training examples -- that remains unmeasurable until the leaderboard
scores the real submission.
"""

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE3,
    RAW_TRAIN_SOURCE2, RAW_TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    EVAL_CANDIDATES_PATH, BASELINE_MODEL_PATH, BEST_THRESHOLD_PATH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    FEATURE_BATCH_SIZE, EVAL_PROGRESS_EVERY,
)
from io_utils import load_tsv
from text_similarity_features import compute_pair_features, FEATURE_COLUMNS

MAX_CANDIDATES_TO_SCORE = 50  # candidate cap being evaluated by this script


def build_lookup(df, include_postal: bool) -> dict:
    """entity_id -> (name, address, postal_code) or entity_id -> name, built via dict(zip(...))."""
    if include_postal:
        return dict(zip(
            df[COL_ENTITY_ID],
            zip(df[COL_NAME], df[COL_ADDRESS], df[COL_POSTAL_CODE]),
        ))
    return dict(zip(df[COL_ENTITY_ID], df[COL_NAME]))


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


def main():
    print("Loading trained model and locked threshold...")
    model = xgb.XGBClassifier()
    model.load_model(str(BASELINE_MODEL_PATH))
    with open(BEST_THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())
    print(f"  threshold = {threshold}, candidate cap = {MAX_CANDIDATES_TO_SCORE}")

    print("\nLoading cleaned source files and building lookups...")
    s1_df = load_tsv(CLEANED_TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s2_df = load_tsv(CLEANED_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s3_df = load_tsv(CLEANED_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s1_lookup = build_lookup(s1_df, include_postal=True)
    s2_lookup = build_lookup(s2_df, include_postal=True)
    s3_lookup = build_lookup(s3_df, include_postal=True)
    del s1_df, s2_df, s3_df
    gc.collect()

    print("Loading RAW source2/3 names for the non-ASCII flag...")
    raw_s2_df = load_tsv(RAW_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s3_df = load_tsv(RAW_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_name_lookup = {}
    raw_name_lookup.update(build_lookup(raw_s2_df, include_postal=False))
    raw_name_lookup.update(build_lookup(raw_s3_df, include_postal=False))
    del raw_s2_df, raw_s3_df
    gc.collect()

    def get_other_fields(cand_id):
        return s2_lookup.get(cand_id) if cand_id.startswith("S2-") else s3_lookup.get(cand_id)

    def get_other_raw_name(cand_id):
        return raw_name_lookup.get(cand_id, "")

    print("\nLoading full ground truth true-match sets...")
    gt = load_tsv(TRAIN_GROUND_TRUTH)
    exploded = explode_matched_ids(gt)
    true_matches = defaultdict(set)
    for s1_id, matched_id in zip(exploded[COL_GT_SOURCE1_ID], exploded["matched_id"]):
        true_matches[s1_id].add(matched_id)
    del gt, exploded
    gc.collect()

    print(f"\nStreaming {EVAL_CANDIDATES_PATH.name}, applying {MAX_CANDIDATES_TO_SCORE}-candidate "
          f"cap, computing features, scoring at threshold={threshold}...")

    f_scores = []
    n_entities_seen = 0
    n_pairs_scored = 0
    start_time = time.time()

    with open(EVAL_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        next(f)  # skip header

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""

            n_entities_seen += 1
            true_set = true_matches.get(s1_id, set())

            if not cand_str:
                # No candidates at all -- predicted empty by necessity.
                f_scores.append(1.0 if not true_set else 0.0)
                continue

            # Apply the SAME cap used in the real test inference run --
            # this is the entire point of this script.
            candidate_ids = cand_str.split(",")[:MAX_CANDIDATES_TO_SCORE]

            s1_name, s1_addr, s1_postal = s1_lookup.get(s1_id, ("", "", ""))

            feature_rows = []
            for cand_id in candidate_ids:
                other_fields = get_other_fields(cand_id)
                if other_fields is None:
                    other_name, other_addr, other_postal = "", "", ""
                else:
                    other_name, other_addr, other_postal = other_fields
                other_name_original = get_other_raw_name(cand_id)

                features = compute_pair_features(
                    s1_name, s1_addr, s1_postal,
                    other_name, other_addr, other_postal,
                    other_name_original,
                )
                feature_rows.append(features)

            n_pairs_scored += len(feature_rows)
            feature_matrix = np.array(feature_rows, dtype="float64")
            probs = model.predict_proba(feature_matrix)[:, 1]

            predicted = {cid for cid, p in zip(candidate_ids, probs) if p >= threshold}

            if not true_set and not predicted:
                f_scores.append(1.0)
            elif not predicted or not true_set:
                f_scores.append(0.0)
            else:
                overlap = predicted & true_set
                precision = len(overlap) / len(predicted)
                recall = len(overlap) / len(true_set)
                f_scores.append(compute_f_beta(precision, recall))

            if n_entities_seen % EVAL_PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                print(f"    {n_entities_seen:,} entities processed, "
                      f"{n_pairs_scored:,} pairs scored ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - start_time
    macro_f0_5 = float(np.mean(f_scores))

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"\n=== CAPPED PERFORMANCE ESTIMATE (cap={MAX_CANDIDATES_TO_SCORE}, threshold={threshold}) ===")
    print(f"  Entities evaluated: {len(f_scores):,}")
    print(f"  Macro F_0.5:        {macro_f0_5:.4f}")
    print(f"\nThis is measured on US/India training data only -- France has no "
          f"ground truth and cannot be measured this way. Treat this as a strong "
          f"estimate for the US/India portion of your real test score, with "
          f"France's contribution remaining unknown until the leaderboard scores it.")


if __name__ == "__main__":
    main()