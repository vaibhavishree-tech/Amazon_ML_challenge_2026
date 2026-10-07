"""
05_score_and_select_threshold.py

Step 2 of threshold selection: computes features for every candidate in
eval_candidates_full.tsv (full pools, not sampled), scores them all with
the trained model, then sweeps THRESHOLD_GRID to find the threshold that
maximizes macro-averaged F_0.5 -- computed per Source 1 entity, exactly
as the competition scores it: a correctly-predicted empty match list
(singleton) scores 1.0, an entity with true matches that receives zero
predictions scores 0.0, and partial overlaps use the real F_0.5 formula.

This re-implements the lookup-building and feature-computation approach
from 01_compute_training_features.py directly, rather than importing
it, since a filename starting with a digit cannot be imported as a
standard Python module without extra tooling.
"""

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE3,
    RAW_TRAIN_SOURCE2, RAW_TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    EVAL_CANDIDATES_PATH, EVAL_SCORED_PATH, BEST_THRESHOLD_PATH,
    BASELINE_MODEL_PATH, THRESHOLD_GRID,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    FEATURE_BATCH_SIZE, PROGRESS_EVERY,
)
from io_utils import load_tsv
from text_similarity_features import compute_pair_features, FEATURE_COLUMNS


def build_lookup(df, include_postal: bool) -> dict:
    """entity_id -> (name, address, postal_code) or entity_id -> name, built via dict(zip(...))."""
    if include_postal:
        return dict(zip(
            df[COL_ENTITY_ID],
            zip(df[COL_NAME], df[COL_ADDRESS], df[COL_POSTAL_CODE]),
        ))
    return dict(zip(df[COL_ENTITY_ID], df[COL_NAME]))


def explode_matched_ids(gt: pd.DataFrame) -> pd.DataFrame:
    """Same explode pattern used throughout the earlier phases of this project."""
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"}).reset_index(drop=True)


def compute_f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    """F-beta score with beta=0.5, matching the competition's official F_0.5 metric definition."""
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


def score_at_threshold(entity_groups: dict, true_matches: dict, threshold: float) -> dict:
    """
    For one threshold, computes per-entity precision/recall/F_0.5 and
    returns the macro-average across all sampled entities -- including
    entities with no candidates at all, and entities whose true match
    set is empty (singletons), both handled per the competition's own
    scoring rule (empty predicted + empty true = 1.0).
    """
    f_scores = []
    for entity_id, cand_probs in entity_groups.items():
        predicted = {cand_id for cand_id, prob in cand_probs if prob >= threshold}
        true_set = true_matches.get(entity_id, set())

        if not true_set and not predicted:
            f_scores.append(1.0)
            continue
        if not predicted or not true_set:
            f_scores.append(0.0)
            continue

        overlap = predicted & true_set
        precision = len(overlap) / len(predicted)
        recall = len(overlap) / len(true_set)
        f_scores.append(compute_f_beta(precision, recall))

    return {
        "threshold": threshold,
        "macro_f0.5": float(np.mean(f_scores)),
        "n_entities": len(f_scores),
    }


def main():
    print("Loading cleaned source files and building lookups...")
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

    print("\nLoading model...")
    model = xgb.XGBClassifier()
    model.load_model(str(BASELINE_MODEL_PATH))

    print("\nLoading full ground truth true-match sets...")
    gt = load_tsv(TRAIN_GROUND_TRUTH)
    exploded = explode_matched_ids(gt)
    true_matches = defaultdict(set)
    for s1_id, matched_id in zip(exploded[COL_GT_SOURCE1_ID], exploded["matched_id"]):
        true_matches[s1_id].add(matched_id)
    del gt, exploded
    gc.collect()

    print(f"\nStreaming {EVAL_CANDIDATES_PATH.name}, computing features and scoring "
          f"in batches of {FEATURE_BATCH_SIZE:,}...")

    entity_groups = defaultdict(list)  # entity_id -> list of (candidate_id, probability)
    n_entities_seen = 0
    n_pairs_scored = 0
    start_time = time.time()

    with open(EVAL_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        next(f)  # skip header

        batch_rows = []  # (s1_id, cand_id, feature_list)

        def flush_batch():
            nonlocal n_pairs_scored
            if not batch_rows:
                return
            feature_matrix = np.array([r[2] for r in batch_rows], dtype="float64")
            probs = model.predict_proba(feature_matrix)[:, 1]
            for (s1_id, cand_id, _), prob in zip(batch_rows, probs):
                entity_groups[s1_id].append((cand_id, float(prob)))
            n_pairs_scored += len(batch_rows)
            batch_rows.clear()

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""

            n_entities_seen += 1
            entity_groups.setdefault(s1_id, [])  # ensure zero-candidate entities are represented

            if not cand_str:
                continue

            s1_name, s1_addr, s1_postal = s1_lookup.get(s1_id, ("", "", ""))

            for cand_id in cand_str.split(","):
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
                batch_rows.append((s1_id, cand_id, features))

                if len(batch_rows) >= FEATURE_BATCH_SIZE:
                    flush_batch()

            if n_entities_seen % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                print(f"    {n_entities_seen:,} entities processed, "
                      f"{n_pairs_scored:,} pairs scored so far ({elapsed:.1f}s elapsed)")

        flush_batch()

    elapsed = time.time() - start_time
    print(f"\nDone scoring. {n_entities_seen:,} entities, {n_pairs_scored:,} candidate "
          f"pairs scored in {elapsed:.1f}s.")

    print("\nSweeping threshold grid for best macro F_0.5...\n")
    results = []
    for threshold in THRESHOLD_GRID:
        result = score_at_threshold(entity_groups, true_matches, threshold)
        results.append(result)
        print(f"  threshold={threshold:.3f}  macro_F0.5={result['macro_f0.5']:.4f}  "
              f"(n_entities={result['n_entities']:,})")

    best = max(results, key=lambda r: r["macro_f0.5"])
    print(f"\nBEST THRESHOLD: {best['threshold']} "
          f"(macro F_0.5 = {best['macro_f0.5']:.4f})")

    with open(BEST_THRESHOLD_PATH, "w") as f:
        f.write(str(best["threshold"]))
    print(f"\nBest threshold saved to: {BEST_THRESHOLD_PATH}")


if __name__ == "__main__":
    main()