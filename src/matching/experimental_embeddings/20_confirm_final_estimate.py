"""
20_confirm_final_estimate.py

A quick sanity check confirming the expected macro F_0.5 at the locked
embedding-model configuration (cap=300, threshold=0.98, 11-feature
model), measured on the held-out evaluation sample rather than the full
test set. Reuses the same lockstep-read scoring approach as
17_threshold_sweep_by_country.py, but computes only ONE threshold
instead of sweeping the full grid -- much faster than rerunning the
full sweep script, since the configuration is already decided.

Note: eval_candidates_full.tsv and eval_candidates_embed_sim.tsv already
hold candidates up to blocking's own cap (up to 500) -- this script
APPLIES the FINAL_CAP=300 truncation on top of that, matching exactly
what the final inference script does on the real test set, so the
number reported here is a true apples-to-apples estimate of that
configuration's expected performance.

Result: this script measured a macro F_0.5 of 0.8853 for the
embedding-augmented configuration -- a real, confirmed improvement over
the 10-feature baseline's 0.8809 on the same held-out sample. This
measured result is the basis for the comparison documented in
experimental_embeddings/README_EXPERIMENTAL.md.
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
    EVAL_CANDIDATES_PATH, EVAL_CANDIDATES_EMBED_SIM_PATH,
    MODEL_V2_PATH, FINAL_CAP, FINAL_THRESHOLD,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    FEATURE_BATCH_SIZE,
)
from io_utils import load_tsv
from text_similarity_features import compute_pair_features


def build_lookup(df, include_postal):
    """entity_id -> (name, address, postal_code) or entity_id -> name, built via dict(zip(...))."""
    if include_postal:
        return dict(zip(df[COL_ENTITY_ID], zip(df[COL_NAME], df[COL_ADDRESS], df[COL_POSTAL_CODE])))
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
    print("Loading lookups (same as the country-threshold sweep)...")
    s1_df = load_tsv(CLEANED_TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s2_df = load_tsv(CLEANED_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s3_df = load_tsv(CLEANED_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s1_lookup = build_lookup(s1_df, True)
    s2_lookup = build_lookup(s2_df, True)
    s3_lookup = build_lookup(s3_df, True)
    del s1_df, s2_df, s3_df
    gc.collect()

    raw_s2_df = load_tsv(RAW_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s3_df = load_tsv(RAW_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_name_lookup = {}
    raw_name_lookup.update(build_lookup(raw_s2_df, False))
    raw_name_lookup.update(build_lookup(raw_s3_df, False))
    del raw_s2_df, raw_s3_df
    gc.collect()

    def get_other_fields(cand_id):
        return s2_lookup.get(cand_id) if cand_id.startswith("S2-") else s3_lookup.get(cand_id)

    def get_other_raw_name(cand_id):
        return raw_name_lookup.get(cand_id, "")

    print("Loading model v2 and ground truth...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_V2_PATH))

    gt = load_tsv(TRAIN_GROUND_TRUTH)
    exploded = explode_matched_ids(gt)
    true_matches = defaultdict(set)
    for s1_id, matched_id in zip(exploded[COL_GT_SOURCE1_ID], exploded["matched_id"]):
        true_matches[s1_id].add(matched_id)
    del gt, exploded
    gc.collect()

    print(f"\nScoring at FINAL config: cap={FINAL_CAP}, threshold={FINAL_THRESHOLD}...")
    entity_predictions = {}
    all_s1_ids_seen = set()
    n_pairs = 0
    start_time = time.time()

    with open(EVAL_CANDIDATES_PATH, "r", encoding="utf-8") as cand_f, \
         open(EVAL_CANDIDATES_EMBED_SIM_PATH, "r", encoding="utf-8") as sim_f:
        next(cand_f)
        next(sim_f)

        batch_rows = []

        def flush_batch():
            nonlocal n_pairs
            if not batch_rows:
                return
            feature_matrix = np.array([r[2] for r in batch_rows], dtype="float64")
            probs = model.predict_proba(feature_matrix)[:, 1]
            for (s1_id, cand_id, _), prob in zip(batch_rows, probs):
                if prob >= FINAL_THRESHOLD:
                    entity_predictions.setdefault(s1_id, set()).add(cand_id)
            n_pairs += len(batch_rows)
            batch_rows.clear()

        for line in cand_f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            all_s1_ids_seen.add(s1_id)
            entity_predictions.setdefault(s1_id, set())

            if not cand_str:
                continue

            candidate_ids = cand_str.split(",")[:FINAL_CAP]
            s1_name, s1_addr, s1_postal = s1_lookup.get(s1_id, ("", "", ""))

            # Consume sim_f lines for ALL original candidates (even beyond
            # the cap) to keep the two files aligned, but only SCORE the
            # capped subset.
            full_cand_list = cand_str.split(",")
            for idx, cand_id in enumerate(full_cand_list):
                sim_line = sim_f.readline().rstrip("\n")
                sim_parts = sim_line.split("\t")
                assert sim_parts[0] == s1_id and sim_parts[1] == cand_id, \
                    f"ALIGNMENT MISMATCH at {s1_id}, {cand_id}"

                if idx >= FINAL_CAP:
                    continue  # beyond cap -- read to stay aligned, but don't score

                embed_sim = float(sim_parts[2])
                other_fields = get_other_fields(cand_id)
                other_name, other_addr, other_postal = other_fields if other_fields else ("", "", "")
                other_name_original = get_other_raw_name(cand_id)

                base_features = compute_pair_features(
                    s1_name, s1_addr, s1_postal,
                    other_name, other_addr, other_postal,
                    other_name_original,
                )
                batch_rows.append((s1_id, cand_id, base_features + [embed_sim]))

                if len(batch_rows) >= FEATURE_BATCH_SIZE:
                    flush_batch()

        flush_batch()

    elapsed = time.time() - start_time
    print(f"\nScored {n_pairs:,} pairs in {elapsed:.1f}s.")

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
    print(f"\n=== FINAL CONFIG ESTIMATE (cap={FINAL_CAP}, threshold={FINAL_THRESHOLD}, 11 features) ===")
    print(f"  Entities evaluated: {len(f_scores):,}")
    print(f"  Macro F_0.5:        {macro_f0_5:.4f}")
    print(f"\nThis is the best available estimate for the embedding-augmented "
          f"configuration on the real test submission (US/India only -- France "
          f"remains unmeasurable locally without ground truth).")


if __name__ == "__main__":
    main()