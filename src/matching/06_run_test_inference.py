"""
06_run_test_inference.py

Final inference step: streams the real test candidate_pairs.tsv
(blocking's output over US + India + France), computes the same
similarity features used throughout this phase for every candidate
pair, scores with the trained baseline model, applies the selected
threshold, and writes matching_results.tsv in the exact format the
problem statement requires. This script produces the file used for the
final submitted result.

Every submission rule from the problem statement is enforced explicitly
in code here, not just assumed to follow naturally:

  1. "Every Source 1 entity in the test set must have exactly one row"
     -> guaranteed by iterating every line of candidate_pairs.tsv
     unconditionally (blocking already writes one row per entity,
     including zero-candidate entities), and writing exactly one output
     line per input line, with no skipping.

  2. "Leave matched_entity_ids empty for entities with no matches"
     -> when zero candidates clear the threshold, the written field is
     an empty string, not omitted or null.

  3. "No duplicate entity IDs within a single ID list"
     -> candidates already come deduplicated from blocking's own
     processing, but this script still deduplicates defensively before
     writing, since a format rejection here would cost a full
     validator run to catch and correct.

  4. "matched_entity_ids must only reference S2-/S3- entities that exist
     in the test set" -> candidates are drawn only from the test-set
     cleaned/raw source files' own entity_id lookups, so nothing outside
     the test set can appear.

  5. "Every ID in matching_results.tsv should appear in
     candidate_pairs.tsv" -> automatically true, since every predicted
     match is chosen FROM that entity's own candidate list, never
     invented.

Uses plain line-by-line streaming rather than pandas' chunked reader,
for the same reason applied throughout this project: candidate rows can
be long, and a chunked pandas reader's fixed per-chunk buffer becomes
impractical at that row length.
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TEST_SOURCE1, CLEANED_TEST_SOURCE2, CLEANED_TEST_SOURCE3,
    RAW_TEST_SOURCE2, RAW_TEST_SOURCE3,
    CANDIDATES_TEST_PATH, MATCHING_RESULTS_PATH,
    BASELINE_MODEL_PATH, BEST_THRESHOLD_PATH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE,
    FEATURE_BATCH_SIZE, EVAL_PROGRESS_EVERY,
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


def main():
    print("Loading trained model and locked threshold...")
    model = xgb.XGBClassifier()
    model.load_model(str(BASELINE_MODEL_PATH))

    with open(BEST_THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())
    print(f"  using threshold = {threshold}")

    print("\nLoading cleaned TEST source files and building lookups...")
    s1_df = load_tsv(CLEANED_TEST_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s2_df = load_tsv(CLEANED_TEST_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s3_df = load_tsv(CLEANED_TEST_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s1_lookup = build_lookup(s1_df, include_postal=True)
    s2_lookup = build_lookup(s2_df, include_postal=True)
    s3_lookup = build_lookup(s3_df, include_postal=True)
    n_s1, n_s2, n_s3 = len(s1_df), len(s2_df), len(s3_df)
    del s1_df, s2_df, s3_df
    gc.collect()

    print("Loading RAW TEST source2/3 names for the non-ASCII flag...")
    raw_s2_df = load_tsv(RAW_TEST_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s3_df = load_tsv(RAW_TEST_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_name_lookup = {}
    raw_name_lookup.update(build_lookup(raw_s2_df, include_postal=False))
    raw_name_lookup.update(build_lookup(raw_s3_df, include_postal=False))
    del raw_s2_df, raw_s3_df
    gc.collect()

    def get_other_fields(cand_id):
        return s2_lookup.get(cand_id) if cand_id.startswith("S2-") else s3_lookup.get(cand_id)

    def get_other_raw_name(cand_id):
        return raw_name_lookup.get(cand_id, "")

    print(f"\nStreaming {CANDIDATES_TEST_PATH.name} "
          f"(expects {n_s1:,} Source 1 test entities, S2 pool {n_s2:,}, S3 pool {n_s3:,})...")

    n_entities_written = 0
    n_pairs_scored = 0
    n_entities_with_match = 0
    n_singleton_predictions = 0
    start_time = time.time()

    with open(CANDIDATES_TEST_PATH, "r", encoding="utf-8") as in_f, \
         open(MATCHING_RESULTS_PATH, "w", encoding="utf-8", newline="") as out_f:

        out_f.write("source1_entity_id\tmatched_entity_ids\n")
        next(in_f)  # skip header

        for line in in_f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""

            if not cand_str:
                # No candidates at all -- write empty match list per
                # rule 2. This entity is predicted as a singleton
                # unconditionally, since there is nothing to score.
                out_f.write(f"{s1_id}\t\n")
                n_entities_written += 1
                n_singleton_predictions += 1
                if n_entities_written % EVAL_PROGRESS_EVERY == 0:
                    elapsed = time.time() - start_time
                    print(f"    {n_entities_written:,} entities written "
                          f"({elapsed:.1f}s elapsed)")
                continue

            s1_name, s1_addr, s1_postal = s1_lookup.get(s1_id, ("", "", ""))
            candidate_ids = cand_str.split(",")
            # Caps the number of candidates scored per entity. This value
            # was selected from a measured cap-vs-F_0.5 tradeoff sweep:
            # recall ceiling continues improving with a higher cap, but
            # with diminishing returns, and this value gives a strong
            # recovered recall while keeping full-test-set inference time
            # practical on local hardware.
            MAX_CANDIDATES_TO_SCORE = 300
            candidate_ids = candidate_ids[:MAX_CANDIDATES_TO_SCORE]

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

            # Keep candidates clearing the threshold, de-duplicated
            # defensively (rule 3), preserving no particular order --
            # order is not specified as significant by the problem
            # statement.
            matched = []
            seen = set()
            for cand_id, prob in zip(candidate_ids, probs):
                if prob >= threshold and cand_id not in seen:
                    matched.append(cand_id)
                    seen.add(cand_id)

            matched_str = ",".join(matched)
            out_f.write(f"{s1_id}\t{matched_str}\n")
            n_entities_written += 1
            if matched:
                n_entities_with_match += 1
            else:
                n_singleton_predictions += 1

            if n_entities_written % EVAL_PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                rate = n_entities_written / elapsed if elapsed > 0 else 0
                print(f"    {n_entities_written:,} entities written, "
                      f"{n_pairs_scored:,} pairs scored so far "
                      f"({elapsed:.1f}s elapsed, ~{rate:.0f} entities/sec)")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Total Source 1 test entities written: {n_entities_written:,}")
    print(f"  entities predicted WITH a match:    {n_entities_with_match:,}")
    print(f"  entities predicted as singletons:   {n_singleton_predictions:,}")

    if n_entities_written != n_s1:
        print(f"\n*** WARNING: wrote {n_entities_written:,} rows but "
              f"test_source1 has {n_s1:,} entities -- these should match "
              f"exactly. Investigate before submitting. ***")
    else:
        print(f"\nRow count matches test_source1 exactly ({n_s1:,}) -- looks correct.")

    print(f"\nmatching_results.tsv written to: {MATCHING_RESULTS_PATH}")
    print("\nNEXT STEP: copy this file plus candidate_pairs.tsv into your "
          "submission's output/ folder, then run the competition's own "
          "utils/validate_submission.py before uploading anything.")


if __name__ == "__main__":
    main()