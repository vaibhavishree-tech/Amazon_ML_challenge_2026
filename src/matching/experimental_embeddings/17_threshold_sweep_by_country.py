"""
17_threshold_sweep_by_country.py

Step B: re-runs the full-candidate-pool threshold sweep, but using the
11-feature model, and SPLIT BY COUNTRY instead of one global sweep --
directly targeting the possibility that a global threshold tuned on
US+India data may be poorly calibrated for a country the model has zero
training labels for (France).

Reads eval_candidates_full.tsv and eval_candidates_embed_sim.tsv IN
PARALLEL, LINE-BY-LINE, matching them by POSITION rather than building a
large lookup dictionary. This works because
15_augment_eval_candidates.py wrote eval_candidates_embed_sim.tsv by
iterating eval_candidates_full.tsv in the exact same order, one output
line per candidate, in the same left-to-right order as each row's
comma-separated candidate list -- so reading both files in lockstep
reconstructs the correct pairing without ever holding the full join in
memory at once. A sanity assertion on ID alignment catches any ordering
mismatch immediately rather than silently scoring wrong pairs.
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
    RAW_TRAIN_SOURCE1, RAW_TRAIN_SOURCE2, RAW_TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    EVAL_CANDIDATES_PATH, EVAL_CANDIDATES_EMBED_SIM_PATH,
    MODEL_V2_PATH, BEST_THRESHOLDS_BY_COUNTRY_PATH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE, COL_COUNTRY,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    FEATURE_BATCH_SIZE, EVAL_PROGRESS_EVERY, THRESHOLD_GRID, V2_FEATURE_COLUMNS,
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


def score_at_threshold_by_country(entity_groups, true_matches, entity_country, threshold):
    """Returns {country: macro_f0.5} plus an OVERALL key, for one threshold."""
    scores_by_country = defaultdict(list)
    for entity_id, cand_probs in entity_groups.items():
        predicted = {cid for cid, p in cand_probs if p >= threshold}
        true_set = true_matches.get(entity_id, set())
        country = entity_country.get(entity_id, "UNKNOWN")

        if not true_set and not predicted:
            f = 1.0
        elif not predicted or not true_set:
            f = 0.0
        else:
            overlap = predicted & true_set
            precision = len(overlap) / len(predicted)
            recall = len(overlap) / len(true_set)
            f = compute_f_beta(precision, recall)
        scores_by_country[country].append(f)

    result = {c: float(np.mean(v)) for c, v in scores_by_country.items()}
    all_scores = [s for v in scores_by_country.values() for s in v]
    result["OVERALL"] = float(np.mean(all_scores))
    return result


def main():
    print("Loading cleaned source files and building lookups...")
    s1_df = load_tsv(CLEANED_TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s2_df = load_tsv(CLEANED_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s3_df = load_tsv(CLEANED_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s1_lookup = build_lookup(s1_df, True)
    s2_lookup = build_lookup(s2_df, True)
    s3_lookup = build_lookup(s3_df, True)
    del s1_df, s2_df, s3_df
    gc.collect()

    print("Loading RAW source2/3 names (non-ASCII flag) and RAW source1 (country)...")
    raw_s2_df = load_tsv(RAW_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s3_df = load_tsv(RAW_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_name_lookup = {}
    raw_name_lookup.update(build_lookup(raw_s2_df, False))
    raw_name_lookup.update(build_lookup(raw_s3_df, False))
    del raw_s2_df, raw_s3_df

    raw_s1_df = load_tsv(RAW_TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_COUNTRY])
    entity_country = dict(zip(raw_s1_df[COL_ENTITY_ID], raw_s1_df[COL_COUNTRY]))
    del raw_s1_df
    gc.collect()

    def get_other_fields(cand_id):
        return s2_lookup.get(cand_id) if cand_id.startswith("S2-") else s3_lookup.get(cand_id)

    def get_other_raw_name(cand_id):
        return raw_name_lookup.get(cand_id, "")

    print("\nLoading model v2 and ground truth...")
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_V2_PATH))

    gt = load_tsv(TRAIN_GROUND_TRUTH)
    exploded = explode_matched_ids(gt)
    true_matches = defaultdict(set)
    for s1_id, matched_id in zip(exploded[COL_GT_SOURCE1_ID], exploded["matched_id"]):
        true_matches[s1_id].add(matched_id)
    del gt, exploded
    gc.collect()

    print(f"\nStreaming {EVAL_CANDIDATES_PATH.name} + {EVAL_CANDIDATES_EMBED_SIM_PATH.name} "
          f"in lockstep, computing 11 features, scoring...")

    entity_groups = defaultdict(list)
    n_entities_seen = 0
    n_pairs_scored = 0
    start_time = time.time()

    with open(EVAL_CANDIDATES_PATH, "r", encoding="utf-8") as cand_f, \
         open(EVAL_CANDIDATES_EMBED_SIM_PATH, "r", encoding="utf-8") as sim_f:

        next(cand_f)   # skip headers
        next(sim_f)

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

        for line in cand_f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""

            n_entities_seen += 1
            entity_groups.setdefault(s1_id, [])

            if not cand_str:
                continue

            candidate_ids = cand_str.split(",")
            s1_name, s1_addr, s1_postal = s1_lookup.get(s1_id, ("", "", ""))

            for cand_id in candidate_ids:
                sim_line = sim_f.readline().rstrip("\n")
                sim_parts = sim_line.split("\t")
                # Sanity check: confirm the two files are still aligned.
                assert sim_parts[0] == s1_id and sim_parts[1] == cand_id, (
                    f"ALIGNMENT MISMATCH: expected ({s1_id}, {cand_id}) but "
                    f"embed_sim file had ({sim_parts[0]}, {sim_parts[1]}) -- "
                    f"the two files are no longer in the same order. Stop and investigate."
                )
                embed_sim = float(sim_parts[2])

                other_fields = get_other_fields(cand_id)
                if other_fields is None:
                    other_name, other_addr, other_postal = "", "", ""
                else:
                    other_name, other_addr, other_postal = other_fields
                other_name_original = get_other_raw_name(cand_id)

                base_features = compute_pair_features(
                    s1_name, s1_addr, s1_postal,
                    other_name, other_addr, other_postal,
                    other_name_original,
                )
                full_features = base_features + [embed_sim]
                batch_rows.append((s1_id, cand_id, full_features))

                if len(batch_rows) >= FEATURE_BATCH_SIZE:
                    flush_batch()

            if n_entities_seen % EVAL_PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                print(f"    {n_entities_seen:,} entities processed, "
                      f"{n_pairs_scored:,} pairs scored ({elapsed:.1f}s elapsed)", flush=True)

        flush_batch()

    elapsed = time.time() - start_time
    print(f"\nDone scoring. {n_entities_seen:,} entities, {n_pairs_scored:,} pairs "
          f"in {elapsed:.1f}s.")

    print("\nSweeping thresholds, split by country...\n")
    all_results = []
    for threshold in THRESHOLD_GRID:
        result = score_at_threshold_by_country(entity_groups, true_matches, entity_country, threshold)
        all_results.append((threshold, result))
        countries_str = "  ".join(f"{c}={v:.4f}" for c, v in sorted(result.items()) if c != "OVERALL")
        print(f"  threshold={threshold:.3f}  OVERALL={result['OVERALL']:.4f}  {countries_str}")

    # Best threshold PER COUNTRY (each country independently maximized),
    # plus the best single global threshold for comparison.
    countries_present = sorted({c for _, r in all_results for c in r if c != "OVERALL"})
    best_per_country = {}
    for country in countries_present:
        best = max(all_results, key=lambda tr: tr[1].get(country, -1))
        best_per_country[country] = best[0]

    best_global = max(all_results, key=lambda tr: tr[1]["OVERALL"])

    print(f"\n=== BEST THRESHOLDS ===")
    for country, thresh in best_per_country.items():
        print(f"  {country}: {thresh}")
    print(f"  (best single global threshold would be: {best_global[0]}, OVERALL={best_global[1]['OVERALL']:.4f})")

    with open(BEST_THRESHOLDS_BY_COUNTRY_PATH, "w") as f:
        for country, thresh in best_per_country.items():
            f.write(f"{country}\t{thresh}\n")
    print(f"\nPer-country thresholds saved to: {BEST_THRESHOLDS_BY_COUNTRY_PATH}")


if __name__ == "__main__":
    main()