"""
02_ground_truth_topology.py

Phase 1 EDA - Step 2: Ground Truth and Match Topology Analysis.

Performance rewrite notes:

- singleton_and_cardinality originally used .apply() with a Python
  function (parse_matches) to count matches per row, then another
  .apply() to bucket the count. Rewritten to compute match_count with a
  single vectorized string operation (str.count(",") + 1, with the
  empty case handled separately), removing the per-row Python function
  call entirely.

- source_split_of_matches and country_consistency_in_true_matches
  originally used gt.iterrows() with dict.get() lookups. Rewritten to
  use the same explode-based long-format table as script 01, then
  pandas merge() to attach country information -- merge is implemented
  in optimized C/Cython inside pandas and is the correct tool for
  "attach a value from another table by matching key," rather than a
  per-row dictionary lookup inside a Python loop.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    COL_ENTITY_ID, COL_COUNTRY,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    OUTPUT_DIR,
)
from io_utils import load_tsv

REPORT_PATH = OUTPUT_DIR / "02_ground_truth_topology_report.txt"
CARDINALITY_CSV = OUTPUT_DIR / "02_match_cardinality_distribution.csv"


def explode_matched_ids(gt: pd.DataFrame) -> pd.DataFrame:
    """Same explode pattern as script 01 -- see that file for the full explanation."""
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"})


def singleton_and_cardinality(gt: pd.DataFrame):
    """
    Vectorized match-count computation: an empty matched_entity_ids cell
    means 0 matches; a non-empty cell has (number of commas + 1) matches.
    str.count(",") runs as a single vectorized regex pass across the
    whole column, avoiding a Python-level function call per row.
    """
    gt = gt.copy()
    raw = gt[COL_GT_MATCHED_IDS]
    is_empty = raw.str.strip() == ""
    gt["match_count"] = np.where(is_empty, 0, raw.str.count(",") + 1)

    # np.select applies each condition in order and assigns the
    # corresponding label on first match -- the vectorized equivalent of
    # an if/elif/else chain run once per row, but computed over the
    # whole column at once.
    gt["cardinality_bucket"] = np.select(
        [gt["match_count"] == 0, gt["match_count"] == 1, gt["match_count"] == 2],
        ["0 (singleton)", "1", "2"],
        default="3+",
    )

    n_total = len(gt)
    n_singleton = int((gt["match_count"] == 0).sum())
    pct_singleton = 100 * n_singleton / n_total if n_total else 0

    lines = ["=== Singleton Proportion ===\n"]
    lines.append(f"Total Source 1 entities in ground truth: {n_total:,}")
    lines.append(f"Singletons (zero matches): {n_singleton:,} ({pct_singleton:.2f}%)")
    lines.append("")

    lines.append("=== Match Cardinality Distribution ===\n")
    dist = gt["cardinality_bucket"].value_counts().sort_index()
    for bucket_label, count in dist.items():
        pct = 100 * count / n_total if n_total else 0
        lines.append(f"  {bucket_label:15s}: {count:>8,} ({pct:.2f}%)")

    dist.rename_axis("cardinality_bucket").reset_index(name="count").to_csv(
        CARDINALITY_CSV, index=False
    )

    # Returns the augmented ground truth dataframe (now carrying
    # match_count/cardinality_bucket) alongside the report text, so the
    # caller can reuse these computed columns later in main() without
    # recomputing them from scratch.
    return gt, "\n".join(lines) + "\n"


def source_split_of_matches(gt: pd.DataFrame, exploded: pd.DataFrame) -> str:
    """
    Rewritten without per-row classification logic. For each
    source1_entity_id, checks (via groupby + vectorized string
    startswith) whether its matched IDs include any S2- prefix, any S3-
    prefix, or both, then classifies using vectorized boolean logic
    instead of a Python function applied row by row.
    """
    lines = ["=== Source Split of Matches (entities with >= 1 match) ===\n"]

    exploded = exploded.copy()
    exploded["is_s2"] = exploded["matched_id"].str.startswith("S2-")
    exploded["is_s3"] = exploded["matched_id"].str.startswith("S3-")

    per_entity = exploded.groupby(COL_GT_SOURCE1_ID).agg(
        has_s2=("is_s2", "any"),
        has_s3=("is_s3", "any"),
    )

    per_entity["source_split"] = np.select(
        [per_entity["has_s2"] & per_entity["has_s3"],
         per_entity["has_s2"],
         per_entity["has_s3"]],
        ["both S2 and S3", "S2 only", "S3 only"],
        default="none",
    )

    n = len(per_entity)
    dist = per_entity["source_split"].value_counts()
    for label, count in dist.items():
        pct = 100 * count / n if n else 0
        lines.append(f"  {label:20s}: {count:>8,} ({pct:.2f}%)")

    return "\n".join(lines) + "\n"


def country_consistency_in_true_matches(
    exploded: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame
) -> str:
    """
    Rewritten using merge() instead of dict.get() inside a loop.

    Steps:
    1. Attach the Source 1 country to every exploded pair by merging on
       source1_entity_id.
    2. Split the exploded table into an S2 half and an S3 half (since
       each needs to merge against a different source file), attach the
       matched record's country by merging on matched_id, then
       concatenate the two halves back together.
    3. Compare the two country columns with a single vectorized
       string comparison.

    This replaces two nested lookups per row with two merge operations
    over the whole table at once.
    """
    lines = ["=== Country Consistency Within True-Match Pairs ===\n"]

    s1_country = s1[[COL_ENTITY_ID, COL_COUNTRY]].rename(
        columns={COL_ENTITY_ID: COL_GT_SOURCE1_ID, COL_COUNTRY: "country_s1"}
    )
    with_s1_country = exploded.merge(s1_country, on=COL_GT_SOURCE1_ID, how="left")

    s2_country = s2[[COL_ENTITY_ID, COL_COUNTRY]].rename(
        columns={COL_ENTITY_ID: "matched_id", COL_COUNTRY: "country_other"}
    )
    s3_country = s3[[COL_ENTITY_ID, COL_COUNTRY]].rename(
        columns={COL_ENTITY_ID: "matched_id", COL_COUNTRY: "country_other"}
    )

    is_s2 = with_s1_country["matched_id"].str.startswith("S2-")
    s2_half = with_s1_country[is_s2].merge(s2_country, on="matched_id", how="left")
    s3_half = with_s1_country[~is_s2].merge(s3_country, on="matched_id", how="left")

    combined = pd.concat([s2_half, s3_half], ignore_index=True)
    combined = combined.dropna(subset=["country_s1", "country_other"])

    mismatch_mask = (
        combined["country_s1"].str.strip().str.lower()
        != combined["country_other"].str.strip().str.lower()
    )

    n_checked = len(combined)
    n_mismatched = int(mismatch_mask.sum())
    pct = 100 * n_mismatched / n_checked if n_checked else 0

    lines.append(f"True-match pairs checked: {n_checked:,}")
    lines.append(f"Pairs with differing country labels: {n_mismatched:,} ({pct:.2f}%)")
    if n_mismatched:
        examples = combined.loc[
            mismatch_mask,
            [COL_GT_SOURCE1_ID, "country_s1", "matched_id", "country_other"],
        ].head(10)
        lines.append("Examples (s1_id, s1_country, match_id, match_country):")
        for _, row in examples.iterrows():
            lines.append(f"  {tuple(row)}")

    return "\n".join(lines) + "\n"


def main():
    # Loads the three train source files plus ground truth, builds the
    # exploded long-format match table once, then reuses it across all
    # three topology checks so the explode operation (the only
    # meaningfully expensive step here) only runs once per script
    # execution rather than once per check.
    print("Loading files...")
    s1 = load_tsv(TRAIN_SOURCE1)
    s2 = load_tsv(TRAIN_SOURCE2)
    s3 = load_tsv(TRAIN_SOURCE3)
    gt = load_tsv(TRAIN_GROUND_TRUTH)

    report_parts = []
    gt, summary = singleton_and_cardinality(gt)
    report_parts.append(summary)

    exploded = explode_matched_ids(gt)
    report_parts.append(source_split_of_matches(gt, exploded))
    report_parts.append(country_consistency_in_true_matches(exploded, s1, s2, s3))

    full_report = "\n".join(report_parts)
    print(full_report)

    REPORT_PATH.write_text(full_report, encoding="utf-8")
    print(f"\nFull report written to: {REPORT_PATH}")
    print(f"Cardinality distribution CSV written to: {CARDINALITY_CSV}")


if __name__ == "__main__":
    main()