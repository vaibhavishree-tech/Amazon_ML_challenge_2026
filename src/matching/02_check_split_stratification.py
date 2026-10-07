"""
02_check_split_stratification.py

Verifies whether the random (non-stratified) train/val split in
training_pairs_sample.tsv is balanced across match-cardinality buckets
(how many true matches each entity has), or whether it happens to be
meaningfully skewed.

Why this matters: 00_prepare_training_sample.py's split is a random
hash-based assignment per entity, not an explicitly stratified split.
This script checks, after the fact, whether that random assignment
landed close to representative -- a large skew in any bucket would mean
the validation set does not reflect training conditions well, which
could bias later threshold-selection decisions.

Reads training_pairs_sample.tsv directly rather than the full feature
table, since this check only needs entity_id, label, and split columns.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import TRAINING_SAMPLE_PATH
from io_utils import load_tsv


def cardinality_bucket(n_true_matches: int) -> str:
    """Buckets a raw match count into the same 0/1/2/3+ categories used in the EDA phase's ground-truth topology report."""
    if n_true_matches == 0:
        return "0 (singleton)"
    if n_true_matches == 1:
        return "1"
    if n_true_matches == 2:
        return "2"
    return "3+"


def main():
    print("Loading training_pairs_sample.tsv...")
    df = load_tsv(TRAINING_SAMPLE_PATH)

    # Count TRUE matches (label == "1") per source1_entity_id -- this
    # reconstructs each entity's real match cardinality directly from
    # the labeled pairs, without needing to re-read ground truth.
    positives = df[df["label"] == "1"]
    match_counts = positives.groupby("source1_entity_id").size()

    # Entities with ZERO positives in this sample are singletons
    # (either truly no match, or blocking missed all their matches --
    # either way, zero true-match rows for them here).
    all_entities = df[["source1_entity_id", "split"]].drop_duplicates()
    all_entities["n_true_matches"] = all_entities["source1_entity_id"].map(match_counts).fillna(0).astype(int)
    all_entities["bucket"] = all_entities["n_true_matches"].apply(cardinality_bucket)

    print("\n=== Cardinality Bucket Proportions: Train vs Validation ===\n")
    for split_name in ["train", "val"]:
        split_df = all_entities[all_entities["split"] == split_name]
        total = len(split_df)
        print(f"-- {split_name} (n={total:,} entities) --")
        dist = split_df["bucket"].value_counts(normalize=True).sort_index() * 100
        for bucket, pct in dist.items():
            print(f"  {bucket:15s}: {pct:.2f}%")
        print()

    print("If the percentages for each bucket are close between train and val "
          "(within a few percentage points), the random split is effectively "
          "representative and safe to use as-is. A large gap in any bucket "
          "means validation may not reflect training conditions well.")


if __name__ == "__main__":
    main()