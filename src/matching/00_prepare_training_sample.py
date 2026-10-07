"""
00_prepare_training_sample.py

Step 0: builds a labeled, manageably-sized sample of candidate pairs
from candidates_train.tsv, for the feature engineering and model
training steps that follow.

Why this step exists: the full training candidate set contains over a
billion rows -- computing similarity features for every single one is
unnecessary and would cost far more time than a baseline model needs.
A model only requires enough labeled examples to learn the decision
boundary, not every example that exists.

What this script does, in one streaming pass over candidates_train.tsv:
  1. Loads the ground truth and builds the true-pairs set.
  2. For every (source1_entity_id, candidate_entity_id) pair in the
     candidate file:
       - if it's a true match -> ALWAYS KEEP, label=1
       - otherwise -> keep with probability NEGATIVE_SAMPLE_PROBABILITY
         (precomputed in config.py from known dataset-wide totals),
         label=0
  3. Assigns each row to "train" or "val" based on a deterministic hash
     of its source1_entity_id -- so one entity's candidates never split
     across both sets, which would leak information between them.
  4. Writes everything to one output TSV with a split column, ready for
     the next script to read and compute features on.

Uses plain line-by-line file reading rather than pandas' chunked
reader. Candidate rows can be long (up to hundreds of comma-separated
IDs), and a chunked pandas reader must build one large contiguous
buffer per chunk before processing can begin -- at this row length,
that buffer allocation becomes impractical. Plain line reading has no
such requirement; memory use depends only on one line at a time.
"""

import hashlib
import random
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_GROUND_TRUTH, CANDIDATES_TRAIN_PATH, TRAINING_SAMPLE_PATH,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    NEGATIVE_SAMPLE_PROBABILITY, VALIDATION_SPLIT_PERCENT, PROGRESS_EVERY,
)
from io_utils import load_tsv

RANDOM_SEED = 42


def explode_matched_ids(gt: pd.DataFrame) -> pd.DataFrame:
    """
    Same explode pattern used throughout the EDA, cleaning, and
    blocking phases. This runs on the ground truth file (not the much
    larger candidate file), a size this operation handles comfortably.
    """
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"}).reset_index(drop=True)


def assign_split(entity_id: str) -> str:
    """
    Deterministic hash-based train/val assignment. Using a hash (rather
    than a per-call random draw) means the SAME entity always lands in
    the same split even if this script is re-run -- important both for
    reproducibility and so later phases can re-derive which split an
    entity belongs to without needing to re-run this script.
    """
    digest = hashlib.md5(entity_id.encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % 100
    return "val" if bucket < VALIDATION_SPLIT_PERCENT else "train"


def main():
    random.seed(RANDOM_SEED)

    print("Loading ground truth and building true-pairs set...")
    gt = load_tsv(TRAIN_GROUND_TRUTH)
    exploded = explode_matched_ids(gt)
    true_pairs_set = set(zip(exploded[COL_GT_SOURCE1_ID], exploded["matched_id"]))
    print(f"  {len(true_pairs_set):,} true match pairs loaded")
    del gt, exploded

    print(f"\nNegative sample probability (precomputed): {NEGATIVE_SAMPLE_PROBABILITY:.5f}")
    print(f"Streaming through {CANDIDATES_TRAIN_PATH.name}...")

    n_entities = 0
    n_positives_kept = 0
    n_negatives_kept = 0
    n_negatives_seen = 0
    start_time = time.time()

    with open(CANDIDATES_TRAIN_PATH, "r", encoding="utf-8") as in_f, \
         open(TRAINING_SAMPLE_PATH, "w", encoding="utf-8", newline="") as out_f:

        out_f.write("source1_entity_id\tcandidate_entity_id\tlabel\tsplit\n")

        header = in_f.readline()  # skip header row

        for line in in_f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                s1_id, cand_str = parts[0], ""
            else:
                s1_id, cand_str = parts[0], parts[1]

            n_entities += 1

            if not cand_str:
                continue  # no candidates for this entity -- nothing to sample from

            split = assign_split(s1_id)

            for cand_id in cand_str.split(","):
                pair = (s1_id, cand_id)
                if pair in true_pairs_set:
                    out_f.write(f"{s1_id}\t{cand_id}\t1\t{split}\n")
                    n_positives_kept += 1
                else:
                    n_negatives_seen += 1
                    if random.random() < NEGATIVE_SAMPLE_PROBABILITY:
                        out_f.write(f"{s1_id}\t{cand_id}\t0\t{split}\n")
                        n_negatives_kept += 1

            if n_entities % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                rate = n_entities / elapsed if elapsed > 0 else 0
                print(f"    {n_entities:,} entities processed "
                      f"({elapsed:.1f}s elapsed, ~{rate:.0f} entities/sec) -- "
                      f"positives kept: {n_positives_kept:,}, negatives kept: {n_negatives_kept:,}")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s.")
    print(f"Total entities processed: {n_entities:,}")
    print(f"Positives kept (all true matches found in candidates): {n_positives_kept:,}")
    print(f"Negatives seen: {n_negatives_seen:,}")
    print(f"Negatives kept (sampled at {NEGATIVE_SAMPLE_PROBABILITY:.5f}): {n_negatives_kept:,}")
    actual_ratio = n_negatives_kept / n_positives_kept if n_positives_kept else 0
    print(f"Actual negative:positive ratio achieved: {actual_ratio:.2f} "
          f"(target was {n_positives_kept and NEGATIVE_SAMPLE_PROBABILITY and round(actual_ratio):.0f}ish)")
    print(f"\nOutput written to: {TRAINING_SAMPLE_PATH}")


if __name__ == "__main__":
    main()