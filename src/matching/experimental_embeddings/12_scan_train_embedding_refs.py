"""
12_scan_train_embedding_refs.py

Step A: scans training_pairs_sample.tsv (via reservoir sampling, capped
at TRAINING_SAMPLE_SUBSET_SIZE rows) and eval_candidates_full.tsv (in
full) to find every UNIQUE entity ID that will need an embedding,
without embedding anything yet.

Why reservoir sampling for training_pairs_sample.tsv: a full scan of
all rows in that file references entities at roughly the same scale as
the full test-set embedding job, which is a substantial time cost on
its own. XGBoost does not need every training row's entities
individually embedded to learn the new feature's pattern -- a
representative subset (reservoir sampling preserves the file's natural
row order and composition, including its positive:negative ratio)
gives the model plenty of signal at a fraction of the entity count and
embedding time.

eval_candidates_full.tsv is scanned IN FULL, unchanged -- this file is
what the country-specific threshold sweep depends on for an accurate
estimate, so its scope is not reduced.

Writes the referenced ID sets to disk (S1 IDs and "other" S2/S3 IDs
separately), so 13_build_train_embeddings.py can read them directly
without re-scanning.
"""

import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAINING_SAMPLE_PATH, EVAL_CANDIDATES_PATH,
    TRAIN_EMBED_REFS_S1_PATH, TRAIN_EMBED_REFS_OTHER_PATH,
)

TRAINING_SAMPLE_SUBSET_SIZE = 1_000_000  # reservoir-sample size, reduced from the full training sample
RANDOM_SEED = 42


def main():
    s1_ids = set()
    other_ids = set()

    print(f"Scanning {TRAINING_SAMPLE_PATH.name}, reservoir-sampling down to "
          f"{TRAINING_SAMPLE_SUBSET_SIZE:,} rows...")
    random.seed(RANDOM_SEED)
    reservoir = []
    with open(TRAINING_SAMPLE_PATH, "r", encoding="utf-8") as f:
        next(f)  # header
        for i, line in enumerate(f):
            if len(reservoir) < TRAINING_SAMPLE_SUBSET_SIZE:
                reservoir.append(line)
            else:
                j = random.randint(0, i)
                if j < TRAINING_SAMPLE_SUBSET_SIZE:
                    reservoir[j] = line

    for line in reservoir:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2:
            continue
        s1_ids.add(parts[0])
        if parts[1]:
            other_ids.add(parts[1])
    print(f"  sampled {len(reservoir):,} rows. Running totals: "
          f"{len(s1_ids):,} S1 IDs, {len(other_ids):,} other IDs")

    print(f"\nScanning {EVAL_CANDIDATES_PATH.name} (comma-separated candidates per row, FULL scan)...")
    with open(EVAL_CANDIDATES_PATH, "r", encoding="utf-8") as f:
        next(f)  # header
        n = 0
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 1:
                continue
            s1_ids.add(parts[0])
            cand_str = parts[1] if len(parts) > 1 else ""
            if cand_str:
                other_ids.update(cand_str.split(","))
            n += 1
    print(f"  {n:,} rows scanned.")

    print(f"\n=== FINAL REFERENCED COUNTS ===")
    print(f"  Unique S1 entities:    {len(s1_ids):,}")
    print(f"  Unique S2/S3 entities: {len(other_ids):,}")
    print(f"  TOTAL to embed:        {len(s1_ids) + len(other_ids):,}")

    with open(TRAIN_EMBED_REFS_S1_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(s1_ids))
    with open(TRAIN_EMBED_REFS_OTHER_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(other_ids))

    print(f"\nReference ID lists written to:")
    print(f"  {TRAIN_EMBED_REFS_S1_PATH}")
    print(f"  {TRAIN_EMBED_REFS_OTHER_PATH}")

    total = len(s1_ids) + len(other_ids)
    est_seconds = total / 950
    print(f"\nAt a measured rate of ~950 entities/sec, this would take roughly "
          f"{est_seconds/3600:.2f} hours to embed. Decide whether to proceed with "
          f"the full set or reduce scope before running 13_build_train_embeddings.py.")


if __name__ == "__main__":
    main()