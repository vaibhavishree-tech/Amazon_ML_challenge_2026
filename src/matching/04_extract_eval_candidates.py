"""
04_extract_eval_candidates.py

Step 1 of threshold selection: samples EVAL_N_ENTITIES held-out
(split=="val") Source 1 entities, then pulls their FULL candidate pools
directly from candidates_train.tsv -- not the 4:1 negative-sampled
version used for training. This is necessary for an honest threshold
selection (see config.py for why the sampled training file would
understate real-world negative volume at inference time).

Streams candidates_train.tsv with plain line reading rather than
pandas' chunked reader, for the same reason applied throughout this
project: candidate rows can hold hundreds of comma-separated IDs, too
long for a chunked pandas reader's fixed-buffer approach at this scale.
"""

import random
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAINING_SAMPLE_PATH, CANDIDATES_TRAIN_PATH, EVAL_CANDIDATES_PATH,
    EVAL_SAMPLE_SEED, EVAL_N_ENTITIES, PROGRESS_EVERY,
)
from io_utils import load_tsv


def main():
    random.seed(EVAL_SAMPLE_SEED)

    print("Loading validation entity IDs from training_pairs_sample.tsv...")
    split_df = load_tsv(TRAINING_SAMPLE_PATH, usecols=["source1_entity_id", "split"])
    val_entities = split_df.loc[split_df["split"] == "val", "source1_entity_id"].unique().tolist()
    del split_df
    print(f"  {len(val_entities):,} unique validation entities available")

    sample_size = min(EVAL_N_ENTITIES, len(val_entities))
    sampled_entities = set(random.sample(val_entities, sample_size))
    print(f"  sampled {len(sampled_entities):,} entities for threshold evaluation")

    print(f"\nStreaming {CANDIDATES_TRAIN_PATH.name} to extract full candidate "
          f"pools for sampled entities...")

    n_scanned = 0
    n_matched = 0
    start_time = time.time()

    with open(CANDIDATES_TRAIN_PATH, "r", encoding="utf-8") as in_f, \
         open(EVAL_CANDIDATES_PATH, "w", encoding="utf-8", newline="") as out_f:

        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        next(in_f)  # skip header

        for line in in_f:
            line = line.rstrip("\n")
            if not line:
                continue
            tab_pos = line.find("\t")
            s1_id = line[:tab_pos] if tab_pos != -1 else line

            n_scanned += 1
            if s1_id in sampled_entities:
                out_f.write(line + "\n")
                n_matched += 1

            if n_scanned % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                rate = n_scanned / elapsed if elapsed > 0 else 0
                print(f"    {n_scanned:,} rows scanned "
                      f"({elapsed:.1f}s elapsed, ~{rate:.0f} rows/sec) -- "
                      f"{n_matched:,}/{len(sampled_entities):,} sampled entities found so far")

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s. Found {n_matched:,} of {len(sampled_entities):,} sampled entities.")
    print(f"Full candidate pools written to: {EVAL_CANDIDATES_PATH}")


if __name__ == "__main__":
    main()