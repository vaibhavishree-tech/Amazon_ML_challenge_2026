"""
14_augment_training_features.py

Final step (part A): joins the embedding cosine-similarity feature into
training_features.tsv, producing training_features_v2.tsv with all 11
features. Does NOT recompute the original 10 features -- only adds the
new column, keyed by (source1_entity_id, candidate_entity_id).

Reads embeddings from the train-side memmap (built by
13_build_train_embeddings.py, covering S1 entities and the sampled
training candidates) -- test-side embeddings are not needed here, since
this file only contains training pairs.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAINING_FEATURES_PATH, TRAINING_FEATURES_AUGMENTED_PATH,
    TRAIN_EMBEDDINGS_MEMMAP_PATH, TRAIN_EMBEDDINGS_ID_INDEX_PATH, TRAIN_EMBEDDINGS_META_PATH,
    EMBEDDING_DIM, FEATURE_BATCH_SIZE, PROGRESS_EVERY,
)
from text_similarity_features import FEATURE_COLUMNS


def load_meta(path):
    """Reads a simple key=value metadata file into a dict."""
    meta = {}
    with open(path, "r") as f:
        for line in f:
            k, v = line.strip().split("=")
            meta[k] = v
    return meta


def main():
    print("Loading train embeddings metadata and ID index...")
    meta = load_meta(TRAIN_EMBEDDINGS_META_PATH)
    total_n = int(meta["total_n"])

    embeddings = np.memmap(TRAIN_EMBEDDINGS_MEMMAP_PATH, dtype="float32", mode="r", shape=(total_n, EMBEDDING_DIM))

    id_index_df = pd.read_csv(TRAIN_EMBEDDINGS_ID_INDEX_PATH, sep="\t", dtype={"entity_id": str, "row_index": int})
    id_to_row = dict(zip(id_index_df["entity_id"], id_index_df["row_index"]))
    print(f"  {len(id_to_row):,} entities indexed, embedding matrix shape={embeddings.shape}")

    print(f"\nStreaming {TRAINING_FEATURES_PATH.name}, computing embedding similarity, "
          f"writing {TRAINING_FEATURES_AUGMENTED_PATH.name}...")

    n_processed = 0
    n_missing_embedding = 0
    start_time = time.time()

    with open(TRAINING_FEATURES_PATH, "r", encoding="utf-8") as in_f, \
         open(TRAINING_FEATURES_AUGMENTED_PATH, "w", encoding="utf-8", newline="") as out_f:

        header = in_f.readline().rstrip("\n")
        out_f.write(header + "\tname_address_embedding_cosine_sim\n")

        for line in in_f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id, cand_id = parts[0], parts[1]

            s1_row = id_to_row.get(s1_id)
            cand_row = id_to_row.get(cand_id)

            if s1_row is None or cand_row is None:
                # Expected to be rare: occurs only if this pair's
                # entities fell outside the reservoir sample used to
                # decide what to embed. Falls back to 0.0 similarity (a
                # neutral value) rather than failing -- the model simply
                # receives slightly less signal for these specific rows.
                sim = 0.0
                n_missing_embedding += 1
            else:
                vec_a = embeddings[s1_row]
                vec_b = embeddings[cand_row]
                sim = float(np.dot(vec_a, vec_b))  # both vectors are pre-normalized, so dot == cosine

            out_f.write(f"{line}\t{sim:.4f}\n")
            n_processed += 1

            if n_processed % PROGRESS_EVERY == 0:
                elapsed = time.time() - start_time
                rate = n_processed / elapsed if elapsed > 0 else 0
                print(f"    {n_processed:,} rows processed ({elapsed:.1f}s elapsed, ~{rate:.0f} rows/sec)")

    elapsed = time.time() - start_time
    pct_missing = 100 * n_missing_embedding / n_processed if n_processed else 0
    print(f"\nDone. {n_processed:,} rows processed in {elapsed:.1f}s.")
    print(f"  Rows with missing embedding (fell back to 0.0): {n_missing_embedding:,} ({pct_missing:.2f}%)")
    print(f"  Output: {TRAINING_FEATURES_AUGMENTED_PATH}")


if __name__ == "__main__":
    main()