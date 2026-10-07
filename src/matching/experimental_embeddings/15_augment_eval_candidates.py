"""
15_augment_eval_candidates.py

Final step (part B): precomputes embedding cosine-similarity for every
(s1_id, candidate_id) pair implied by eval_candidates_full.tsv, writing
a lookup file the threshold-sweep script can join against. Kept as a
SEPARATE lookup file (not modifying eval_candidates_full.tsv itself)
since that file's format (comma-separated candidate lists) isn't
row-per-pair -- the sweep script reads this lookup alongside the
original file.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    EVAL_CANDIDATES_PATH, EVAL_CANDIDATES_EMBED_SIM_PATH,
    TRAIN_EMBEDDINGS_MEMMAP_PATH, TRAIN_EMBEDDINGS_ID_INDEX_PATH, TRAIN_EMBEDDINGS_META_PATH,
    EMBEDDING_DIM, PROGRESS_EVERY,
)


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
    # Loaded fully into RAM as float16 (half the footprint of float32)
    # rather than accessed as a lazy memmap, since this script performs
    # many repeated random-access lookups per entity -- an in-RAM array
    # keeps those lookups fast and consistent rather than depending on
    # the OS page cache behavior of a memory-mapped file under heavy
    # random access.
    print("  loading embeddings fully into RAM as float16 (halves memory footprint)...")
    embeddings = np.array(
        np.memmap(TRAIN_EMBEDDINGS_MEMMAP_PATH, dtype="float32", mode="r", shape=(total_n, EMBEDDING_DIM)),
        dtype="float16"
    )

    id_index_df = pd.read_csv(TRAIN_EMBEDDINGS_ID_INDEX_PATH, sep="\t", dtype={"entity_id": str, "row_index": int})
    id_to_row = dict(zip(id_index_df["entity_id"], id_index_df["row_index"]))
    print(f"  {len(id_to_row):,} entities indexed")

    print(f"\nStreaming {EVAL_CANDIDATES_PATH.name}, computing pairwise embedding similarity...")

    n_pairs = 0
    n_missing = 0
    start_time = time.time()

    with open(EVAL_CANDIDATES_PATH, "r", encoding="utf-8") as in_f, \
         open(EVAL_CANDIDATES_EMBED_SIM_PATH, "w", encoding="utf-8", newline="") as out_f:

        out_f.write("source1_entity_id\tcandidate_entity_id\tembedding_cosine_sim\n")
        next(in_f)  # header

        for line in in_f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1_id = parts[0]
            cand_str = parts[1] if len(parts) > 1 else ""
            if not cand_str:
                continue

            s1_row = id_to_row.get(s1_id)

            for cand_id in cand_str.split(","):
                cand_row = id_to_row.get(cand_id)
                if s1_row is None or cand_row is None:
                    sim = 0.0
                    n_missing += 1
                else:
                    sim = float(np.dot(embeddings[s1_row].astype("float32"), embeddings[cand_row].astype("float32")))

                out_f.write(f"{s1_id}\t{cand_id}\t{sim:.4f}\n")
                n_pairs += 1

            # flush=True ensures progress is visible immediately in the
            # console, rather than buffered and only appearing in bursts.
            if n_pairs % 25_000 == 0:
                elapsed = time.time() - start_time
                rate = n_pairs / elapsed if elapsed > 0 else 0
                print(f"    {n_pairs:,} pairs processed ({elapsed:.1f}s elapsed, ~{rate:.0f} pairs/sec)", flush=True)

    elapsed = time.time() - start_time
    pct_missing = 100 * n_missing / n_pairs if n_pairs else 0
    print(f"\nDone. {n_pairs:,} pairs processed in {elapsed:.1f}s.")
    print(f"  Pairs with missing embedding (fell back to 0.0): {n_missing:,} ({pct_missing:.2f}%)")
    print(f"  Output: {EVAL_CANDIDATES_EMBED_SIM_PATH}")


if __name__ == "__main__":
    main()