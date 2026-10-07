"""
13_build_train_embeddings.py

Step B: embeds every entity ID referenced in train_embed_refs_s1.txt
and train_embed_refs_other.txt (produced by
12_scan_train_embedding_refs.py), using RAW (uncleaned) text from
train_source1/2/3.tsv, the same tagged format and embedding model used
for the test-side embeddings.

Uses the same memmap + ID-index storage pattern as
11_build_test_embeddings.py, scoped to the referenced train-side
entities rather than the full test-side entity set.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    RAW_TRAIN_SOURCE1, RAW_TRAIN_SOURCE2, RAW_TRAIN_SOURCE3,
    TRAIN_EMBED_REFS_S1_PATH, TRAIN_EMBED_REFS_OTHER_PATH,
    TRAIN_EMBEDDINGS_MEMMAP_PATH, TRAIN_EMBEDDINGS_ID_INDEX_PATH, TRAIN_EMBEDDINGS_META_PATH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY,
    EMBEDDING_DIM, EMBEDDING_BATCH_SIZE, EMBEDDING_PROGRESS_EVERY,
)
from embedding_utils import load_embedding_model, build_tagged_text, encode_texts


def load_ref_ids(path):
    """Reads a newline-separated file of entity IDs into a set."""
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def main():
    model = load_embedding_model()

    print("\nLoading reference ID sets...")
    s1_ref_ids = load_ref_ids(TRAIN_EMBED_REFS_S1_PATH)
    other_ref_ids = load_ref_ids(TRAIN_EMBED_REFS_OTHER_PATH)
    print(f"  {len(s1_ref_ids):,} S1 refs, {len(other_ref_ids):,} other refs")

    print("\nLoading raw train source files and filtering to referenced IDs...")
    s1_df = pd.read_csv(RAW_TRAIN_SOURCE1, sep="\t", dtype=str, keep_default_na=False,
                        usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY])
    s2_df = pd.read_csv(RAW_TRAIN_SOURCE2, sep="\t", dtype=str, keep_default_na=False,
                        usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY])
    s3_df = pd.read_csv(RAW_TRAIN_SOURCE3, sep="\t", dtype=str, keep_default_na=False,
                        usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY])

    s1_df = s1_df[s1_df[COL_ENTITY_ID].isin(s1_ref_ids)].reset_index(drop=True)
    s2_df = s2_df[s2_df[COL_ENTITY_ID].isin(other_ref_ids)].reset_index(drop=True)
    s3_df = s3_df[s3_df[COL_ENTITY_ID].isin(other_ref_ids)].reset_index(drop=True)

    n1, n2, n3 = len(s1_df), len(s2_df), len(s3_df)
    total_n = n1 + n2 + n3
    print(f"  filtered: s1={n1:,}, s2={n2:,}, s3={n3:,}, total to embed={total_n:,}")

    print(f"\nCreating memory-mapped array: shape=({total_n:,}, {EMBEDDING_DIM}), "
          f"estimated disk size: {total_n * EMBEDDING_DIM * 4 / 1e9:.2f} GB")
    embeddings_memmap = np.memmap(
        TRAIN_EMBEDDINGS_MEMMAP_PATH, dtype="float32", mode="w+", shape=(total_n, EMBEDDING_DIM)
    )

    id_index_lines = []
    row_offset = 0
    start_time = time.time()
    n_embedded_total = 0

    def embed_dataframe(df, source_label, row_offset):
        """Embeds one source file's (referenced subset of) entities in batches, writing directly into the memmap."""
        nonlocal n_embedded_total
        n_rows = len(df)
        ids = df[COL_ENTITY_ID].tolist()
        names = df[COL_NAME].tolist()
        addresses = df[COL_ADDRESS].tolist()
        countries = df[COL_COUNTRY].tolist()

        for batch_start in range(0, n_rows, EMBEDDING_BATCH_SIZE):
            batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, n_rows)
            batch_texts = [
                build_tagged_text(names[i], addresses[i], countries[i])
                for i in range(batch_start, batch_end)
            ]
            batch_embeddings = encode_texts(model, batch_texts, batch_size=EMBEDDING_BATCH_SIZE)

            write_start = row_offset + batch_start
            write_end = row_offset + batch_end
            embeddings_memmap[write_start:write_end] = batch_embeddings

            for i in range(batch_start, batch_end):
                id_index_lines.append(f"{ids[i]}\t{row_offset + i}")

            n_embedded_total += (batch_end - batch_start)
            if n_embedded_total % EMBEDDING_PROGRESS_EVERY < EMBEDDING_BATCH_SIZE:
                elapsed = time.time() - start_time
                rate = n_embedded_total / elapsed if elapsed > 0 else 0
                eta_seconds = (total_n - n_embedded_total) / rate if rate > 0 else 0
                print(f"    [{source_label}] {n_embedded_total:,}/{total_n:,} embedded "
                      f"({elapsed:.0f}s elapsed, ~{rate:.0f}/sec, ETA ~{eta_seconds/60:.0f} min)")

        return row_offset + n_rows

    print("\nEmbedding train_source1 (referenced subset)...")
    row_offset = embed_dataframe(s1_df, "source1", row_offset)
    del s1_df

    print("\nEmbedding train_source2 (referenced subset)...")
    row_offset = embed_dataframe(s2_df, "source2", row_offset)
    del s2_df

    print("\nEmbedding train_source3 (referenced subset)...")
    row_offset = embed_dataframe(s3_df, "source3", row_offset)
    del s3_df

    print("\nFlushing memmap to disk...")
    embeddings_memmap.flush()
    del embeddings_memmap

    print(f"Writing ID -> row index lookup ({len(id_index_lines):,} entries)...")
    with open(TRAIN_EMBEDDINGS_ID_INDEX_PATH, "w", encoding="utf-8") as f:
        f.write("entity_id\trow_index\n")
        f.write("\n".join(id_index_lines))
        f.write("\n")

    with open(TRAIN_EMBEDDINGS_META_PATH, "w") as f:
        f.write(f"total_n={total_n}\ndim={EMBEDDING_DIM}\ndtype=float32\n")

    elapsed = time.time() - start_time
    print(f"\nDone. Embedded {total_n:,} entities in {elapsed:.0f}s ({elapsed/3600:.2f} hours).")
    print(f"  Memmap:   {TRAIN_EMBEDDINGS_MEMMAP_PATH}")
    print(f"  ID index: {TRAIN_EMBEDDINGS_ID_INDEX_PATH}")


if __name__ == "__main__":
    main()