"""
11_build_test_embeddings.py

Embeds every entity in the RAW (uncleaned) test_source1/2/3 files, once
each, using the tagged raw-text format from embedding_utils.py, and
stores the results as a disk-backed memory-mapped float32 array plus a
separate entity_id -> row_index lookup.

Why RAW text, not cleaned: this is restated here because it is the
single most important design decision in this feature. Cleaning's
unidecode transliteration already destroys the semantic content of
non-ASCII business names (confirmed during blocking validation: name
token-overlap similarity collapsed sharply for that subset after
cleaning). Feeding the already-transliterated cleaned text into a
multilingual embedding model would defeat the entire purpose -- the
model needs to see the original Devanagari/Tamil/Gujarati script
directly to recognise its semantic equivalence to the English name.

Why memory-mapped, not an in-RAM array or dict: with roughly 11-12
million entities (combined across all three test sources) at 384
dimensions in float32, the full embedding matrix is on the order of
17-18 GB -- far more than comfortably fits in RAM alongside everything
else this project runs. np.memmap keeps the array on disk and only
pages in the rows actually being read or written at any moment, the
same "don't hold more than needed in memory at once" discipline used
throughout this project, applied here via numpy's built-in mechanism
instead of manual batching.

Why one full embedding pass instead of filtering to only "referenced"
candidates first: scanning the full candidate_pairs.tsv just to collect
referenced IDs is itself a large, slow operation. Since every unique
test entity needs embedding at most ONCE regardless of how many
candidate pairs it appears in, embedding the full raw source files
directly is both simpler and no more expensive in total embedding
calls than the alternative -- it just moves the cost to a predictable,
one-time pass over three known-size files instead of an extra pass over
a much larger, harder-to-bound one.

This script is designed to be started and left running unattended for
an extended period -- it prints progress regularly so you can check on
it without needing to guess whether it has stalled.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    RAW_TEST_SOURCE1, RAW_TEST_SOURCE2, RAW_TEST_SOURCE3,
    TEST_EMBEDDINGS_MEMMAP_PATH, TEST_EMBEDDINGS_ID_INDEX_PATH, TEST_EMBEDDINGS_META_PATH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY,
    EMBEDDING_DIM, EMBEDDING_BATCH_SIZE, EMBEDDING_PROGRESS_EVERY,
)
from io_utils import load_tsv
from embedding_utils import load_embedding_model, build_tagged_text, encode_texts


def main():
    model = load_embedding_model()

    print("\nLoading raw (uncleaned) test source files...")
    s1_df = load_tsv(RAW_TEST_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY])
    s2_df = load_tsv(RAW_TEST_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY])
    s3_df = load_tsv(RAW_TEST_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY])

    n1, n2, n3 = len(s1_df), len(s2_df), len(s3_df)
    total_n = n1 + n2 + n3
    print(f"  test_source1: {n1:,}, test_source2: {n2:,}, test_source3: {n3:,}, "
          f"total entities to embed: {total_n:,}")

    print(f"\nCreating memory-mapped array: shape=({total_n:,}, {EMBEDDING_DIM}), "
          f"dtype=float32, path={TEST_EMBEDDINGS_MEMMAP_PATH}")
    print(f"  estimated disk size: {total_n * EMBEDDING_DIM * 4 / 1e9:.1f} GB")
    embeddings_memmap = np.memmap(
        TEST_EMBEDDINGS_MEMMAP_PATH, dtype="float32", mode="w+", shape=(total_n, EMBEDDING_DIM)
    )

    id_index_lines = []  # (entity_id, row_index) pairs, written out at the end in one pass
    row_offset = 0
    start_time = time.time()
    n_embedded_total = 0

    def embed_dataframe(df, source_label, row_offset):
        """
        Embeds one source file's entities in batches, writing directly
        into the memmap at the correct row range. Returns the new row
        offset after this source is done.
        """
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
                      f"({elapsed:.0f}s elapsed, ~{rate:.0f} entities/sec, "
                      f"ETA ~{eta_seconds/60:.0f} min)")

        return row_offset + n_rows

    print("\nEmbedding test_source1...")
    row_offset = embed_dataframe(s1_df, "source1", row_offset)
    del s1_df

    print("\nEmbedding test_source2...")
    row_offset = embed_dataframe(s2_df, "source2", row_offset)
    del s2_df

    print("\nEmbedding test_source3...")
    row_offset = embed_dataframe(s3_df, "source3", row_offset)
    del s3_df

    print("\nFlushing memmap to disk...")
    embeddings_memmap.flush()
    del embeddings_memmap

    print(f"Writing ID -> row index lookup ({len(id_index_lines):,} entries)...")
    with open(TEST_EMBEDDINGS_ID_INDEX_PATH, "w", encoding="utf-8") as f:
        f.write("entity_id\trow_index\n")
        f.write("\n".join(id_index_lines))
        f.write("\n")

    with open(TEST_EMBEDDINGS_META_PATH, "w") as f:
        f.write(f"total_n={total_n}\n")
        f.write(f"dim={EMBEDDING_DIM}\n")
        f.write(f"dtype=float32\n")

    elapsed = time.time() - start_time
    print(f"\nDone. Embedded {total_n:,} entities in {elapsed:.0f}s ({elapsed/3600:.2f} hours).")
    print(f"  Memmap:    {TEST_EMBEDDINGS_MEMMAP_PATH}")
    print(f"  ID index:  {TEST_EMBEDDINGS_ID_INDEX_PATH}")
    print(f"  Metadata:  {TEST_EMBEDDINGS_META_PATH}")


if __name__ == "__main__":
    main()