"""
19_run_final_inference.py

The experimental embedding-augmented final inference script: the
11-feature model with embedding similarity, at the locked candidate cap
and threshold, run against the real test candidate_pairs.tsv.

STATUS: this script was not successfully completed at full test scale.
It encountered memory constraints when the embedding array and the
required candidate-lookup tables needed to coexist in available system
memory during multiprocessing worker setup. See
experimental_embeddings/README_EXPERIMENTAL.md for the full account of
what was attempted and why the submitted result instead comes from the
10-feature baseline pipeline (matching/06_run_test_inference.py).

Kept here, unmodified, as a complete, working record of the approach --
including its multiprocessing design, file-backed lookup pattern (to
avoid pickling large dictionaries through multiprocessing's initargs
pipe), and file-backed embedding memmap pattern. These techniques are
sound in principle and would very plausibly complete successfully on a
machine with more available RAM than was used for this competition.
"""

import gc
import multiprocessing as mp
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TEST_SOURCE1, CLEANED_TEST_SOURCE2, CLEANED_TEST_SOURCE3,
    RAW_TEST_SOURCE1, RAW_TEST_SOURCE2, RAW_TEST_SOURCE3,
    CANDIDATES_TEST_PATH, MATCHING_RESULTS_PATH, MODEL_V2_PATH,
    TEST_EMBEDDINGS_MEMMAP_PATH, TEST_EMBEDDINGS_ID_INDEX_PATH, TEST_EMBEDDINGS_META_PATH,
    TEST_EMBED_REFS_S1_PATH, TEST_EMBED_REFS_OTHER_PATH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE,
    EMBEDDING_DIM, N_WORKERS, WORKER_CHUNK_SIZE, FINAL_CAP, FINAL_THRESHOLD,
)
from io_utils import load_tsv
from parallel_worker_v2 import init_worker_v2, score_and_predict_batch


def load_ref_ids(path):
    """Reads a newline-separated file of entity IDs into a set."""
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def build_lookup(df, include_postal):
    """entity_id -> (name, address, postal_code) or entity_id -> name, built via dict(zip(...))."""
    if include_postal:
        return dict(zip(df[COL_ENTITY_ID], zip(df[COL_NAME], df[COL_ADDRESS], df[COL_POSTAL_CODE])))
    return dict(zip(df[COL_ENTITY_ID], df[COL_NAME]))


def chunk_list(items, chunk_size):
    """Yields successive fixed-size chunks from a list, for dispatching work to the worker pool."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def main():
    print("Loading referenced ID sets from Step A scan...")
    s1_ref_ids = load_ref_ids(TEST_EMBED_REFS_S1_PATH)
    other_ref_ids = load_ref_ids(TEST_EMBED_REFS_OTHER_PATH)
    print(f"  {len(s1_ref_ids):,} S1 refs, {len(other_ref_ids):,} other refs")

    # --- Step 1: build ALL lookups first, nothing pickled yet ---

    print("\nLoading cleaned test source files, filtering to referenced IDs...")
    s1_df = load_tsv(CLEANED_TEST_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s2_df = load_tsv(CLEANED_TEST_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s3_df = load_tsv(CLEANED_TEST_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s1_df = s1_df[s1_df[COL_ENTITY_ID].isin(s1_ref_ids)]
    s2_df = s2_df[s2_df[COL_ENTITY_ID].isin(other_ref_ids)]
    s3_df = s3_df[s3_df[COL_ENTITY_ID].isin(other_ref_ids)]

    s1_lookup = build_lookup(s1_df, True)
    other_lookup = {}
    other_lookup.update(build_lookup(s2_df, True))
    other_lookup.update(build_lookup(s3_df, True))
    del s1_df, s2_df, s3_df
    gc.collect()
    print(f"  s1_lookup={len(s1_lookup):,}, other_lookup={len(other_lookup):,}")

    print("\nLoading RAW test source2/3 names (non-ASCII flag), filtering...")
    raw_s2_df = load_tsv(RAW_TEST_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s3_df = load_tsv(RAW_TEST_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s2_df = raw_s2_df[raw_s2_df[COL_ENTITY_ID].isin(other_ref_ids)]
    raw_s3_df = raw_s3_df[raw_s3_df[COL_ENTITY_ID].isin(other_ref_ids)]
    raw_name_lookup = {}
    raw_name_lookup.update(build_lookup(raw_s2_df, False))
    raw_name_lookup.update(build_lookup(raw_s3_df, False))
    del raw_s2_df, raw_s3_df
    gc.collect()
    print(f"  raw_name_lookup={len(raw_name_lookup):,}")

    print("\nLoading test embeddings ID index (full, lightweight)...")
    id_index_df = pd.read_csv(TEST_EMBEDDINGS_ID_INDEX_PATH, sep="\t", dtype={"entity_id": str, "row_index": int})
    embed_id_to_row = dict(zip(id_index_df["entity_id"], id_index_df["row_index"]))
    del id_index_df
    gc.collect()
    print(f"  embed_id_to_row={len(embed_id_to_row):,} entities indexed")

    # --- Step 2: NOW that all four dicts exist, write them all to disk ---

    LOOKUP_DUMP_DIR = MATCHING_RESULTS_PATH.parent / "lookup_dumps"
    LOOKUP_DUMP_DIR.mkdir(exist_ok=True)

    s1_lookup_path = LOOKUP_DUMP_DIR / "s1_lookup.pkl"
    other_lookup_path = LOOKUP_DUMP_DIR / "other_lookup.pkl"
    raw_name_lookup_path = LOOKUP_DUMP_DIR / "raw_name_lookup.pkl"
    embed_id_to_row_path = LOOKUP_DUMP_DIR / "embed_id_to_row.pkl"

    # Lookups are written to disk, and each worker loads its own copy
    # from these files during pool startup, rather than passing the
    # dictionaries directly through multiprocessing's initargs pipe --
    # this avoids the memory spike of pickling large in-memory
    # dictionaries for transmission to each spawned worker process.
    print("\nWriting lookups to disk (avoids pickling large dicts through "
          "multiprocessing pipes)...")
    with open(s1_lookup_path, "wb") as f:
        pickle.dump(s1_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(other_lookup_path, "wb") as f:
        pickle.dump(other_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(raw_name_lookup_path, "wb") as f:
        pickle.dump(raw_name_lookup, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(embed_id_to_row_path, "wb") as f:
        pickle.dump(embed_id_to_row, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("  all lookups written to disk.")

    # Free the in-memory copies now that they're safely on disk -- workers
    # will load their own copies from these files, so the main process
    # doesn't need to keep holding them.
    del s1_lookup, other_lookup, raw_name_lookup, embed_id_to_row
    gc.collect()

    # --- Step 3: embeddings, converted to a file-backed float16 memmap ---

    with open(TEST_EMBEDDINGS_META_PATH, "r") as f:
        meta = dict(line.strip().split("=") for line in f)
    total_n = int(meta["total_n"])

    FLOAT16_EMBED_PATH = TEST_EMBEDDINGS_MEMMAP_PATH.parent / "test_embeddings_f16.dat"
    print(f"\nConverting test embeddings to a FILE-BACKED float16 memmap "
          f"(shape=({total_n:,}, {EMBEDDING_DIM}), ~{total_n*EMBEDDING_DIM*2/1e9:.2f} GB, "
          f"path={FLOAT16_EMBED_PATH})...")

    embeddings_f32_memmap = np.memmap(TEST_EMBEDDINGS_MEMMAP_PATH, dtype="float32", mode="r", shape=(total_n, EMBEDDING_DIM))
    embed_f16_memmap = np.memmap(FLOAT16_EMBED_PATH, dtype="float16", mode="w+", shape=(total_n, EMBEDDING_DIM))

    conv_batch = 500_000
    for start in range(0, total_n, conv_batch):
        end = min(start + conv_batch, total_n)
        embed_f16_memmap[start:end] = embeddings_f32_memmap[start:end].astype("float16")
        if start % 2_000_000 == 0:
            print(f"    converted {end:,}/{total_n:,}", flush=True)
    embed_f16_memmap.flush()
    del embed_f16_memmap
    del embeddings_f32_memmap
    gc.collect()
    print("  file-backed float16 embeddings ready.")

    # --- Step 4: run the pool, passing only PATHS (tiny strings) through initargs ---

    print(f"\nReading {CANDIDATES_TEST_PATH.name} into memory for chunked dispatch...")
    with open(CANDIDATES_TEST_PATH, "r", encoding="utf-8") as f:
        next(f)
        all_lines = f.readlines()
    print(f"  {len(all_lines):,} lines loaded")

    chunks = list(chunk_list(all_lines, WORKER_CHUNK_SIZE))
    print(f"\nStarting pool of {N_WORKERS} workers, {len(chunks)} chunks, "
          f"cap={FINAL_CAP}, threshold={FINAL_THRESHOLD}...")

    n_written = 0
    n_with_match = 0
    start_time = time.time()

    with mp.Pool(
        processes=N_WORKERS,
        initializer=init_worker_v2,
        initargs=(str(s1_lookup_path), str(other_lookup_path), str(raw_name_lookup_path),
                  str(MODEL_V2_PATH), FINAL_CAP, FINAL_THRESHOLD, str(embed_id_to_row_path),
                  str(FLOAT16_EMBED_PATH), (total_n, EMBEDDING_DIM)),
    ) as pool:
        with open(MATCHING_RESULTS_PATH, "w", encoding="utf-8", newline="") as out_f:
            out_f.write("source1_entity_id\tmatched_entity_ids\n")

            for batch_results in pool.imap_unordered(score_and_predict_batch, chunks):
                for s1_id, matched_str in batch_results:
                    out_f.write(f"{s1_id}\t{matched_str}\n")
                    n_written += 1
                    if matched_str:
                        n_with_match += 1

                if n_written % 50_000 < WORKER_CHUNK_SIZE:
                    elapsed = time.time() - start_time
                    rate = n_written / elapsed if elapsed > 0 else 0
                    eta = (len(all_lines) - n_written) / rate if rate > 0 else 0
                    print(f"    {n_written:,}/{len(all_lines):,} entities written "
                          f"({elapsed:.0f}s elapsed, ~{rate:.0f}/sec, ETA ~{eta/60:.0f} min)", flush=True)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s ({elapsed/3600:.2f} hours).")
    print(f"Total entities written: {n_written:,}")
    print(f"  with a match:  {n_with_match:,}")
    print(f"  singletons:    {n_written - n_with_match:,}")
    print(f"\nmatching_results.tsv written to: {MATCHING_RESULTS_PATH}")


if __name__ == "__main__":
    main()