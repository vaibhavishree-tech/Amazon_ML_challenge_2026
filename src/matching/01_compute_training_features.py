"""
01_compute_training_features.py

Step 1: computes similarity features for every labeled pair in
training_pairs_sample.tsv (built by 00_prepare_training_sample.py),
producing the final feature table used to train the model.

Design choices:

  - Lookups built with dict(zip(...)) rather than row-by-row iteration,
    applying the same performance lesson learned in the EDA phase:
    zip() reads the raw underlying columns directly instead of
    reconstructing a row object for every entry, which matters at
    multi-million-row scale.

  - Plain line-by-line streaming of training_pairs_sample.tsv rather
    than pandas' chunked reader, consistent with the approach used
    elsewhere in this project for files with long, variable-width rows.

  - Feature computation batched at FEATURE_BATCH_SIZE (from config.py):
    read a batch of lines, compute features for the whole batch, write
    the batch, discard it, move on. Bounds peak memory to one batch's
    intermediate data rather than the full labeled sample.
"""

import gc
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE3,
    RAW_TRAIN_SOURCE2, RAW_TRAIN_SOURCE3,
    TRAINING_SAMPLE_PATH, TRAINING_FEATURES_PATH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE,
    FEATURE_BATCH_SIZE, PROGRESS_EVERY,
)
from io_utils import load_tsv
from text_similarity_features import compute_pair_features, FEATURE_COLUMNS


def build_lookup(df, include_postal: bool) -> dict:
    """
    entity_id -> (name, address, postal_code) tuple, built via
    dict(zip(...)) over raw columns rather than row-by-row iteration.
    include_postal=False is used for the raw (uncleaned) lookups, which
    only ever need business_name, for the non-ASCII flag.
    """
    if include_postal:
        return dict(zip(
            df[COL_ENTITY_ID],
            zip(df[COL_NAME], df[COL_ADDRESS], df[COL_POSTAL_CODE]),
        ))
    return dict(zip(df[COL_ENTITY_ID], df[COL_NAME]))


def get_fields(entity_id: str, lookup: dict):
    """Safe 3-tuple lookup with empty-string defaults for a missing ID."""
    return lookup.get(entity_id, ("", "", ""))


def main():
    print("Loading cleaned source files and building lookups...")
    s1_df = load_tsv(CLEANED_TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s2_df = load_tsv(CLEANED_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])
    s3_df = load_tsv(CLEANED_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_POSTAL_CODE])

    s1_lookup = build_lookup(s1_df, include_postal=True)
    s2_lookup = build_lookup(s2_df, include_postal=True)
    s3_lookup = build_lookup(s3_df, include_postal=True)
    del s1_df, s2_df, s3_df
    gc.collect()

    print("Loading RAW (uncleaned) source2/3 names for the non-ASCII flag...")
    raw_s2_df = load_tsv(RAW_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_s3_df = load_tsv(RAW_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    raw_name_lookup = {}
    raw_name_lookup.update(build_lookup(raw_s2_df, include_postal=False))
    raw_name_lookup.update(build_lookup(raw_s3_df, include_postal=False))
    del raw_s2_df, raw_s3_df
    gc.collect()

    def get_other_fields(cand_id: str):
        return s2_lookup.get(cand_id) if cand_id.startswith("S2-") else s3_lookup.get(cand_id)

    def get_other_raw_name(cand_id: str) -> str:
        return raw_name_lookup.get(cand_id, "")

    print(f"\nStreaming {TRAINING_SAMPLE_PATH.name} and computing features "
          f"in batches of {FEATURE_BATCH_SIZE:,}...")

    n_processed = 0
    start_time = time.time()

    with open(TRAINING_SAMPLE_PATH, "r", encoding="utf-8") as in_f, \
         open(TRAINING_FEATURES_PATH, "w", encoding="utf-8", newline="") as out_f:

        header_line = "source1_entity_id\tcandidate_entity_id\tlabel\tsplit\t" + "\t".join(FEATURE_COLUMNS) + "\n"
        out_f.write(header_line)

        next(in_f)  # skip input header

        batch_lines = []
        for line in in_f:
            batch_lines.append(line.rstrip("\n"))
            if len(batch_lines) >= FEATURE_BATCH_SIZE:
                _process_batch(
                    batch_lines, out_f, s1_lookup, get_other_fields, get_other_raw_name
                )
                n_processed += len(batch_lines)
                batch_lines = []

                if n_processed % PROGRESS_EVERY == 0 or n_processed % FEATURE_BATCH_SIZE == 0:
                    elapsed = time.time() - start_time
                    rate = n_processed / elapsed if elapsed > 0 else 0
                    print(f"    {n_processed:,} pairs processed "
                          f"({elapsed:.1f}s elapsed, ~{rate:.0f} pairs/sec)")
                gc.collect()

        # final partial batch
        if batch_lines:
            _process_batch(batch_lines, out_f, s1_lookup, get_other_fields, get_other_raw_name)
            n_processed += len(batch_lines)

    elapsed = time.time() - start_time
    print(f"\nDone. {n_processed:,} pairs processed in {elapsed:.1f}s.")
    print(f"Feature table written to: {TRAINING_FEATURES_PATH}")


def _process_batch(lines, out_f, s1_lookup, get_other_fields, get_other_raw_name):
    """
    Computes features for one batch and writes it immediately -- keeps
    only one batch's worth of parsed lines and feature rows in memory
    at any time, discarded after each write.
    """
    out_lines = []
    for line in lines:
        if not line:
            continue
        s1_id, cand_id, label, split = line.split("\t")

        s1_name, s1_addr, s1_postal = s1_lookup.get(s1_id, ("", "", ""))
        other_fields = get_other_fields(cand_id)
        if other_fields is None:
            other_name, other_addr, other_postal = "", "", ""
        else:
            other_name, other_addr, other_postal = other_fields
        other_name_original = get_other_raw_name(cand_id)

        features = compute_pair_features(
            s1_name, s1_addr, s1_postal,
            other_name, other_addr, other_postal,
            other_name_original,
        )

        row = f"{s1_id}\t{cand_id}\t{label}\t{split}\t" + "\t".join(map(str, features))
        out_lines.append(row)

    out_f.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()