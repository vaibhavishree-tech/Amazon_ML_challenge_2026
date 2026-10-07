"""
02_measure_recall_ceiling.py

Compares the training candidate set (01's output) against the training
ground truth to compute:

  1. Overall recall ceiling: what fraction of true matches survive
     blocking. This is the hard upper bound on final leaderboard
     recall -- nothing downstream can recover a match blocking drops,
     since the matching model only ever scores candidates blocking
     actually produced.
  2. Recall broken down by match-cardinality bucket (0/1/2/3+ matches
     per Source 1 entity), reusing the same buckets established during
     the EDA phase's ground-truth topology analysis.
  3. Recall broken down by whether the matched record's ORIGINAL name
     was ASCII or non-ASCII -- directly testing whether the
     transliteration weakness identified during the cleaning phase
     (token-based name similarity collapsing for non-ASCII-origin
     pairs) also shows up as a blocking recall gap, since Signals A/B
     rely on name tokens that are known to be less reliable for this
     subset.
  4. Candidate volume stats: average/max candidates per entity, percent
     of entities with zero candidates (each one is a guaranteed miss
     regardless of matching-model quality).

Processes the candidate file in CHUNKS rather than loading it fully,
since total candidate rows can be very large (up to millions of Source
1 entities times up to MAX_CANDIDATES_PER_ENTITY each) -- the same
memory-bounding discipline used throughout this project.
"""

import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CANDIDATES_TRAIN_PATH, TRAIN_GROUND_TRUTH,
    TRAIN_SOURCE2_RAW, TRAIN_SOURCE3_RAW,
    COL_ENTITY_ID, COL_NAME,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    REPORT_DIR,
)
from io_utils import load_tsv

REPORT_PATH = REPORT_DIR / "02_recall_ceiling_report.txt"
CANDIDATE_READ_CHUNKSIZE = 50_000
NON_ASCII_PATTERN = r"[^\x20-\x7E]"


def explode_matched_ids(gt: pd.DataFrame) -> pd.DataFrame:
    """Same explode pattern used throughout the EDA and cleaning phases."""
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"}).reset_index(drop=True)


def cardinality_bucket(count: int) -> str:
    """Buckets a raw match count into the same 0/1/2/3+ categories used in the EDA phase's topology report."""
    if count == 0:
        return "0 (singleton)"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def main():
    print("Loading ground truth...")
    gt = load_tsv(TRAIN_GROUND_TRUTH)
    gt_exploded = explode_matched_ids(gt)

    # Builds a source1_entity_id -> cardinality_bucket lookup from the
    # raw (non-exploded) ground truth, needed later to break recall
    # down by how many true matches each entity actually has.
    match_count_per_s1 = gt.copy()
    is_empty = match_count_per_s1[COL_GT_MATCHED_IDS].str.strip() == ""
    match_count_per_s1["match_count"] = (
        match_count_per_s1[COL_GT_MATCHED_IDS].str.count(",").fillna(-1) + 1
    )
    match_count_per_s1.loc[is_empty, "match_count"] = 0
    bucket_lookup = dict(zip(
        match_count_per_s1[COL_GT_SOURCE1_ID],
        match_count_per_s1["match_count"].apply(cardinality_bucket),
    ))
    del match_count_per_s1
    gc.collect()

    print("Building true-pair lookup set...")
    true_pairs_set = set(zip(gt_exploded[COL_GT_SOURCE1_ID], gt_exploded["matched_id"]))
    total_true_pairs = len(true_pairs_set)
    print(f"  {total_true_pairs:,} true match pairs")

    # Builds the ascii/non-ascii label from the RAW (uncleaned) source
    # files, since this label must reflect what the text looked like
    # before cleaning's unidecode step, not after.
    print("Building non-ASCII origin flag from original (uncleaned) names...")
    orig_s2 = load_tsv(TRAIN_SOURCE2_RAW, usecols=[COL_ENTITY_ID, COL_NAME])
    orig_s3 = load_tsv(TRAIN_SOURCE3_RAW, usecols=[COL_ENTITY_ID, COL_NAME])
    non_ascii_lookup = {}
    for df in (orig_s2, orig_s3):
        flags = df[COL_NAME].str.contains(NON_ASCII_PATTERN, regex=True)
        non_ascii_lookup.update(dict(zip(df[COL_ENTITY_ID], flags)))
    del orig_s2, orig_s3
    gc.collect()

    found_pairs = set()
    total_candidate_rows = 0
    candidate_counts = []  # per-entity candidate count, for volume stats
    zero_candidate_entities = 0
    total_entities = 0

    # Plain file iteration instead of pandas' chunked reader. Individual
    # candidate rows can be very long (hundreds of comma-separated IDs),
    # and a chunked pandas reader must build one large contiguous buffer
    # per chunk before processing can begin -- at this row length, that
    # buffer allocation becomes the actual bottleneck. Since this script
    # only ever performs a tab split and a comma split, there is no real
    # need for pandas here: plain Python line reading processes one row
    # at a time, so peak memory depends only on a single line's length,
    # never on a whole chunk's worth of rows at once.
    print("Scanning candidate file (plain text line reading, no pandas)...")
    PROGRESS_EVERY = 20_000

    with open(CANDIDATES_TRAIN_PATH, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header row

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                s1_id, cand_str = parts[0], ""
            else:
                s1_id, cand_str = parts[0], parts[1]

            total_entities += 1

            if not cand_str:
                candidate_counts.append(0)
                zero_candidate_entities += 1
                continue

            cand_ids = cand_str.split(",")
            n_cands = len(cand_ids)
            candidate_counts.append(n_cands)

            for cand_id in cand_ids:
                total_candidate_rows += 1
                pair = (s1_id, cand_id)
                if pair in true_pairs_set:
                    found_pairs.add(pair)

            if total_entities % PROGRESS_EVERY == 0:
                print(f"    {total_entities:,} entities scanned so far")

    overall_recall = len(found_pairs) / total_true_pairs if total_true_pairs else 0

    # Per-bucket and per-ascii-origin recall: iterates the true-pairs
    # set (denominator) and the found-pairs set (numerator) separately,
    # tagging each pair by its cardinality bucket and ASCII status via
    # the lookups built above.
    bucket_totals, bucket_found = {}, {}
    ascii_totals, ascii_found = {"ascii": 0, "non_ascii": 0}, {"ascii": 0, "non_ascii": 0}

    for s1_id, matched_id in true_pairs_set:
        bucket = bucket_lookup.get(s1_id, "unknown")
        bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1

        is_non_ascii = non_ascii_lookup.get(matched_id, False)
        ascii_key = "non_ascii" if is_non_ascii else "ascii"
        ascii_totals[ascii_key] += 1

    for s1_id, matched_id in found_pairs:
        bucket = bucket_lookup.get(s1_id, "unknown")
        bucket_found[bucket] = bucket_found.get(bucket, 0) + 1

        is_non_ascii = non_ascii_lookup.get(matched_id, False)
        ascii_key = "non_ascii" if is_non_ascii else "ascii"
        ascii_found[ascii_key] += 1

    lines = ["=== Blocking Recall Ceiling Report ===\n"]
    lines.append(f"Total true match pairs (training ground truth): {total_true_pairs:,}")
    lines.append(f"True match pairs found in candidate set:        {len(found_pairs):,}")
    lines.append(f"OVERALL RECALL CEILING: {overall_recall:.4f} ({overall_recall*100:.2f}%)\n")

    lines.append("=== Recall by Match Cardinality Bucket ===\n")
    for bucket in sorted(bucket_totals.keys()):
        total = bucket_totals[bucket]
        found = bucket_found.get(bucket, 0)
        recall = found / total if total else 0
        lines.append(f"  {bucket:15s}: {found:,} / {total:,} found ({recall*100:.2f}%)")
    lines.append("")

    lines.append("=== Recall by Matched-Side Original ASCII Status ===\n")
    for key in ("ascii", "non_ascii"):
        total = ascii_totals[key]
        found = ascii_found[key]
        recall = found / total if total else 0
        lines.append(f"  {key:10s}: {found:,} / {total:,} found ({recall*100:.2f}%)")
    lines.append("")
    lines.append(
        "If non_ascii recall is meaningfully lower than ascii recall, blocking is "
        "inheriting the same transliteration weakness found during cleaning "
        "(name-token similarity collapsing for this subset) -- the address-token "
        "and postal-code signals should be carrying more of the load for these "
        "entities, and the document-frequency / rare-K settings may be worth "
        "revisiting specifically to compensate.\n"
    )

    lines.append("=== Candidate Volume Stats ===\n")
    avg_candidates = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0
    max_candidates = max(candidate_counts) if candidate_counts else 0
    pct_zero = 100 * zero_candidate_entities / total_entities if total_entities else 0
    lines.append(f"  Total Source 1 entities processed: {total_entities:,}")
    lines.append(f"  Total candidate pairs generated:   {total_candidate_rows:,}")
    lines.append(f"  Average candidates per entity:     {avg_candidates:.1f}")
    lines.append(f"  Max candidates for one entity:     {max_candidates:,}")
    lines.append(f"  Entities with ZERO candidates:     {zero_candidate_entities:,} ({pct_zero:.2f}%)")
    lines.append(
        "  Every zero-candidate entity is a guaranteed miss regardless of what the "
        "matching model does -- if this percentage is high, it's worth checking whether "
        "those specific entities have unusually short/sparse names or addresses.\n"
    )

    full_report = "\n".join(lines)
    print(full_report)
    REPORT_PATH.write_text(full_report, encoding="utf-8")
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()