"""
01_volume_integrity.py

Phase 1 EDA - Step 1: Dataset Volume and Completeness / Integrity Audit.

Performance rewrite notes:

- ground_truth_referential_integrity and multi_mapping_check originally
  used gt.iterrows() with a manual string .split(",") per row. At scale
  this is the single slowest pattern in pandas: iterrows() reconstructs
  a full pandas Series object for every row, which is very expensive
  when repeated millions of times.

- Both functions are rewritten to use str.split(",").explode(), which
  turns the comma-separated matched_entity_ids column into one row per
  individual matched ID using pandas' internal vectorized C code, not a
  Python-level loop. The set-based existence checks and groupby-based
  multi-mapping check that follow are then vectorized as well.

Functional behaviour (what is checked and why) is unchanged from the
original version -- only how it is computed has changed.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    OUTPUT_DIR,
)
from io_utils import load_tsv

REPORT_PATH = OUTPUT_DIR / "01_volume_integrity_report.txt"


def row_count_summary(datasets: dict) -> str:
    """
    Reports the raw row count for every provided dataset (all six
    source files plus ground truth, passed in by main()). This is the
    simplest possible check, but it's the first thing worth confirming
    before anything else -- if a file loaded with an unexpectedly low
    or high row count, every downstream check in this script would be
    working from the wrong data.
    """
    lines = ["=== Row Counts ===\n"]
    for label, df in datasets.items():
        lines.append(f"{label:20s}: {len(df):>10,} rows")
    return "\n".join(lines) + "\n"


def missing_value_profile(datasets: dict) -> str:
    """
    Unchanged logic, already vectorized in the original version:
    str.strip() == "" on a full column is a single vectorized pandas
    operation, not a per-row loop, so this was never the slow part.
    """
    lines = ["=== Missing / Empty Value Profile ===\n"]
    for label, df in datasets.items():
        lines.append(f"-- {label} --")
        n = len(df)
        for col in df.columns:
            empty_mask = df[col].str.strip() == ""
            count = int(empty_mask.sum())
            pct = 100 * count / n if n else 0
            lines.append(f"  {col:20s}: {count:>9,} / {n:,} empty ({pct:.2f}%)")
        lines.append("")
    return "\n".join(lines) + "\n"


def whitespace_and_special_char_audit(datasets: dict) -> str:
    """
    Rewritten to use vectorized str.contains() calls instead of the
    original's per-row lambda via .apply() for the curly-quote check.
    str.contains() with a regex character class runs as a single
    vectorized pass; .apply() with a Python lambda runs the lambda once
    per row in the interpreter, which is far slower at 5 million rows.
    """
    lines = ["=== Whitespace / Special Character Audit ===\n"]
    text_cols = [COL_NAME, COL_ADDRESS]
    curly_quote_pattern = r"[\u201c\u201d\u2018\u2019]"

    for label, df in datasets.items():
        lines.append(f"-- {label} --")
        for col in text_cols:
            if col not in df.columns:
                continue
            series = df[col]
            n_tab = int(series.str.contains("\t", regex=False).sum())
            n_newline = int(series.str.contains(r"[\n\r]", regex=True).sum())
            n_leading_trailing_ws = int((series != series.str.strip()).sum())
            n_curly_quotes = int(series.str.contains(curly_quote_pattern, regex=True).sum())
            lines.append(
                f"  {col:20s}: tabs={n_tab}, newlines={n_newline}, "
                f"leading/trailing whitespace={n_leading_trailing_ws}, "
                f"curly quotes={n_curly_quotes}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def entity_id_uniqueness(datasets: dict) -> str:
    """
    Confirms entity_id is unique within each individual source file.
    This is a prerequisite for every later phase of the project: if IDs
    were not unique, dictionary-based lookups (used throughout cleaning,
    blocking, and matching) would silently collide and drop data, with
    no error raised -- so this check needs to pass cleanly before any
    of that downstream work can be trusted.
    """
    lines = ["=== Entity ID Uniqueness ===\n"]
    for label, df in datasets.items():
        if COL_ENTITY_ID not in df.columns:
            continue
        n_total = len(df)
        n_unique = df[COL_ENTITY_ID].nunique()
        n_dupes = n_total - n_unique
        status = "OK" if n_dupes == 0 else "PROBLEM"
        lines.append(f"{label:20s}: {n_total:,} rows, {n_unique:,} unique IDs, "
                      f"{n_dupes:,} duplicates [{status}]")
    return "\n".join(lines) + "\n"


def explode_matched_ids(gt: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the ground truth's one-row-per-source1-entity,
    comma-separated matched_entity_ids format into a long-format
    dataframe with one row per (source1_entity_id, matched_entity_id)
    pair.

    Why this replaces the loop: str.split(",") on the full column
    produces a column of Python lists in one vectorized pass; .explode()
    then turns each list into its own row, again as a single vectorized
    pandas operation. This is the standard pandas pattern for turning
    delimited-list columns into a proper relational shape, and it is
    dramatically faster than iterating rows and splitting strings
    one at a time.
    """
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    # Rows where the original cell was empty produce either NaN or an
    # empty string after explode/strip -- drop those, they represent
    # "no match", not an actual matched ID.
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"})


def ground_truth_referential_integrity(
    gt: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame
) -> str:
    """
    Checks that the ground truth file and the three source files agree
    with each other, in both directions: every ID the ground truth
    references must actually exist in the corresponding source file,
    and every Source 1 entity must have a ground truth row at all.
    Also checks that every matched ID uses a recognised S2-/S3- prefix.
    Any failure here would mean downstream training/evaluation code is
    silently working with IDs that don't resolve to real records.
    """
    lines = ["=== Ground Truth Referential Integrity ===\n"]

    s1_ids = set(s1[COL_ENTITY_ID])
    s2_ids = set(s2[COL_ENTITY_ID])
    s3_ids = set(s3[COL_ENTITY_ID])

    gt_s1_ids = set(gt[COL_GT_SOURCE1_ID])
    orphan_s1 = gt_s1_ids - s1_ids
    lines.append(f"Ground truth S1 IDs not found in train_source1.tsv: {len(orphan_s1)}")
    if orphan_s1:
        lines.append(f"  examples: {list(orphan_s1)[:5]}")

    missing_from_gt = s1_ids - gt_s1_ids
    lines.append(f"Source1 IDs with no row at all in ground truth: {len(missing_from_gt)}")
    if missing_from_gt:
        lines.append(f"  examples: {list(missing_from_gt)[:5]}")

    exploded = explode_matched_ids(gt)

    is_s2 = exploded["matched_id"].str.startswith("S2-")
    is_s3 = exploded["matched_id"].str.startswith("S3-")
    is_other_prefix = ~is_s2 & ~is_s3

    # Split the exploded matches by prefix so each half can be checked
    # against the correct source file's ID set -- an S2- ID has no
    # business being validated against s3_ids and vice versa.
    s2_rows = exploded[is_s2]
    orphan_s2 = s2_rows[~s2_rows["matched_id"].isin(s2_ids)]["matched_id"]

    s3_rows = exploded[is_s3]
    orphan_s3 = s3_rows[~s3_rows["matched_id"].isin(s3_ids)]["matched_id"]

    wrong_prefix = exploded[is_other_prefix]["matched_id"]

    n_orphan = len(orphan_s2) + len(orphan_s3)
    lines.append(f"Matched IDs referenced in ground truth but missing from "
                 f"source2/source3 files: {n_orphan:,}")
    if n_orphan:
        examples = pd.concat([orphan_s2, orphan_s3]).head(5).tolist()
        lines.append(f"  examples: {examples}")

    lines.append(f"Matched IDs with an unrecognised prefix (not S2-/S3-): "
                 f"{len(wrong_prefix):,}")
    if len(wrong_prefix):
        lines.append(f"  examples: {wrong_prefix.head(5).tolist()}")

    return "\n".join(lines) + "\n"


def multi_mapping_check(gt: pd.DataFrame) -> str:
    """
    Rewritten using groupby instead of a manual dict built inside a
    Python loop. groupby("matched_id")[COL_GT_SOURCE1_ID].nunique() does
    the same "how many distinct S1 entities reference this matched ID"
    computation as the original loop, but as a single vectorized
    aggregation.
    """
    lines = ["=== Multi-Mapping Check (does an S2/S3 ID match multiple S1 entities) ===\n"]

    exploded = explode_matched_ids(gt)
    mapping_counts = exploded.groupby("matched_id")[COL_GT_SOURCE1_ID].nunique()

    multi_mapped = mapping_counts[mapping_counts > 1]
    lines.append(f"Total distinct matched IDs referenced: {len(mapping_counts):,}")
    lines.append(f"Matched IDs that map to more than one Source 1 entity: "
                 f"{len(multi_mapped):,}")
    if len(multi_mapped):
        example_id = multi_mapped.index[0]
        example_s1_ids = exploded.loc[
            exploded["matched_id"] == example_id, COL_GT_SOURCE1_ID
        ].unique()
        lines.append(f"  example: {example_id} -> matches S1 entities {list(example_s1_ids)}")

    return "\n".join(lines) + "\n"


def main():
    # Loads every train and test source file plus ground truth, then
    # runs each integrity check in turn, writing a single combined
    # report. Train and test files are checked together (via
    # all_sources) for row counts and the missing-value/whitespace
    # audits, since those checks don't depend on ground truth; the
    # referential-integrity and multi-mapping checks only apply to the
    # train split, since only train has a ground truth file to check
    # against.
    print("Loading files...")
    train = {
        "train_source1": load_tsv(TRAIN_SOURCE1),
        "train_source2": load_tsv(TRAIN_SOURCE2),
        "train_source3": load_tsv(TRAIN_SOURCE3),
    }
    test = {
        "test_source1": load_tsv(TEST_SOURCE1),
        "test_source2": load_tsv(TEST_SOURCE2),
        "test_source3": load_tsv(TEST_SOURCE3),
    }
    gt = load_tsv(TRAIN_GROUND_TRUTH)
    all_sources = {**train, **test}

    report_parts = [
        row_count_summary({**all_sources, "train_ground_truth": gt}),
        missing_value_profile(all_sources),
        whitespace_and_special_char_audit(all_sources),
        entity_id_uniqueness(all_sources),
        ground_truth_referential_integrity(
            gt, train["train_source1"], train["train_source2"], train["train_source3"]
        ),
        multi_mapping_check(gt),
    ]

    full_report = "\n".join(report_parts)
    print(full_report)

    REPORT_PATH.write_text(full_report, encoding="utf-8")
    print(f"\nFull report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()