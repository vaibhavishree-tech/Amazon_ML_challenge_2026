"""
03_validate_transliteration_sample.py

Directly checks whether unidecode's transliteration of non-ASCII
business names actually converges toward the Source 1 spelling of the
same true-match pair, per the locked plan's instruction not to trust
unidecode blindly. Reuses the ground truth to find real true-match
pairs where the Source 2/3 side is written in a non-ASCII script, then
shows the ORIGINAL Source 1 name next to the CLEANED (post-pipeline)
Source 2/3 name, side by side, for manual visual confirmation.

This is read-only against a small sample -- it does not modify any
files. Output is a CSV meant to be read by a person, the same way
05_true_match_sample_for_manual_review.csv was in the EDA phase.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    COL_ENTITY_ID, COL_NAME,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    REPORT_DIR, UNIDECODE_BATCH_SIZE,
)
from io_utils import load_tsv
from clean_pipeline import clean_business_name

SAMPLE_SIZE = 50
RANDOM_SEED = 42
OUTPUT_CSV = REPORT_DIR / "03_transliteration_validation_sample.csv"

NON_ASCII_PATTERN = r"[^\x20-\x7E]"


def explode_matched_ids(gt: pd.DataFrame) -> pd.DataFrame:
    """Same explode pattern used throughout the EDA phase."""
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"}).reset_index(drop=True)


def main():
    print("Loading files...")
    s1 = load_tsv(TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME])
    s2 = load_tsv(TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    s3 = load_tsv(TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])
    gt = load_tsv(TRAIN_GROUND_TRUTH)

    exploded = explode_matched_ids(gt)

    s2_lookup = dict(zip(s2[COL_ENTITY_ID], s2[COL_NAME]))
    s3_lookup = dict(zip(s3[COL_ENTITY_ID], s3[COL_NAME]))
    s1_lookup = dict(zip(s1[COL_ENTITY_ID], s1[COL_NAME]))

    def get_other_name(match_id):
        return s2_lookup.get(match_id) if match_id.startswith("S2-") else s3_lookup.get(match_id)

    exploded["other_name"] = exploded["matched_id"].map(get_other_name)
    exploded["s1_name"] = exploded[COL_GT_SOURCE1_ID].map(s1_lookup.get)

    # Only keep pairs where the non-S1 side actually contains non-ASCII
    # text -- these are exactly the cases unidecode needs to handle
    # correctly, so the validation sample should be drawn from them
    # specifically rather than a plain random sample.
    is_non_ascii = exploded["other_name"].str.contains(NON_ASCII_PATTERN, regex=True, na=False)
    non_ascii_pairs = exploded[is_non_ascii].dropna(subset=["s1_name", "other_name"])

    if len(non_ascii_pairs) == 0:
        print("No non-ASCII true-match pairs found -- nothing to validate.")
        return

    sample_size = min(SAMPLE_SIZE, len(non_ascii_pairs))
    sample = non_ascii_pairs.sample(n=sample_size, random_state=RANDOM_SEED).copy()

    print(f"Running cleaning pipeline on {sample_size} sampled non-ASCII names...")
    # Runs the SAME clean_business_name() pipeline used in
    # 01_clean_source_files.py, on just this small sample -- so the
    # "cleaned" column here reflects exactly what the real cleaned
    # output files would contain for these specific names.
    sample["other_name_cleaned"] = clean_business_name(
        sample["other_name"], batch_size=UNIDECODE_BATCH_SIZE
    )
    sample["s1_name_cleaned"] = clean_business_name(
        sample["s1_name"], batch_size=UNIDECODE_BATCH_SIZE
    )

    output = sample[[
        COL_GT_SOURCE1_ID, "s1_name", "s1_name_cleaned",
        "matched_id", "other_name", "other_name_cleaned",
    ]].rename(columns={
        "s1_name": "source1_name_original",
        "s1_name_cleaned": "source1_name_cleaned",
        "other_name": "matched_name_original_non_ascii",
        "other_name_cleaned": "matched_name_after_unidecode",
    })

    output.to_csv(OUTPUT_CSV, index=False)
    print(f"\nValidation sample written to: {OUTPUT_CSV}")
    print("Open this file and visually compare 'source1_name_cleaned' against "
          "'matched_name_after_unidecode' for each row. If unidecode is working well, "
          "the two should look close to each other despite starting from different scripts. "
          "If they look unrelated for a given script, that script's transliteration is "
          "unreliable and worth flagging in your methodology document.")


if __name__ == "__main__":
    main()