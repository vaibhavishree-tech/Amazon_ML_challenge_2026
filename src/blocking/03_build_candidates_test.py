"""
03_build_candidates_test.py

Runs the exact same blocking pipeline as 01, but against the cleaned
TEST files, producing the real candidate_pairs.tsv the matching model
will run inference over -- and the file the problem statement's
validator checks.

Only run this once 02's recall ceiling numbers look acceptable --
re-running the full test build after every blocking tweak wastes time
better spent iterating on the smaller, ground-truth-backed train run.

Per the problem statement: every Source 1 test entity must have exactly
one row, with candidate_entity_ids empty (not missing) when blocking
found nothing. process_country's per-entity loop already guarantees one
line per entity unconditionally, satisfying this requirement without
any extra logic in this script.
"""

import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TEST_SOURCE1, CLEANED_TEST_SOURCE2, CLEANED_TEST_SOURCE3,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY, COL_POSTAL_CODE,
    OUTPUT_DIR, CANDIDATES_TEST_PATH, COUNTRIES_OVERRIDE,
)
from io_utils import load_tsv
from blocking_pipeline import process_country


def main():
    print("Loading cleaned test files...")
    cols = [COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY, COL_POSTAL_CODE]
    s1 = load_tsv(CLEANED_TEST_SOURCE1, usecols=cols)
    s2 = load_tsv(CLEANED_TEST_SOURCE2, usecols=cols)
    s3 = load_tsv(CLEANED_TEST_SOURCE3, usecols=cols)

    # Discovers countries from the data rather than hardcoding
    # {US, India} -- the test set includes France, which has no
    # equivalent rows in training, so this must generalize to whatever
    # country values are actually present.
    countries = sorted(s1[COL_COUNTRY].unique().tolist())
    if COUNTRIES_OVERRIDE is not None:
        countries = [c for c in countries if c in COUNTRIES_OVERRIDE]
    print(f"Countries to process (test set -- includes France): {countries}")

    country_files = []
    for country in countries:
        s1_country = s1[s1[COL_COUNTRY] == country]
        s2_country = s2[s2[COL_COUNTRY] == country]
        s3_country = s3[s3[COL_COUNTRY] == country]
        s2s3_country = pd.concat([s2_country, s3_country], ignore_index=True)

        result_path = process_country(
            country_label=f"test_{country}",
            s1_country_df=s1_country,
            s2s3_country_df=s2s3_country,
            output_dir=OUTPUT_DIR,
        )
        country_files.append(result_path)

        del s1_country, s2_country, s3_country, s2s3_country
        gc.collect()

    print("\nConcatenating per-country files into final test candidate_pairs.tsv...")
    with open(CANDIDATES_TEST_PATH, "w", encoding="utf-8", newline="") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for path in country_files:
            with open(path, "r", encoding="utf-8") as in_f:
                next(in_f)
                for line in in_f:
                    out_f.write(line)

    print(f"Final candidate_pairs.tsv written to: {CANDIDATES_TEST_PATH}")
    print("Copy/rename this into your submission package's output/ folder when ready.")


if __name__ == "__main__":
    main()