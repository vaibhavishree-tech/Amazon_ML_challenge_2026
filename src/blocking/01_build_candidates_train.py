"""
01_build_candidates_train.py

Runs the blocking pipeline over the TRAINING data (cleaned source1/2/3),
country by country. Output is INTERNAL -- used only by
02_measure_recall_ceiling.py to validate the blocking design against
known ground truth before spending time on the real test-set run.

Countries are discovered from the data itself (not hardcoded), so this
script makes no assumption about which countries exist -- consistent
with the problem statement's requirement to treat country as an open
set, which matters since the test set introduces a country (France)
absent from training.
"""

import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE3,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY, COL_POSTAL_CODE,
    OUTPUT_DIR, CANDIDATES_TRAIN_PATH, COUNTRIES_OVERRIDE,
)
from io_utils import load_tsv
from blocking_pipeline import process_country


def main():
    # Loads all three cleaned train source files, restricted to only
    # the columns blocking actually needs (entity_id, name, address,
    # country, postal code) to keep memory proportional to what's used.
    print("Loading cleaned training files...")
    cols = [COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY, COL_POSTAL_CODE]
    s1 = load_tsv(CLEANED_TRAIN_SOURCE1, usecols=cols)
    s2 = load_tsv(CLEANED_TRAIN_SOURCE2, usecols=cols)
    s3 = load_tsv(CLEANED_TRAIN_SOURCE3, usecols=cols)

    countries = sorted(s1[COL_COUNTRY].unique().tolist())
    if COUNTRIES_OVERRIDE is not None:
        countries = [c for c in countries if c in COUNTRIES_OVERRIDE]
    print(f"Countries to process: {countries}")

    country_files = []
    for country in countries:
        # Partitions each source file by country before blocking --
        # Source 2 and Source 3 are combined into one pool per country,
        # since Signal A-D all search across both sources together,
        # with the source distinguished later only by entity_id prefix.
        s1_country = s1[s1[COL_COUNTRY] == country]
        s2_country = s2[s2[COL_COUNTRY] == country]
        s3_country = s3[s3[COL_COUNTRY] == country]
        s2s3_country = pd.concat([s2_country, s3_country], ignore_index=True)

        result_path = process_country(
            country_label=country,
            s1_country_df=s1_country,
            s2s3_country_df=s2s3_country,
            output_dir=OUTPUT_DIR,
        )
        country_files.append(result_path)

        del s1_country, s2_country, s3_country, s2s3_country
        gc.collect()

    # Concatenates each country's per-country output file into one
    # final train candidate file -- done as a plain text line copy
    # (not re-loaded into pandas) since the per-country files are
    # already in the correct final format and this step only needs to
    # merge them, not transform them.
    print("\nConcatenating per-country files into final train candidate file...")
    with open(CANDIDATES_TRAIN_PATH, "w", encoding="utf-8", newline="") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for path in country_files:
            with open(path, "r", encoding="utf-8") as in_f:
                next(in_f)  # skip that file's own header
                for line in in_f:
                    out_f.write(line)

    print(f"Final train candidate file written to: {CANDIDATES_TRAIN_PATH}")


if __name__ == "__main__":
    main()