"""
01_clean_source_files.py

Applies the full cleaning pipeline to all six source files (train and
test, sources 1/2/3) and writes cleaned versions to cleaning/cleaned_outputs/.

"""

import gc
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    CLEANED_TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE3,
    CLEANED_TEST_SOURCE1, CLEANED_TEST_SOURCE2, CLEANED_TEST_SOURCE3,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY, COL_POSTAL_CODE,
    UNIDECODE_BATCH_SIZE,
)
from io_utils import load_tsv, save_tsv
from clean_pipeline import clean_business_name, clean_business_address

# (input_path, output_path) pairs, processed one at a time in this order.
# Train files first, then test -- order doesn't affect correctness, but
# keeps console output grouped by split while the pipeline runs.
FILE_PAIRS = [
    (TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE1),
    (TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE2),
    (TRAIN_SOURCE3, CLEANED_TRAIN_SOURCE3),
    (TEST_SOURCE1, CLEANED_TEST_SOURCE1),
    (TEST_SOURCE2, CLEANED_TEST_SOURCE2),
    (TEST_SOURCE3, CLEANED_TEST_SOURCE3),
]


def clean_one_file(input_path, output_path):
    """
    Loads one source file, runs the full name/address cleaning pipeline
    on it, writes the cleaned result, then explicitly frees memory
    before returning -- called once per file in FILE_PAIRS by main(),
    never with more than one file's data alive at a time.
    """
    print(f"\n--- Cleaning {input_path.name} ---")

    df = load_tsv(input_path, usecols=[COL_ENTITY_ID, COL_NAME, COL_ADDRESS, COL_COUNTRY])

    print("  cleaning business_name...")
    df[COL_NAME] = clean_business_name(df[COL_NAME], batch_size=UNIDECODE_BATCH_SIZE)

    print("  cleaning business_address...")
    cleaned_address, postal_codes = clean_business_address(
        df[COL_ADDRESS], batch_size=UNIDECODE_BATCH_SIZE
    )
    df[COL_ADDRESS] = cleaned_address
    df[COL_POSTAL_CODE] = postal_codes

    # country is already confirmed clean by the EDA (exact-match strings
    # only, 0% inconsistency in true-match pairs) so it passes through
    # untouched -- no cleaning step needed or applied.

    save_tsv(df, output_path)

    del df, cleaned_address, postal_codes
    gc.collect()
    print(f"--- Done: {output_path.name} ---")


def main():
    # Runs clean_one_file() for all six (input, output) pairs in
    # sequence -- one file fully processed and freed before the next
    # begins, per the module-level memory strategy described above.
    for input_path, output_path in FILE_PAIRS:
        clean_one_file(input_path, output_path)
    print("\nAll six files cleaned.")


if __name__ == "__main__":
    main()