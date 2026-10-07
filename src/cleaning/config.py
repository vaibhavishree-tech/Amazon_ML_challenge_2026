"""
config.py

Central configuration for Phase 1, Step 2: Preprocessing and Cleaning.
Mirrors the structure of eda/config.py so paths stay consistent across
the whole project. All other cleaning scripts import from here rather
than hardcoding paths, so the folder layout only needs to change in one
place if it ever moves.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Ground truth is not cleaned (it only contains IDs, no free text), but
# is referenced by the transliteration validation script to pull real
# true-match pairs to check unidecode quality against.
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# Where cleaned files are written. Kept fully separate from dataset/ so
# the raw input data is never at risk of being overwritten -- every
# cleaning script only ever reads from DATASET_DIR and writes here.
CLEANED_DIR = PROJECT_ROOT / "cleaning" / "cleaned_outputs"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_TRAIN_SOURCE1 = CLEANED_DIR / "train_source1_cleaned.tsv"
CLEANED_TRAIN_SOURCE2 = CLEANED_DIR / "train_source2_cleaned.tsv"
CLEANED_TRAIN_SOURCE3 = CLEANED_DIR / "train_source3_cleaned.tsv"
CLEANED_TEST_SOURCE1 = CLEANED_DIR / "test_source1_cleaned.tsv"
CLEANED_TEST_SOURCE2 = CLEANED_DIR / "test_source2_cleaned.tsv"
CLEANED_TEST_SOURCE3 = CLEANED_DIR / "test_source3_cleaned.tsv"

# Where validation/report scripts (02, 03, 04) write their output --
# kept as a subfolder of CLEANED_DIR so all cleaning-phase evidence
# (cleaned data + the reports proving it worked) lives in one place.
REPORT_DIR = CLEANED_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical column names, identical to eda/config.py.
COL_ENTITY_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDRESS = "business_address"
COL_COUNTRY = "country"
COL_POSTAL_CODE = "postal_code_extracted"  # new column added by cleaning

COL_GT_SOURCE1_ID = "source1_entity_id"
COL_GT_MATCHED_IDS = "matched_entity_ids"

# How many rows to process per batch during unidecode transliteration --
# see clean_pipeline.py for why this step needs batching specifically.
UNIDECODE_BATCH_SIZE = 500_000