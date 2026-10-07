"""
config.py

Central configuration for the EDA phase of the Business Entity Resolution
Challenge. Every other EDA script imports paths and constants from here so
that if the folder structure changes, only this file needs to be edited.
"""

from pathlib import Path

# Root of the student_resource/ directory. Using Path(__file__).parent
# keeps the reference relative to this config file instead of the current
# working directory, so scripts work regardless of where you launch them.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Input files
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# Where every EDA script writes its results (CSV summaries, reports).
# Kept separate from dataset/ so raw data is never at risk of being
# overwritten.
OUTPUT_DIR = PROJECT_ROOT / "eda" / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Canonical column names, taken directly from the problem statement.
# Centralising these avoids typos like "buisness_name" scattered across
# five different scripts.
COL_ENTITY_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDRESS = "business_address"
COL_COUNTRY = "country"

COL_GT_SOURCE1_ID = "source1_entity_id"
COL_GT_MATCHED_IDS = "matched_entity_ids"

# Source ID prefixes, used to identify which file an ID came from once
# IDs are merged/joined across sources. Defined once here rather than
# as string literals scattered through the scripts, so a prefix change
# (unlikely, but possible if the dataset format ever changes) only
# needs to be made in one place.
PREFIX_S1 = "S1-"
PREFIX_S2 = "S2-"
PREFIX_S3 = "S3-"