"""
config.py

Central configuration for the matching phase: feature engineering,
baseline model training, threshold selection, and final inference.

Note: this file also retains configuration constants for an
experimental embedding-augmented matching variant (model v2, country-
specific thresholds) explored but not used in the final submitted
result -- see experimental_embeddings/README_EXPERIMENTAL.md for
details on that variant's status. The constants are kept here rather
than duplicated, since the experimental scripts still import from this
shared file.
"""

from pathlib import Path
from text_similarity_features import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Inputs from earlier phases ---
TRAIN_GROUND_TRUTH = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"
CANDIDATES_TRAIN_PATH = PROJECT_ROOT / "blocking" / "blocking_outputs" / "candidates_train.tsv"

CLEANED_DIR = PROJECT_ROOT / "cleaning" / "cleaned_outputs"
CLEANED_TRAIN_SOURCE1 = CLEANED_DIR / "train_source1_cleaned.tsv"
CLEANED_TRAIN_SOURCE2 = CLEANED_DIR / "train_source2_cleaned.tsv"
CLEANED_TRAIN_SOURCE3 = CLEANED_DIR / "train_source3_cleaned.tsv"

# RAW (uncleaned) source2/3 -- needed only to compute the
# "matched_side_was_non_ascii" feature, which must reflect what the
# text looked like BEFORE cleaning, not after (cleaning's unidecode
# step already strips non-ASCII characters from the cleaned files by
# design, so that signal would be lost if read from the cleaned files).
RAW_TRAIN_SOURCE2 = PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv"
RAW_TRAIN_SOURCE3 = PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv"

COL_ENTITY_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDRESS = "business_address"
COL_COUNTRY = "country"
COL_POSTAL_CODE = "postal_code_extracted"

COL_GT_SOURCE1_ID = "source1_entity_id"
COL_GT_MATCHED_IDS = "matched_entity_ids"
COL_CANDIDATE_IDS = "candidate_entity_ids"

OUTPUT_DIR = PROJECT_ROOT / "matching" / "matching_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAINING_SAMPLE_PATH = OUTPUT_DIR / "training_pairs_sample.tsv"
TRAINING_FEATURES_PATH = OUTPUT_DIR / "training_features.tsv"

# ---------------------------------------------------------------------
# Negative sampling parameters.
#
# The full candidate set is far too large to compute features for in
# its entirety, so a sampled subset is used for training instead: every
# true-match pair (positive) is kept, and non-matching candidate pairs
# (negatives) are kept at a target ratio of NEGATIVE_TO_POSITIVE_RATIO
# per positive. The approximate totals below (from blocking's full
# train-set run) are used to precompute a sampling PROBABILITY applied
# per-row during a single streaming pass, avoiding a separate counting
# pass over the full candidate file.
# ---------------------------------------------------------------------
NEGATIVE_TO_POSITIVE_RATIO = 4
APPROX_TOTAL_TRUE_PAIRS_FOUND = 7_085_224
APPROX_TOTAL_CANDIDATE_ROWS = 1_100_023_136

APPROX_TOTAL_NEGATIVE_ROWS = APPROX_TOTAL_CANDIDATE_ROWS - APPROX_TOTAL_TRUE_PAIRS_FOUND
TARGET_NEGATIVES_TO_KEEP = NEGATIVE_TO_POSITIVE_RATIO * APPROX_TOTAL_TRUE_PAIRS_FOUND
NEGATIVE_SAMPLE_PROBABILITY = min(1.0, TARGET_NEGATIVES_TO_KEEP / APPROX_TOTAL_NEGATIVE_ROWS)

VALIDATION_SPLIT_PERCENT = 20

# --- Feature computation batching ---
# Bounds peak memory during feature computation the same way batching
# was used throughout the earlier phases: process a fixed number of
# pairs at a time, write results incrementally, never hold the full
# labeled training set's features in memory at once.
FEATURE_BATCH_SIZE = 200_000

PROGRESS_EVERY = 100_000

# --- Model training ---
MODEL_DIR = OUTPUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_MODEL_PATH = MODEL_DIR / "xgb_baseline.json"

# Deliberately sensible, unambitious defaults for a fast, reliable
# baseline rather than a heavily tuned final model:
#   - max_depth=6: XGBoost's own default, a reasonable middle ground
#     for tabular data with only 10 features -- no need to go deeper.
#   - learning_rate=0.1: standard starting point, balances training
#     speed against convergence quality.
#   - n_estimators=300 with early stopping: gives the model enough
#     rounds to converge without needing to hand-tune an exact count;
#     early stopping halts training once validation stops improving.
#   - scale_pos_weight left at XGBoost's default of 1, rather than set
#     to the raw negative:positive ratio -- the 4:1 sampling ratio in
#     NEGATIVE_TO_POSITIVE_RATIO above is already a deliberate choice
#     (oversampling negatives well beyond their true rarity in the full
#     candidate set) to give the model useful negative examples;
#     applying scale_pos_weight on top of an already-adjusted sample
#     would double-correct for the same imbalance.
XGB_PARAMS = dict(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",  # fast, low-memory histogram-based training
    early_stopping_rounds=20,
    random_state=42,
)

# ---------------------------------------------------------------------
# Threshold selection on FULL candidate pools.
#
# Why this is a separate evaluation from training_features.tsv: that
# file only contains the 4:1 negative-sampled subset used for training,
# which understates how many "hard negative" candidates the model will
# actually face at real inference time. Selecting a decision threshold
# against sampled negatives would therefore produce an overly
# optimistic threshold. This evaluation instead pulls each sampled
# entity's FULL candidate pool exactly as blocking produced it, and
# each entity's FULL true-match set from ground truth (not limited to
# matches that survived blocking), so the simulated F_0.5 here matches
# how the competition actually scores the submission.
# ---------------------------------------------------------------------
EVAL_SAMPLE_SEED = 123
EVAL_N_ENTITIES = 50_000  # large enough for a stable F_0.5 estimate, small enough to finish in reasonable time

EVAL_CANDIDATES_PATH = OUTPUT_DIR / "eval_candidates_full.tsv"
EVAL_SCORED_PATH = OUTPUT_DIR / "eval_scored.tsv"
BEST_THRESHOLD_PATH = OUTPUT_DIR / "best_threshold.txt"

# Coarse-to-fine grid, with finer steps concentrated near the top end.
# The trained model's very low validation logloss indicates predicted
# probabilities cluster tightly near 0 or 1, so a threshold search
# benefits from finer resolution in the high-probability region rather
# than even spacing across the full [0, 1] range.
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                  0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]

# --- Test inference ---
CLEANED_TEST_SOURCE1 = CLEANED_DIR / "test_source1_cleaned.tsv"
CLEANED_TEST_SOURCE2 = CLEANED_DIR / "test_source2_cleaned.tsv"
CLEANED_TEST_SOURCE3 = CLEANED_DIR / "test_source3_cleaned.tsv"

RAW_TEST_SOURCE2 = PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv"
RAW_TEST_SOURCE3 = PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv"

# Blocking's real test output -- the same file that is included in the
# final submission package as output/candidate_pairs.tsv.
CANDIDATES_TEST_PATH = PROJECT_ROOT / "blocking" / "blocking_outputs" / "candidate_pairs.tsv"

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"

# Separate, smaller progress interval than PROGRESS_EVERY -- the larger
# interval would never fire at all on evaluation runs with fewer total
# entities than the interval itself, leaving those runs with no visible
# progress output.
EVAL_PROGRESS_EVERY = 5_000


# =======================================================================
# The constants below support the experimental embedding-augmented
# matching variant (frozen multilingual sentence embeddings as an
# additional model feature). This variant is NOT used in the final
# submitted result -- see experimental_embeddings/README_EXPERIMENTAL.md
# for the full rationale and status. Kept here because the experimental
# scripts import these values from this shared config file.
# =======================================================================

# --- Embedding model setup ---
#
# intfloat/multilingual-e5-small: MIT licensed, 384-dim, small enough to
# run efficiently on modest hardware, and specifically trained for
# cross-lingual semantic similarity -- directly targeting the
# transliteration gap identified during blocking validation, where
# lexical name-similarity metrics (token Jaccard) collapsed sharply for
# names originally written in non-Latin scripts.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384

# E5-family models are trained with task-prefixed inputs ("query: " or
# "passage: "). For symmetric similarity (comparing two records to each
# other, rather than a search query against documents), the model card
# recommends using "query: " on both sides -- omitting this prefix is a
# known pitfall that measurably degrades embedding quality for this
# model family.
EMBEDDING_QUERY_PREFIX = "query: "

# --- Multiprocessing feature computation (experimental) ---
#
# N_WORKERS: number of worker processes for the parallelized feature-
# computation pipeline. Each worker loads its own copy of the required
# lookup data, so total memory scales with worker count, not just
# speed -- this value should be chosen conservatively relative to
# available system RAM.
N_WORKERS = 1

# Lines per task handed to a worker at once. Too small wastes time on
# inter-process overhead per task; too large hurts load balancing
# (workers finishing early sit idle while one worker still has a large
# chunk remaining).
WORKER_CHUNK_SIZE = 2_000

# Candidate cap used specifically to verify the parallelized scoring
# pipeline's correctness against the single-threaded implementation,
# before trusting the parallel version on the real test set.
PARALLEL_TEST_MAX_CANDIDATES = 50

# --- Candidate-cap sweep (experimental) ---
CAP_TEST_VALUES = [50, 100, 150, 200, 300]

# --- Test-set embeddings (experimental) ---
RAW_TEST_SOURCE1 = PROJECT_ROOT / "dataset" / "test" / "test_source1.tsv"
RAW_TEST_SOURCE2 = PROJECT_ROOT / "dataset" / "test" / "test_source2.tsv"
RAW_TEST_SOURCE3 = PROJECT_ROOT / "dataset" / "test" / "test_source3.tsv"

EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

TEST_EMBEDDINGS_MEMMAP_PATH = EMBEDDINGS_DIR / "test_embeddings.dat"
TEST_EMBEDDINGS_ID_INDEX_PATH = EMBEDDINGS_DIR / "test_embeddings_id_index.tsv"
TEST_EMBEDDINGS_META_PATH = EMBEDDINGS_DIR / "test_embeddings_meta.txt"

EMBEDDING_BATCH_SIZE = 512  # tuned for GPU throughput at full-dataset scale
EMBEDDING_PROGRESS_EVERY = 100_000  # print progress every N entities embedded

# --- Train-side embeddings (experimental) ---
RAW_TRAIN_SOURCE1 = PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv"

TRAIN_EMBED_REFS_S1_PATH = EMBEDDINGS_DIR / "train_embed_refs_s1.txt"
TRAIN_EMBED_REFS_OTHER_PATH = EMBEDDINGS_DIR / "train_embed_refs_other.txt"

TRAIN_EMBEDDINGS_MEMMAP_PATH = EMBEDDINGS_DIR / "train_embeddings.dat"
TRAIN_EMBEDDINGS_ID_INDEX_PATH = EMBEDDINGS_DIR / "train_embeddings_id_index.tsv"
TRAIN_EMBEDDINGS_META_PATH = EMBEDDINGS_DIR / "train_embeddings_meta.txt"

TRAINING_FEATURES_AUGMENTED_PATH = OUTPUT_DIR / "training_features_v2.tsv"

# --- Feature augmentation (experimental) ---
EVAL_CANDIDATES_EMBED_SIM_PATH = OUTPUT_DIR / "eval_candidates_embed_sim.tsv"

# --- 11-feature model + country-specific thresholds (experimental) ---
V2_FEATURE_COLUMNS = FEATURE_COLUMNS + ["name_address_embedding_cosine_sim"]

MODEL_V2_PATH = MODEL_DIR / "xgb_v2.json"
BEST_THRESHOLDS_BY_COUNTRY_PATH = OUTPUT_DIR / "best_thresholds_by_country.txt"

# --- Final embedding-augmented inference (experimental, not completed at full scale) ---
FINAL_CAP = 300           # candidate cap selected from the cap-sweep results
FINAL_THRESHOLD = 0.98    # threshold at which both US and India independently peaked in evaluation

TEST_EMBED_REFS_S1_PATH = EMBEDDINGS_DIR / "test_embed_refs_s1.txt"
TEST_EMBED_REFS_OTHER_PATH = EMBEDDINGS_DIR / "test_embed_refs_other.txt"