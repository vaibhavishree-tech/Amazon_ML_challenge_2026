"""
config.py

Central configuration for Phase 1, Step 3: Blocking (Candidate Generation).

Reads from cleaning/cleaned_outputs/ (the output of the cleaning phase)
rather than dataset/ directly, since blocking must run on cleaned text --
every signal below (token overlap, n-gram overlap) depends on the
normalization cleaning already applied (suffix expansion, transliteration,
whitespace/casing standardization).
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLEANED_DIR = PROJECT_ROOT / "cleaning" / "cleaned_outputs"

CLEANED_TRAIN_SOURCE1 = CLEANED_DIR / "train_source1_cleaned.tsv"
CLEANED_TRAIN_SOURCE2 = CLEANED_DIR / "train_source2_cleaned.tsv"
CLEANED_TRAIN_SOURCE3 = CLEANED_DIR / "train_source3_cleaned.tsv"
CLEANED_TEST_SOURCE1 = CLEANED_DIR / "test_source1_cleaned.tsv"
CLEANED_TEST_SOURCE2 = CLEANED_DIR / "test_source2_cleaned.tsv"
CLEANED_TEST_SOURCE3 = CLEANED_DIR / "test_source3_cleaned.tsv"

# The ORIGINAL (uncleaned) train source files are needed only by
# 02_measure_recall_ceiling.py, to recompute the "was the matched name
# originally non-ASCII" flag used in the transliteration-impact
# breakdown -- this flag must reflect the pre-cleaning script of the
# text, since cleaning (unidecode) already strips non-ASCII characters
# from the cleaned files by design.
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_SOURCE2_RAW = DATASET_DIR / "train" / "train_source2.tsv"
TRAIN_SOURCE3_RAW = DATASET_DIR / "train" / "train_source3.tsv"
TRAIN_GROUND_TRUTH = DATASET_DIR / "train" / "train_ground_truth.tsv"

OUTPUT_DIR = PROJECT_ROOT / "blocking" / "blocking_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = OUTPUT_DIR / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_TRAIN_PATH = OUTPUT_DIR / "candidates_train.tsv"
CANDIDATES_TEST_PATH = OUTPUT_DIR / "candidate_pairs.tsv"  # matches required submission filename

# Column names, matching cleaning/config.py.
COL_ENTITY_ID = "entity_id"
COL_NAME = "business_name"
COL_ADDRESS = "business_address"
COL_COUNTRY = "country"
COL_POSTAL_CODE = "postal_code_extracted"

COL_GT_SOURCE1_ID = "source1_entity_id"
COL_GT_MATCHED_IDS = "matched_entity_ids"

# ---------------------------------------------------------------------
# Tokenization parameters.
#
# WORD_MIN_LEN=2: excludes single-character tokens from word-token
# indexing. Evidence: the EDA phase's address token frequency table
# shows single letters (a, c, b, d, h, o, r, s) at very high frequency
# -- these are almost certainly unit/block letters (e.g. "Block A") or
# artifacts, not discriminative for blocking, and would create
# enormous, useless posting lists if indexed.
#
# NGRAM_N=4: character n-gram size used for the typo-tolerant name
# signal. Deliberately larger than the 3-grams used for SIMILARITY
# SCORING in the EDA/cleaning phases -- a larger n produces more
# distinct n-grams, which means lower document frequency per n-gram and
# smaller, more discriminative posting lists for BLOCKING specifically.
# Scoring (later, in the matching model) can still use 3-grams; blocking
# uses 4 to control posting-list size.
# ---------------------------------------------------------------------
WORD_MIN_LEN = 2
NGRAM_N = 4

# ---------------------------------------------------------------------
# Stopword-style exclusions, applied on top of automatic document-
# frequency filtering.
#
# NAME_STOPWORDS: tokens known in advance to be near-universal because
# they are the EXPANDED forms produced by the cleaning phase's legal-
# suffix mapping (pvt->private, ltd->limited, etc.), confirmed by
# cleaning's own suffix-collapse validation report to occur millions of
# times across the dataset. These are excluded directly rather than
# relying on document-frequency filtering alone to discover them.
#
# ADDRESS_STOPWORDS: drawn from the EDA phase's address token frequency
# analysis (road, street, floor, near, unit, etc.) plus the full set of
# expanded state names produced by cleaning's street/state mapping
# dictionaries, which become extremely frequent WITHIN a single country
# partition (e.g. most India addresses repeat a small set of state
# names).
# ---------------------------------------------------------------------
NAME_STOPWORDS = {
    "private", "limited", "incorporated", "corporation", "company",
    "professional", "partnership", "liability", "and", "of", "the",
    "llc", "ltd", "inc", "corp", "pvt", "llp", "lp", "pc", "co",
}

ADDRESS_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "court",
    "place", "floor", "near", "unit", "block", "door", "plot", "no",
    "city", "town", "village", "district", "of", "and",
    # expanded US state names -- see cleaning/text_cleaning_rules.py US_STATE_MAP
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new", "hampshire", "jersey", "mexico", "york", "north", "carolina",
    "dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode",
    "island", "south", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west", "wisconsin", "wyoming",
    # expanded Indian state names -- see cleaning/text_cleaning_rules.py INDIA_STATE_MAP
    "maharashtra", "telangana", "andhra", "pradesh", "gujarat",
    "tamil", "nadu", "delhi", "uttar", "bengal", "karnataka", "kerala",
    "rajasthan", "punjab", "haryana", "odisha",
}

# ---------------------------------------------------------------------
# Document-frequency filtering for indexes -- a token whose posting
# list exceeds BOTH this fraction of the partition and this absolute
# count is dropped from the index as too common to be useful (the
# smaller of the two thresholds wins, i.e. whichever cap is hit first).
#
# These values were selected after empirically testing looser and
# tighter DF settings against measured blocking recall on the training
# ground truth: looser filtering flooded candidate pools with common
# tokens without meaningfully improving recall, while filtering too
# tight caused entities to lose their only distinguishing token and
# produced measurable zero-candidate regressions. The current values
# represent the best-performing configuration found during that search.
# ---------------------------------------------------------------------
MAX_DF_FRACTION = 0.01
MAX_DF_ABSOLUTE = 8_000

# How many of an entity's rarest available tokens/n-grams to actually
# query per signal. Querying only the rarest K (rather than all tokens)
# bounds per-entity work and result-set size while preserving recall,
# since the rarest tokens are the most discriminative -- a true match
# is far more likely to share a rare, distinctive token than a common
# one.
RARE_K = 10

# Soft cap on candidates per Source 1 entity. Only applied when the raw
# union of all signals exceeds this -- when it fires, candidates are
# ranked by how many separate rarest-token "hits" contributed them
# (more independent signals agreeing = stronger candidate) and only the
# top MAX_CANDIDATES_PER_ENTITY are kept. This exists to keep
# candidate_pairs.tsv a genuinely narrowed set (as the problem statement
# requires: "the final candidate list just before the ML model scores
# them"), not a barely-filtered near-cross-product for common names.
MAX_CANDIDATES_PER_ENTITY = 500

# Batch sizes -- same purpose as in the EDA/cleaning phases: bound peak
# memory regardless of file size, and give visible progress rather than
# an apparently frozen script.
INDEX_BUILD_BATCH_SIZE = 500_000
QUERY_PROGRESS_BATCH_SIZE = 50_000

# Optional: restrict processing to specific countries, for a fast
# end-to-end sanity check before committing to a full run. Set to None
# to process every country present in the data (the real run). Example
# for a quick test: COUNTRIES_OVERRIDE = ["France"], the smallest
# test-set partition by row count among the three countries present.
COUNTRIES_OVERRIDE = None