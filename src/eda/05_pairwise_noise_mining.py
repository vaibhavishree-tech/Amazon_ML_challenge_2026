"""
05_pairwise_noise_mining.py

Phase 1 EDA - Step 5: Pairwise Noise Mining (Ground Truth Cross-Analysis).

Performance rewrite notes (this was the worst-performing script at scale):

- build_lookup originally used df.iterrows() to build a Python dict.
  This is the single slowest operation in the whole EDA suite at 5
  million rows per source, since it creates a new pandas Series object
  for every row just to read three values out of it. Rewritten to use
  dict(zip(...)) directly on the underlying columns -- zip() operates on
  the raw arrays, so this is orders of magnitude faster and uses far
  less memory.

- export_manual_review_sample originally looped over gt.iterrows() to
  build a list of (s1_id, match_id) pairs before sampling. Rewritten to
  reuse the explode-based long-format table (same pattern as scripts 01
  and 02), then use pandas' own .sample() instead of Python's
  random.sample() on a manually built list.

- compute_positive_pair_similarity originally looped over gt.iterrows()
  and, for every match, did two dict lookups plus a similarity
  computation. Rewritten to build the exploded pair table, attach names
  and addresses via .map() (a vectorized Series-to-Series lookup, unlike
  a Python loop calling dict.get() one key at a time), and only then run
  the similarity function per pair. This is still a Python-level loop at
  the very last step, because rapidfuzz's similarity functions operate
  on one string pair at a time, but the number of true-match pairs is
  bounded by your ground truth size, not the full 5-million-row source
  files, so this step stays fast.

- compute_negative_pair_similarity originally used random.choice() in a
  Python while loop, repeatedly calling dict.get(). Rewritten to draw
  all random indices at once with numpy, then vectorized-map the
  name/address columns in one pass, filtering out any accidental true
  positives afterward instead of checking membership one at a time
  inside the loop.
"""

import random
import re
import sys
from pathlib import Path
import gc

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    COL_ENTITY_ID, COL_NAME, COL_ADDRESS,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    OUTPUT_DIR,
)
from io_utils import load_tsv

# Fixed seed so the manual-review sample and negative-pair sample are
# reproducible across runs -- re-running this script produces the same
# 100 rows to read and the same 2000 negative pairs, rather than a
# different random slice each time.
RANDOM_SEED = 42
# How many true-match pairs to export for manual, by-eye reading --
# large enough to spot real patterns, small enough to actually read by
# hand in one sitting.
N_MANUAL_REVIEW_SAMPLE = 100
# How many random non-matching pairs to sample for the negative-pair
# similarity distribution, used as the baseline "these are NOT the same
# business" comparison against the positive-pair distribution below.
N_NEGATIVE_PAIRS = 2000

MANUAL_SAMPLE_CSV = OUTPUT_DIR / "05_true_match_sample_for_manual_review.csv"
POSITIVE_SIM_CSV = OUTPUT_DIR / "05_positive_pair_similarity.csv"
NEGATIVE_SIM_CSV = OUTPUT_DIR / "05_negative_pair_similarity.csv"
REPORT_PATH = OUTPUT_DIR / "05_similarity_summary_report.txt"


def explode_matched_ids(gt: pd.DataFrame) -> pd.DataFrame:
    """Same explode pattern as scripts 01 and 02."""
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"}).reset_index(drop=True)


def char_ngrams(text: str, n: int = 3) -> set:
    """
    Returns the set of character n-grams (default trigrams) for a
    string -- e.g. "abc" with n=3 gives a single 3-character window. If
    the string is shorter than n, the whole string is returned as one
    "n-gram" rather than an empty set, so very short names/addresses
    still contribute something to the Jaccard comparison below instead
    of always scoring zero overlap.
    """
    text = text.lower().strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def token_set(text: str) -> set:
    """
    Returns the set of lowercase alphanumeric tokens (words/numbers) in
    a string, used as the basis for token-level Jaccard similarity --
    a coarser, word-level counterpart to the character-level n-grams
    above.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(set_a: set, set_b: set) -> float:
    """
    Standard Jaccard similarity: size of the intersection divided by
    size of the union. Two empty sets are treated as a perfect match
    (1.0) rather than undefined, and one empty / one non-empty set is
    treated as no overlap at all (0.0), so the function always returns
    a usable number rather than raising a divide-by-zero error.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def levenshtein_ratio(a: str, b: str) -> float:
    """
    Normalized Levenshtein similarity (0 = completely different, 1 =
    identical), via rapidfuzz's C-accelerated implementation rather than
    a hand-written edit-distance function, since this is called once per
    pair and needs to be fast at scale.
    """
    return Levenshtein.normalized_similarity(a, b)


def compute_similarity_row(name_a, addr_a, name_b, addr_b) -> dict:
    """
    Computes all six similarity metrics (Levenshtein, token Jaccard,
    and character-trigram Jaccard, each for both name and address) for
    one pair of records in a single call, returning them as a dict
    whose keys become the output CSV's column names. This is the one
    function called per-pair by both compute_positive_pair_similarity
    and compute_negative_pair_similarity below, so both the true-match
    and random-pair similarity distributions are measured identically.
    """
    return {
        "name_levenshtein": round(levenshtein_ratio(name_a, name_b), 4),
        "name_token_jaccard": round(jaccard(token_set(name_a), token_set(name_b)), 4),
        "name_char3gram_jaccard": round(jaccard(char_ngrams(name_a), char_ngrams(name_b)), 4),
        "address_levenshtein": round(levenshtein_ratio(addr_a, addr_b), 4),
        "address_token_jaccard": round(jaccard(token_set(addr_a), token_set(addr_b)), 4),
        "address_char3gram_jaccard": round(jaccard(char_ngrams(addr_a), char_ngrams(addr_b)), 4),
    }


def build_lookup(df: pd.DataFrame) -> dict:
    """
    Builds entity_id -> (name, address) using dict(zip(...)) directly on
    the underlying pandas columns, rather than df.iterrows().

    zip() iterates over the raw column arrays in C, without constructing
    a pandas Series per row -- this is the single biggest performance
    fix in this script, since the original iterrows() version was doing
    this over files with millions of rows.
    """
    return dict(zip(
        df[COL_ENTITY_ID],
        zip(df[COL_NAME].fillna(""), df[COL_ADDRESS].fillna("")),
    ))


def export_manual_review_sample(exploded: pd.DataFrame, lookups: dict):
    """
    Samples true-match pairs using pandas' built-in .sample(), which is
    implemented in C and avoids building an intermediate Python list of
    every pair before sampling, as the original version did.
    """
    sample_size = min(N_MANUAL_REVIEW_SAMPLE, len(exploded))
    sampled = exploded.sample(n=sample_size, random_state=RANDOM_SEED)

    rows = []
    for _, row in sampled.iterrows():
        # This loop runs only over the small sample (up to 100 rows),
        # not the full dataset, so a simple loop here is fine and does
        # not need vectorization.
        s1_id = row[COL_GT_SOURCE1_ID]
        match_id = row["matched_id"]
        s1_name, s1_addr = lookups["source1"].get(s1_id, ("", ""))
        source_key = "source2" if match_id.startswith("S2-") else "source3"
        m_name, m_addr = lookups[source_key].get(match_id, ("", ""))
        rows.append({
            "source1_entity_id": s1_id,
            "source1_name": s1_name,
            "source1_address": s1_addr,
            "matched_entity_id": match_id,
            "matched_name": m_name,
            "matched_address": m_addr,
            "name_mutation_notes": "",
            "address_mutation_notes": "",
        })

    sample_df = pd.DataFrame(rows)
    sample_df.to_csv(MANUAL_SAMPLE_CSV, index=False)
    print(f"Manual review sample ({sample_size} pairs) written to: {MANUAL_SAMPLE_CSV}")
    print("Open this file and fill in the mutation-notes columns by hand "
          "while reading through the pairs.")


def compute_positive_pair_similarity(exploded: pd.DataFrame, lookups: dict) -> pd.DataFrame:
    """
    Attaches name/address text via .map() -- a vectorized Series lookup
    against the dict built in build_lookup -- instead of calling
    dict.get() inside a Python loop for every row. The final similarity
    computation still runs per pair (rapidfuzz operates on one string
    pair at a time), but the number of true-match pairs is bounded by
    your ground truth size, not by the 5-million-row source files, so
    this remains fast.
    """
    working = exploded.copy()

    s1_lookup = lookups["source1"]
    working["s1_name"] = working[COL_GT_SOURCE1_ID].map(lambda i: s1_lookup.get(i, ("", ""))[0])
    working["s1_addr"] = working[COL_GT_SOURCE1_ID].map(lambda i: s1_lookup.get(i, ("", ""))[1])

    def other_lookup(match_id):
        source_key = "source2" if match_id.startswith("S2-") else "source3"
        return lookups[source_key].get(match_id, ("", ""))

    other_pairs = working["matched_id"].map(other_lookup)
    working["other_name"] = other_pairs.map(lambda t: t[0])
    working["other_addr"] = other_pairs.map(lambda t: t[1])

    # result_type="expand" turns the dict returned by
    # compute_similarity_row into separate output columns (rather than
    # a single column of dicts), giving a proper tabular result in one
    # call instead of a manual column-by-column assignment afterward.
    sim_rows = working.apply(
        lambda r: compute_similarity_row(r["s1_name"], r["s1_addr"], r["other_name"], r["other_addr"]),
        axis=1,
        result_type="expand",
    )
    sim_rows[COL_GT_SOURCE1_ID] = working[COL_GT_SOURCE1_ID].values
    sim_rows["matched_entity_id"] = working["matched_id"].values

    sim_rows.to_csv(POSITIVE_SIM_CSV, index=False)
    return sim_rows


def compute_negative_pair_similarity(
    exploded: pd.DataFrame, s1_ids: pd.Series, s2_ids: pd.Series, s3_ids: pd.Series, lookups: dict
) -> pd.DataFrame:
    """
    Rewritten to draw random indices with numpy in one batch instead of
    calling random.choice() inside a Python while loop. True-positive
    pairs are filtered out afterward using a vectorized isin-style
    check against a set, rather than checking set membership once per
    loop iteration before deciding whether to keep the sample.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    true_pairs = set(zip(exploded[COL_GT_SOURCE1_ID], exploded["matched_id"]))

    s1_array = s1_ids.to_numpy()
    other_array = np.concatenate([s2_ids.to_numpy(), s3_ids.to_numpy()])

    # Oversample slightly so that after removing accidental true
    # positives we still have at least N_NEGATIVE_PAIRS rows.
    oversample_size = int(N_NEGATIVE_PAIRS * 1.2)
    sampled_s1 = rng.choice(s1_array, size=oversample_size, replace=True)
    sampled_other = rng.choice(other_array, size=oversample_size, replace=True)

    candidate = pd.DataFrame({
        COL_GT_SOURCE1_ID: sampled_s1,
        "random_other_id": sampled_other,
    })

    # Checks each sampled (s1_id, other_id) pair against the true_pairs
    # set built above, so any random pair that happens to actually be a
    # real match gets excluded -- the negative sample must only contain
    # genuine non-matches for the comparison to mean anything.
    is_true_pair = candidate.apply(
        lambda r: (r[COL_GT_SOURCE1_ID], r["random_other_id"]) in true_pairs, axis=1
    )
    candidate = candidate[~is_true_pair].head(N_NEGATIVE_PAIRS).reset_index(drop=True)

    s1_lookup = lookups["source1"]
    candidate["s1_name"] = candidate[COL_GT_SOURCE1_ID].map(lambda i: s1_lookup.get(i, ("", ""))[0])
    candidate["s1_addr"] = candidate[COL_GT_SOURCE1_ID].map(lambda i: s1_lookup.get(i, ("", ""))[1])

    def other_lookup(other_id):
        source_key = "source2" if other_id.startswith("S2-") else "source3"
        return lookups[source_key].get(other_id, ("", ""))

    other_pairs = candidate["random_other_id"].map(other_lookup)
    candidate["other_name"] = other_pairs.map(lambda t: t[0])
    candidate["other_addr"] = other_pairs.map(lambda t: t[1])

    sim_rows = candidate.apply(
        lambda r: compute_similarity_row(r["s1_name"], r["s1_addr"], r["other_name"], r["other_addr"]),
        axis=1,
        result_type="expand",
    )
    sim_rows[COL_GT_SOURCE1_ID] = candidate[COL_GT_SOURCE1_ID].values
    sim_rows["random_other_id"] = candidate["random_other_id"].values

    sim_rows.to_csv(NEGATIVE_SIM_CSV, index=False)
    return sim_rows


def summarise_distributions(positive_df: pd.DataFrame, negative_df: pd.DataFrame) -> str:
    """
    For each of the six similarity metrics, prints side-by-side summary
    statistics (min, p10, median, p90, max) for the positive (true
    match) and negative (random, non-matching) pair distributions. This
    is the key comparison for judging whether a given metric actually
    separates true matches from non-matches -- a metric where the two
    distributions barely overlap is a strong candidate signal for
    blocking/matching; one where they're nearly identical is a weak
    signal on its own.
    """
    lines = ["=== Positive vs Negative Pair Similarity Summary ===\n"]
    metrics = [
        "name_levenshtein", "name_token_jaccard", "name_char3gram_jaccard",
        "address_levenshtein", "address_token_jaccard", "address_char3gram_jaccard",
    ]
    for metric in metrics:
        pos_stats = positive_df[metric].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        neg_stats = negative_df[metric].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        lines.append(f"-- {metric} --")
        lines.append(f"  positive pairs: min={pos_stats['min']:.3f}, "
                     f"p10={pos_stats['10%']:.3f}, median={pos_stats['50%']:.3f}, "
                     f"p90={pos_stats['90%']:.3f}, max={pos_stats['max']:.3f}")
        lines.append(f"  negative pairs: min={neg_stats['min']:.3f}, "
                     f"p10={neg_stats['10%']:.3f}, median={neg_stats['50%']:.3f}, "
                     f"p90={neg_stats['90%']:.3f}, max={neg_stats['max']:.3f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    # Only loads entity_id/name/address (via usecols=cols) -- country is
    # not used anywhere in this script's similarity computations, so
    # it's skipped to reduce memory at multi-million-row scale.
    print("Loading files...")
    cols = [COL_ENTITY_ID, COL_NAME, COL_ADDRESS]
    s1 = load_tsv(TRAIN_SOURCE1, usecols=cols)
    s2 = load_tsv(TRAIN_SOURCE2, usecols=cols)
    s3 = load_tsv(TRAIN_SOURCE3, usecols=cols)
    gt = load_tsv(TRAIN_GROUND_TRUTH)

    print("Building lookup dictionaries...")
    lookups = {
        "source1": build_lookup(s1),
        "source2": build_lookup(s2),
        "source3": build_lookup(s3),
    }

    # Keep only the entity_id columns needed later for negative
    # sampling, then free the full dataframes -- name and address text
    # is now fully captured inside the lookup dicts, so holding the
    # original dataframes in memory alongside the dicts is redundant
    # and roughly doubles memory usage for no benefit.
    s1_ids = s1[COL_ENTITY_ID]
    s2_ids = s2[COL_ENTITY_ID]
    s3_ids = s3[COL_ENTITY_ID]
    del s1, s2, s3
    gc.collect()

    exploded = explode_matched_ids(gt)

    export_manual_review_sample(exploded, lookups)

    print("Computing positive-pair similarity...")
    positive_df = compute_positive_pair_similarity(exploded, lookups)

    print("Computing negative-pair similarity...")
    negative_df = compute_negative_pair_similarity(
        exploded, s1_ids, s2_ids, s3_ids, lookups
    )

    report = summarise_distributions(positive_df, negative_df)
    print(report)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Positive pair similarity CSV: {POSITIVE_SIM_CSV}")
    print(f"Negative pair similarity CSV: {NEGATIVE_SIM_CSV}")
    print(f"Summary report: {REPORT_PATH}")


if __name__ == "__main__":
    main()