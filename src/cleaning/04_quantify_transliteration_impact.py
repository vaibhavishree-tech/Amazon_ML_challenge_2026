"""
04_quantify_transliteration_impact.py

Phase 1, Step 2 - Validation Script 4: Quantifying the Transliteration Gap.

03_validate_transliteration_sample.py showed, by eye, that unidecode
converges well for accented Latin (French) names but poorly for Indic
scripts (Devanagari, Tamil, Odia, Gujarati, etc.). This script puts a
number on that gap instead of relying on 30 rows of visual judgment.

What it does:
  For every true-match pair in the training ground truth, compute name
  similarity (Levenshtein, token Jaccard, character 3-gram Jaccard)
  between the CLEANED Source 1 name and the CLEANED matched name, then
  split the results into two groups by whether the matched side's
  ORIGINAL (pre-cleaning) name contained non-ASCII characters.

Why this design:
  - Uses the CLEANED files (output of 01_clean_source_files.py) for the
    actual similarity computation -- this avoids re-running unidecode,
    which is the slowest step in the whole pipeline, since it was
    already run once and saved.
  - Uses the ORIGINAL raw files only to derive the ascii/non-ascii
    label per pair, since that label must reflect what the text looked
    like BEFORE cleaning, not after.
  - If cleaning worked equally well regardless of script, the two
    groups' similarity distributions should look similar. A large gap
    confirms business_name is a weaker signal for the non-ASCII-origin
    subset even after cleaning, which should inform how heavily
    blocking/matching leans on address, country, and postal code for
    those specific pairs.

Processing is batched, matching the pattern used throughout this
project, so peak memory stays bounded regardless of the ~7.6 million
true-match pairs in the training ground truth.

Output:
  cleaning/cleaned_outputs/reports/04_transliteration_impact_report.txt
  cleaning/cleaned_outputs/reports/04_transliteration_impact_pairs.csv
    (per-pair scores + ascii/non-ascii label, kept for further analysis)
"""

import gc
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import Levenshtein

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    CLEANED_TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE3,
    COL_ENTITY_ID, COL_NAME,
    COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS,
    REPORT_DIR,
)
from io_utils import load_tsv

REPORT_PATH = REPORT_DIR / "04_transliteration_impact_report.txt"
PAIRS_CSV = REPORT_DIR / "04_transliteration_impact_pairs.csv"

BATCH_SIZE = 500_000
NON_ASCII_PATTERN = re.compile(r"[^\x20-\x7E]")

# A pair scoring below this on Levenshtein similarity is treated as
# "not usefully similar" for the practical threshold check at the end
# of the report -- gives a concrete, decision-relevant number rather
# than just comparing distribution shapes.
LOW_SIMILARITY_THRESHOLD = 0.5


def explode_matched_ids(gt: pd.DataFrame) -> pd.DataFrame:
    """Same explode pattern used throughout the EDA and cleaning phases."""
    working = gt[[COL_GT_SOURCE1_ID, COL_GT_MATCHED_IDS]].copy()
    working[COL_GT_MATCHED_IDS] = working[COL_GT_MATCHED_IDS].str.split(",")
    exploded = working.explode(COL_GT_MATCHED_IDS)
    exploded[COL_GT_MATCHED_IDS] = exploded[COL_GT_MATCHED_IDS].str.strip()
    exploded = exploded[exploded[COL_GT_MATCHED_IDS].fillna("") != ""]
    return exploded.rename(columns={COL_GT_MATCHED_IDS: "matched_id"}).reset_index(drop=True)


def build_name_lookup(df: pd.DataFrame) -> dict:
    """
    entity_id -> business_name, built with dict(zip(...)) rather than
    iterrows(), for the same performance reason established during the
    EDA phase: zip() reads the raw columns directly instead of
    reconstructing a pandas Series object per row.
    """
    return dict(zip(df[COL_ENTITY_ID], df[COL_NAME]))


def char_ngrams(text: str, n: int = 3) -> set:
    """Character n-gram set for one string -- same definition used throughout the project."""
    text = text.strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def token_set(text: str) -> set:
    """Lowercase alphanumeric token set for one string -- same definition used throughout the project."""
    return set(re.findall(r"[a-z0-9]+", text))


def jaccard(set_a: set, set_b: set) -> float:
    """Standard Jaccard similarity, with both-empty treated as a perfect match (1.0)."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_batch_similarity(names_a: list, names_b: list) -> dict:
    """
    Computes three similarity metrics for a batch of (name_a, name_b)
    pairs using zip() + list comprehensions rather than
    DataFrame.apply(). apply() carries per-row overhead of its own on
    top of the actual computation; looping over plain Python lists
    avoids that overhead, which matters at a few hundred thousand pairs
    per batch, several million pairs in total.
    """
    lev = [Levenshtein.normalized_similarity(a, b) for a, b in zip(names_a, names_b)]
    tok_jac = [jaccard(token_set(a), token_set(b)) for a, b in zip(names_a, names_b)]
    char_jac = [jaccard(char_ngrams(a), char_ngrams(b)) for a, b in zip(names_a, names_b)]
    return {
        "name_levenshtein": lev,
        "name_token_jaccard": tok_jac,
        "name_char3gram_jaccard": char_jac,
    }


def summarize(df: pd.DataFrame) -> str:
    """
    Builds the final report text: overall ASCII/non-ASCII split, then
    per-metric distribution statistics for each group side by side, and
    finally the practical "below threshold" percentage used as a
    concrete decision signal for downstream blocking/matching design.
    """
    lines = ["=== Transliteration Impact: Name Similarity, ASCII vs Non-ASCII Matched Side ===\n"]

    n_total = len(df)
    n_non_ascii = int(df["match_was_non_ascii"].sum())
    n_ascii = n_total - n_non_ascii
    lines.append(f"Total true-match pairs: {n_total:,}")
    lines.append(f"  matched side originally ASCII:      {n_ascii:,} ({100*n_ascii/n_total:.2f}%)")
    lines.append(f"  matched side originally non-ASCII:  {n_non_ascii:,} ({100*n_non_ascii/n_total:.2f}%)")
    lines.append("")

    metrics = ["name_levenshtein", "name_token_jaccard", "name_char3gram_jaccard"]
    ascii_group = df[~df["match_was_non_ascii"]]
    non_ascii_group = df[df["match_was_non_ascii"]]

    for metric in metrics:
        a_stats = ascii_group[metric].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        n_stats = non_ascii_group[metric].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        lines.append(f"-- {metric} --")
        lines.append(f"  originally ASCII     : median={a_stats['50%']:.3f}, "
                     f"p10={a_stats['10%']:.3f}, p90={a_stats['90%']:.3f}, mean={a_stats['mean']:.3f}")
        lines.append(f"  originally non-ASCII : median={n_stats['50%']:.3f}, "
                     f"p10={n_stats['10%']:.3f}, p90={n_stats['90%']:.3f}, mean={n_stats['mean']:.3f}")
        gap = a_stats["50%"] - n_stats["50%"]
        lines.append(f"  median gap (ascii - non_ascii): {gap:.3f}")
        lines.append("")

    low_sim_non_ascii = (non_ascii_group["name_levenshtein"] < LOW_SIMILARITY_THRESHOLD).mean() * 100
    low_sim_ascii = (ascii_group["name_levenshtein"] < LOW_SIMILARITY_THRESHOLD).mean() * 100
    lines.append(f"=== Practical Threshold Check (Levenshtein < {LOW_SIMILARITY_THRESHOLD}) ===\n")
    lines.append(f"  originally ASCII pairs below threshold:     {low_sim_ascii:.2f}%")
    lines.append(f"  originally non-ASCII pairs below threshold: {low_sim_non_ascii:.2f}%")
    lines.append("")
    lines.append(
        "If the non-ASCII percentage above is substantially higher, business_name "
        "similarity is not a reliable signal for that subset of pairs even after cleaning, "
        "and blocking/matching should weight address, country, and postal code more heavily "
        "for entities whose name was originally written in a non-Latin script.\n"
    )

    return "\n".join(lines) + "\n"


def main():
    # Raw (uncleaned) source2/3 names are loaded ONLY to determine the
    # ascii/non-ascii label per matched ID -- the actual similarity
    # scoring below uses the CLEANED names, loaded separately.
    print("Loading original (uncleaned) files -- used only to label ascii/non-ascii...")
    orig_s2 = load_tsv(TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    orig_s3 = load_tsv(TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])

    print("Loading cleaned files -- used for the actual similarity computation...")
    clean_s1 = load_tsv(CLEANED_TRAIN_SOURCE1, usecols=[COL_ENTITY_ID, COL_NAME])
    clean_s2 = load_tsv(CLEANED_TRAIN_SOURCE2, usecols=[COL_ENTITY_ID, COL_NAME])
    clean_s3 = load_tsv(CLEANED_TRAIN_SOURCE3, usecols=[COL_ENTITY_ID, COL_NAME])

    gt = load_tsv(TRAIN_GROUND_TRUTH)
    exploded = explode_matched_ids(gt)
    del gt
    gc.collect()

    print("Building lookups...")
    orig_name_lookup = {}
    orig_name_lookup.update(dict(zip(orig_s2[COL_ENTITY_ID], orig_s2[COL_NAME])))
    orig_name_lookup.update(dict(zip(orig_s3[COL_ENTITY_ID], orig_s3[COL_NAME])))
    del orig_s2, orig_s3
    gc.collect()

    clean_s1_lookup = build_name_lookup(clean_s1)
    clean_s2_lookup = build_name_lookup(clean_s2)
    clean_s3_lookup = build_name_lookup(clean_s3)
    del clean_s1, clean_s2, clean_s3
    gc.collect()

    def get_clean_other_name(match_id):
        return clean_s2_lookup.get(match_id) if match_id.startswith("S2-") else clean_s3_lookup.get(match_id)

    n_rows = len(exploded)
    n_batches = (n_rows // BATCH_SIZE) + 1
    print(f"Computing similarity over {n_rows:,} true-match pairs in {n_batches} batch(es)...")

    result_frames = []
    for start in range(0, n_rows, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_rows)
        batch = exploded.iloc[start:end].copy()

        batch["s1_name_clean"] = batch[COL_GT_SOURCE1_ID].map(clean_s1_lookup.get).fillna("")
        batch["match_name_clean"] = batch["matched_id"].map(get_clean_other_name).fillna("")
        batch["match_name_original"] = batch["matched_id"].map(orig_name_lookup.get).fillna("")

        batch["match_was_non_ascii"] = batch["match_name_original"].apply(
            lambda s: bool(NON_ASCII_PATTERN.search(s))
        )

        sims = compute_batch_similarity(
            batch["s1_name_clean"].tolist(), batch["match_name_clean"].tolist()
        )
        for key, values in sims.items():
            batch[key] = values

        keep_cols = [
            COL_GT_SOURCE1_ID, "matched_id", "match_was_non_ascii",
            "name_levenshtein", "name_token_jaccard", "name_char3gram_jaccard",
        ]
        result_frames.append(batch[keep_cols])

        print(f"    batch {start:,}-{end:,} done")
        # Frees this batch's intermediate columns before the next batch
        # starts, so peak memory stays bounded to one batch at a time
        # (plus the growing list of already-trimmed result frames,
        # which only keep the final handful of needed columns).
        del batch, sims
        gc.collect()

    all_results = pd.concat(result_frames, ignore_index=True)
    del result_frames
    gc.collect()

    all_results.to_csv(PAIRS_CSV, index=False)
    print(f"\nPer-pair results written to: {PAIRS_CSV}")

    report = summarize(all_results)
    print(report)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Summary report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()