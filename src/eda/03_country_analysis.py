"""
03_country_analysis.py

Phase 1 EDA - Step 3: Country Distribution and Formatting Analysis.

Why this matters for this competition specifically:
The test set introduces France, a country label that never appears in
training. Any pipeline that hard-codes or one-hot-encodes country as a
closed set of {US, India} will silently fail on every French entity.
This script:

  1. Catalogues every distinct country label present, so we know whether
     "US" also appears as "USA", "U.S.A.", "United States", etc.
  2. Confirms exactly how France is labelled in the test set and how
     much volume it represents.
  3. Compares train vs test distributions (missingness, text length)
     split by country, to catch cases where cleaning rules built only
     from training data might not transfer to test.

Outputs:
  - eda_outputs/03_country_analysis_report.txt
  - eda_outputs/03_country_label_counts.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    COL_COUNTRY, COL_NAME, COL_ADDRESS,
    OUTPUT_DIR,
)
from io_utils import load_tsv

REPORT_PATH = OUTPUT_DIR / "03_country_analysis_report.txt"
LABEL_COUNTS_CSV = OUTPUT_DIR / "03_country_label_counts.csv"


def country_label_variation(datasets: dict):
    """
    For every source file, lists every distinct country value and its
    frequency -- surfacing casing issues (US vs us), abbreviation
    inconsistencies (US vs USA vs U.S.A.), and stray whitespace or typos
    that a naive "== 'US'" filter would miss.
    """
    lines = ["=== Country Label Variation (all distinct values, by source) ===\n"]
    all_rows = []

    for label, df in datasets.items():
        if COL_COUNTRY not in df.columns:
            continue
        counts = df[COL_COUNTRY].value_counts(dropna=False)
        lines.append(f"-- {label} --")
        for value, count in counts.items():
            display_value = value if value != "" else "(empty)"
            lines.append(f"  {display_value!r:20s}: {count:,}")
            all_rows.append({"source_file": label, "country_value": value, "count": count})
        lines.append("")

    # Returns both the human-readable report text AND the underlying
    # per-value counts as a dataframe, so main() can write the latter
    # out to a CSV for easier spreadsheet inspection alongside the
    # plain-text report.
    combined = pd.DataFrame(all_rows)
    return "\n".join(lines) + "\n", combined


def france_inspection(test_datasets: dict) -> str:
    """
    Isolates rows labelled as France (case-insensitively, checking common
    variants like "FR", "France", "FRANCE") in the test set, since the
    problem statement calls this out as a country that must be handled
    correctly despite never appearing in training.
    """
    lines = ["=== France-Specific Inspection (Test Set Only) ===\n"]
    france_variants = {"france", "fr"}

    for label, df in test_datasets.items():
        if COL_COUNTRY not in df.columns:
            continue
        mask = df[COL_COUNTRY].str.strip().str.lower().isin(france_variants)
        n_france = mask.sum()
        n_total = len(df)
        pct = 100 * n_france / n_total if n_total else 0
        lines.append(f"{label:15s}: {n_france:,} / {n_total:,} rows ({pct:.2f}%) "
                     f"labelled as France")
        if n_france > 0:
            sample = df.loc[mask, COL_COUNTRY].value_counts()
            lines.append(f"  exact label values used: {dict(sample)}")
    lines.append("")
    return "\n".join(lines) + "\n"


def train_vs_test_distribution_by_country(train_datasets: dict, test_datasets: dict) -> str:
    """
    Compares, per source and per country, missingness rate and average
    text length of business_name / business_address between train and
    test. Large discrepancies are an early warning that cleaning or
    feature-engineering rules tuned only on training data may not
    transfer to test -- most relevant for France, which has zero training
    examples to tune against, but also useful for catching drift within
    US/India between train and test.
    """
    lines = ["=== Train vs Test Distribution by Country (missingness & length) ===\n"]

    def summarise(df, source_label, split_label):
        """
        Computes one summary row per distinct country value present in
        df: row count, average name/address length, and missingness
        percentage for each field. Called once per (source file, train
        or test) combination by the loop below, so the final table has
        one row per (source, split, country) triple.
        """
        rows = []
        if COL_COUNTRY not in df.columns:
            return rows
        for country_value, group in df.groupby(df[COL_COUNTRY].str.strip()):
            name_len = group[COL_NAME].fillna("").str.len()
            addr_len = group[COL_ADDRESS].fillna("").str.len()
            name_missing_pct = 100 * (group[COL_NAME].fillna("").str.strip() == "").mean()
            addr_missing_pct = 100 * (group[COL_ADDRESS].fillna("").str.strip() == "").mean()
            rows.append({
                "source": source_label,
                "split": split_label,
                "country": country_value,
                "n_rows": len(group),
                "avg_name_len": round(name_len.mean(), 2),
                "avg_address_len": round(addr_len.mean(), 2),
                "name_missing_pct": round(name_missing_pct, 2),
                "address_missing_pct": round(addr_missing_pct, 2),
            })
        return rows

    all_rows = []
    for label, df in train_datasets.items():
        all_rows.extend(summarise(df, label.replace("train_", ""), "train"))
    for label, df in test_datasets.items():
        all_rows.extend(summarise(df, label.replace("test_", ""), "test"))

    summary_df = pd.DataFrame(all_rows)
    lines.append(summary_df.to_string(index=False))
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    # Loads all six source files (train and test separately, since the
    # France inspection and the train-vs-test comparison both need to
    # distinguish which split a file came from), then runs all three
    # country-focused checks and writes both a text report and a CSV of
    # the raw label counts.
    print("Loading files...")
    train_datasets = {
        "train_source1": load_tsv(TRAIN_SOURCE1),
        "train_source2": load_tsv(TRAIN_SOURCE2),
        "train_source3": load_tsv(TRAIN_SOURCE3),
    }
    test_datasets = {
        "test_source1": load_tsv(TEST_SOURCE1),
        "test_source2": load_tsv(TEST_SOURCE2),
        "test_source3": load_tsv(TEST_SOURCE3),
    }
    all_datasets = {**train_datasets, **test_datasets}

    report_parts = []
    variation_report, combined_counts = country_label_variation(all_datasets)
    report_parts.append(variation_report)
    report_parts.append(france_inspection(test_datasets))
    report_parts.append(train_vs_test_distribution_by_country(train_datasets, test_datasets))

    full_report = "\n".join(report_parts)
    print(full_report)

    REPORT_PATH.write_text(full_report, encoding="utf-8")
    combined_counts.to_csv(LABEL_COUNTS_CSV, index=False)
    print(f"\nFull report written to: {REPORT_PATH}")
    print(f"Label counts CSV written to: {LABEL_COUNTS_CSV}")


if __name__ == "__main__":
    main()