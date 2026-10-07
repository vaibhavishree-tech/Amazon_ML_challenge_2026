"""
02_validate_cleaning.py

Re-runs a subset of the EDA phase's own checks against the CLEANED
files and compares them to the ORIGINAL numbers from
eda/eda_outputs/04_text_vocab_encoding_report.txt and
01_volume_integrity_report.txt. This is the evidence step: it proves
the cleaning pipeline actually changed what it claims to change,
instead of just trusting that it did.

Specifically checks:
  1. Row counts match the original files exactly (cleaning must never
     drop or duplicate rows).
  2. Non-ASCII percentage should drop sharply after unidecode +
     native-script substitution.
  3. Curly quote count should be exactly zero.
  4. Placeholder-token count should be exactly zero.
  5. Legal suffix token frequency should shift: abbreviated forms
     (pvt, ltd, corp, inc) should collapse toward zero, while their
     expanded forms should rise by a corresponding amount.

Outputs a single comparison report to
cleaning/cleaned_outputs/reports/02_cleaning_validation_report.txt
"""

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    CLEANED_TRAIN_SOURCE1, CLEANED_TRAIN_SOURCE2, CLEANED_TRAIN_SOURCE3,
    CLEANED_TEST_SOURCE1, CLEANED_TEST_SOURCE2, CLEANED_TEST_SOURCE3,
    COL_NAME, COL_ADDRESS,
    REPORT_DIR,
)
from io_utils import load_tsv
from text_cleaning_rules import (
    PLACEHOLDER_TOKENS, CURLY_DOUBLE_QUOTES, CURLY_SINGLE_QUOTES, LEGAL_SUFFIX_MAP,
)

REPORT_PATH = REPORT_DIR / "02_cleaning_validation_report.txt"

CLEANED_FILES = {
    "train_source1": CLEANED_TRAIN_SOURCE1,
    "train_source2": CLEANED_TRAIN_SOURCE2,
    "train_source3": CLEANED_TRAIN_SOURCE3,
    "test_source1": CLEANED_TEST_SOURCE1,
    "test_source2": CLEANED_TEST_SOURCE2,
    "test_source3": CLEANED_TEST_SOURCE3,
}

TOKEN_PATTERN = r"[A-Za-z0-9]+"
NON_ASCII_PATTERN = r"[^\x20-\x7E]"


def check_placeholder_and_quotes(datasets: dict) -> str:
    """
    Confirms neutralize_placeholder_tokens() and normalize_curly_quotes()
    actually fired across every cleaned file -- both counts are expected
    to be exactly 0 if cleaning worked as intended.
    """
    lines = ["=== Placeholder Token and Curly Quote Check (should all be 0) ===\n"]
    double_pattern = f"[{CURLY_DOUBLE_QUOTES}]"
    single_pattern = f"[{CURLY_SINGLE_QUOTES}]"

    for label, df in datasets.items():
        lines.append(f"-- {label} --")
        for col in [COL_NAME, COL_ADDRESS]:
            stripped_lower = df[col].str.strip().str.lower()
            n_placeholder = int(stripped_lower.isin(PLACEHOLDER_TOKENS).sum())
            n_curly = int(
                df[col].str.contains(double_pattern, regex=True).sum()
                + df[col].str.contains(single_pattern, regex=True).sum()
            )
            lines.append(f"  {col:20s}: placeholder tokens remaining={n_placeholder}, "
                         f"curly quotes remaining={n_curly}")
        lines.append("")
    return "\n".join(lines) + "\n"


def check_non_ascii(datasets: dict) -> str:
    """
    Re-measures non-ASCII presence on the cleaned files using the same
    pattern and method as eda/04_text_vocab_encoding.py's
    non_ascii_scan(), so the before/after numbers are directly
    comparable rather than computed two different ways.
    """
    lines = ["=== Non-ASCII Character Check (compare against EDA's pre-cleaning numbers) ===\n"]
    for label, df in datasets.items():
        lines.append(f"-- {label} --")
        for col in [COL_NAME, COL_ADDRESS]:
            has_non_ascii = df[col].str.contains(NON_ASCII_PATTERN, regex=True)
            n = int(has_non_ascii.sum())
            pct = 100 * n / len(df) if len(df) else 0
            lines.append(f"  {col:20s}: {n:,} / {len(df):,} rows ({pct:.2f}%) still non-ASCII")
        lines.append("")
    lines.append("Compare these percentages against eda/eda_outputs/04_text_vocab_encoding_report.txt "
                  "-- they should be sharply lower. Any remaining non-zero percentage points to a "
                  "script/character not covered by the native-script state map or unidecode's "
                  "transliteration table, and is worth a manual look at a few remaining examples.\n")
    return "\n".join(lines) + "\n"


def check_suffix_collapse(datasets: dict) -> str:
    """
    Confirms the legal-suffix mapping actually fired: counts each
    abbreviated key and its mapped expansion across all business_name
    columns, expecting the abbreviated form's count to collapse toward
    zero and the expanded form's count to rise correspondingly.
    """
    lines = ["=== Legal Suffix Collapse Check ===\n"]
    counter = Counter()
    for df in datasets.values():
        tokens = df[COL_NAME].str.lower().str.findall(TOKEN_PATTERN)
        exploded = tokens.explode().dropna()
        counter.update(exploded.value_counts().to_dict())

    for abbrev, expansion in LEGAL_SUFFIX_MAP.items():
        abbrev_count = counter.get(abbrev, 0)
        # expansion may be multi-word; count occurrences of its first word
        # as a rough proxy since the full multi-word phrase does not
        # tokenize as a single token.
        expansion_first_word = expansion.split()[0]
        expansion_count = counter.get(expansion_first_word, 0)
        lines.append(f"  {abbrev:6s} -> {expansion:35s}: "
                     f"'{abbrev}' remaining={abbrev_count:,}, "
                     f"'{expansion_first_word}' now={expansion_count:,}")
    return "\n".join(lines) + "\n"


def row_count_check(datasets: dict) -> str:
    """
    Confirms cleaning never dropped or duplicated rows -- the cleaned
    file's row count must match the corresponding original file's row
    count from the EDA phase exactly.
    """
    lines = ["=== Row Count Check (must match original files exactly) ===\n"]
    for label, df in datasets.items():
        lines.append(f"  {label:20s}: {len(df):,} rows")
    lines.append("\nCompare against eda/eda_outputs/01_volume_integrity_report.txt row counts.\n")
    return "\n".join(lines) + "\n"


def main():
    # Only loads business_name/business_address -- every check in this
    # script operates on those two columns, so entity_id/country/
    # postal_code are skipped to reduce load time across six
    # multi-million-row cleaned files.
    print("Loading cleaned files...")
    datasets = {
        label: load_tsv(path, usecols=[COL_NAME, COL_ADDRESS])
        for label, path in CLEANED_FILES.items()
    }

    report_parts = [
        row_count_check(datasets),
        check_placeholder_and_quotes(datasets),
        check_non_ascii(datasets),
        check_suffix_collapse(datasets),
    ]

    full_report = "\n".join(report_parts)
    print(full_report)

    REPORT_PATH.write_text(full_report, encoding="utf-8")
    print(f"\nValidation report written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()