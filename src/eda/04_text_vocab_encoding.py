"""
04_text_vocab_encoding.py

Phase 1 EDA - Step 4: Text Length, Vocabulary and Encoding Analysis.

Performance rewrite notes:

- token_frequency originally built a Python collections.Counter by
  looping over every row, calling a Python tokenize() function, and
  calling counter.update() once per row. At 5 million rows per source
  across 6 source files, that is tens of millions of Python function
  calls.

  Rewritten to use str.lower().str.findall(pattern) -- a single
  vectorized regex pass across the whole column, producing a column of
  token lists -- followed by .explode() to get one row per token, then
  .value_counts(), which is a C-level hash-based count. This does the
  same work with far less Python-interpreter overhead.

- non_ascii_scan originally used .apply() with a Python lambda
  (s.isascii()). Rewritten to use str.contains() with a Unicode-aware
  regex pattern matching any character outside the ASCII range, which
  is a single vectorized pass instead of one function call per row.
"""

import re
import sys
from pathlib import Path
import gc
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    COL_NAME, COL_ADDRESS,
    OUTPUT_DIR,
)
from io_utils import load_tsv

REPORT_PATH = OUTPUT_DIR / "04_text_vocab_encoding_report.txt"
NAME_TOKEN_CSV = OUTPUT_DIR / "04_name_token_frequency.csv"
ADDRESS_TOKEN_CSV = OUTPUT_DIR / "04_address_token_frequency.csv"

# Thresholds used only for flagging unusually short/long names in the
# report below -- these are reporting cutoffs, not filters applied to
# the data itself, so changing them only changes what gets called out
# as "short"/"long" in the printed summary.
SHORT_NAME_CHAR_THRESHOLD = 4
LONG_NAME_CHAR_THRESHOLD = 80

# Matches runs of letters/digits -- the tokenization unit used for both
# the business_name and business_address frequency counts below.
TOKEN_PATTERN = r"[A-Za-z0-9]+"

# Matches a standalone 5-6 digit run, used as a rough proxy for "looks
# like a postal/PIN code" when scanning address text for code presence.
POSTAL_CODE_PATTERN = r"\b\d{5,6}\b"

# Any character that is NOT in the printable ASCII range (0x20-0x7E).
# Used as a vectorized substitute for the original str.isascii() lambda.
NON_ASCII_PATTERN = r"[^\x20-\x7E]"


def length_distribution_report(datasets: dict) -> str:
    """
    Unchanged logic -- .str.len() and .str.split() length counts were
    already vectorized in the original version and are not the
    bottleneck at this scale.
    """
    lines = ["=== Length Distributions ===\n"]
    for label, df in datasets.items():
        lines.append(f"-- {label} --")
        for col in [COL_NAME, COL_ADDRESS]:
            if col not in df.columns:
                continue
            text = df[col]
            char_len = text.str.len()
            # Counts words by counting whitespace-separated runs directly via
            # regex, instead of str.split(), which builds a full list of
            # substrings per row in memory before counting them. This gets the
            # same word count without that intermediate list-of-lists.
            word_count = text.str.count(r"\S+")

            lines.append(f"  {col}:")
            lines.append(
                f"    char length -> min={char_len.min()}, "
                f"p25={char_len.quantile(0.25):.1f}, "
                f"median={char_len.median():.1f}, "
                f"mean={char_len.mean():.1f}, "
                f"p75={char_len.quantile(0.75):.1f}, "
                f"max={char_len.max()}"
            )
            lines.append(
                f"    word count  -> min={word_count.min()}, "
                f"median={word_count.median():.1f}, "
                f"mean={word_count.mean():.1f}, "
                f"max={word_count.max()}"
            )

            if col == COL_NAME:
                n_short = int((char_len < SHORT_NAME_CHAR_THRESHOLD).sum())
                n_long = int((char_len > LONG_NAME_CHAR_THRESHOLD).sum())
                lines.append(
                    f"    flagged: {n_short:,} names shorter than "
                    f"{SHORT_NAME_CHAR_THRESHOLD} chars, "
                    f"{n_long:,} names longer than {LONG_NAME_CHAR_THRESHOLD} chars"
                )
        lines.append("")
    return "\n".join(lines) + "\n"

def token_frequency(datasets: dict, column: str, top_n: int = 50, batch_size: int = 500_000):
    """
    Computes token frequency without ever holding a full exploded token
    table in memory at once.

    Why batching is necessary here: explode() on a full 5-million-row
    column can produce tens of millions of token rows in a single
    operation, and pandas runs an internal type-inference pass on the
    result that adds further memory overhead on top of the data itself.
    Splitting each dataframe into row batches (default 500,000 rows)
    means explode() only ever has to handle a few million tokens at
    once, capping peak memory regardless of how large the source file
    is.

    Running total: value_counts() on each batch is added into a
    running pd.Series via .add(fill_value=0), so only the running
    total (bounded by vocabulary size, not row count) and one batch's
    worth of tokens are ever in memory together.

    gc.collect() is called after each batch to force Python to release
    memory back to the OS immediately, rather than waiting for garbage
    collection to happen on its own schedule -- important when peak
    memory is already tight.
    """
    total_counts = pd.Series(dtype="int64")

    for label, df in datasets.items():
        if column not in df.columns:
            continue

        n_rows = len(df)
        n_batches = (n_rows // batch_size) + 1
        print(f"  tokenizing {label}.{column}: {n_rows:,} rows in {n_batches} batch(es)")

        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            batch = df[column].iloc[start:end]

            tokens = batch.str.lower().str.findall(TOKEN_PATTERN)
            exploded = tokens.explode().dropna()
            batch_counts = exploded.value_counts()

            total_counts = total_counts.add(batch_counts, fill_value=0)

            # Explicitly drop references to the batch's intermediate
            # data and force garbage collection, so memory is freed
            # before the next batch starts rather than accumulating.
            del tokens, exploded, batch_counts, batch
            gc.collect()

        # Free the full column from this dataset once all its batches
        # are done, before moving to the next dataset.
        gc.collect()

    total_counts = total_counts.sort_values(ascending=False).astype("int64")
    freq_df = total_counts.rename_axis("token").reset_index(name="count")

    lines = [f"Top {top_n} tokens in {column} (across all provided sources):\n"]
    for _, row in freq_df.head(top_n).iterrows():
        lines.append(f"  {row['token']:20s}: {row['count']:,}")

    # Returns both the printable top-N lines AND the full frequency
    # dataframe, so main() can write the complete (not just top-50)
    # token counts out to CSV for later use when building cleaning
    # dictionaries.
    return lines, freq_df


def postal_code_pattern_check(datasets: dict) -> str:
    """Already vectorized in the original (str.contains with regex) -- unchanged."""
    lines = ["=== Postal / PIN Code Presence in Addresses ===\n"]
    for label, df in datasets.items():
        if COL_ADDRESS not in df.columns:
            continue
        addr = df[COL_ADDRESS]
        has_code = addr.str.contains(POSTAL_CODE_PATTERN, regex=True)
        n = int(has_code.sum())
        pct = 100 * n / len(addr) if len(addr) else 0
        lines.append(f"  {label:20s}: {n:,} / {len(addr):,} "
                     f"addresses ({pct:.2f}%) contain a 5-6 digit postal-like code")
    return "\n".join(lines) + "\n"


def non_ascii_scan(datasets: dict) -> str:
    """
    Rewritten to use str.contains() with a regex character class instead
    of .apply(lambda s: not s.isascii()). Same result, vectorized.
    """
    lines = ["=== Non-ASCII Character Scan ===\n"]
    for label, df in datasets.items():
        lines.append(f"-- {label} --")
        for col in [COL_NAME, COL_ADDRESS]:
            if col not in df.columns:
                continue
            text = df[col]
            has_non_ascii = text.str.contains(NON_ASCII_PATTERN, regex=True)
            n = int(has_non_ascii.sum())
            pct = 100 * n / len(text) if len(text) else 0
            lines.append(f"  {col:20s}: {n:,} / {len(text):,} "
                         f"rows ({pct:.2f}%) contain non-ASCII characters")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    # Only loads business_name/business_address (via usecols=cols) --
    # entity_id and country are irrelevant to every check in this
    # script, so skipping them reduces memory and load time across all
    # six multi-million-row source files.
    print("Loading files...")
    cols = [COL_NAME, COL_ADDRESS]
    datasets = {
        "train_source1": load_tsv(TRAIN_SOURCE1, usecols=cols),
        "train_source2": load_tsv(TRAIN_SOURCE2, usecols=cols),
        "train_source3": load_tsv(TRAIN_SOURCE3, usecols=cols),
        "test_source1": load_tsv(TEST_SOURCE1, usecols=cols),
        "test_source2": load_tsv(TEST_SOURCE2, usecols=cols),
        "test_source3": load_tsv(TEST_SOURCE3, usecols=cols),
    }

    report_parts = [length_distribution_report(datasets)]

    print("Computing business_name token frequency...")
    name_lines, name_freq_df = token_frequency(datasets, COL_NAME, top_n=50)
    report_parts.append("=== Business Name Token Frequency (Legal Suffix Inventory) ===\n"
                         + "\n".join(name_lines) + "\n")

    print("Computing business_address token frequency...")
    addr_lines, addr_freq_df = token_frequency(datasets, COL_ADDRESS, top_n=50)
    report_parts.append("=== Business Address Token Frequency (Keyword Inventory) ===\n"
                         + "\n".join(addr_lines) + "\n")

    report_parts.append(postal_code_pattern_check(datasets))
    report_parts.append(non_ascii_scan(datasets))

    full_report = "\n".join(report_parts)
    print(full_report)

    REPORT_PATH.write_text(full_report, encoding="utf-8")
    name_freq_df.to_csv(NAME_TOKEN_CSV, index=False)
    addr_freq_df.to_csv(ADDRESS_TOKEN_CSV, index=False)
    print(f"\nFull report written to: {REPORT_PATH}")
    print(f"Full name token frequency table written to: {NAME_TOKEN_CSV}")
    print(f"Full address token frequency table written to: {ADDRESS_TOKEN_CSV}")


if __name__ == "__main__":
    main()