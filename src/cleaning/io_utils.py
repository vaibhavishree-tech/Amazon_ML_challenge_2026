"""
io_utils.py

Shared load/save utilities for the cleaning phase.

This reuses the corrected loading approach from the EDA phase --
dtype="string[pyarrow]" rather than dtype=str combined with
dtype_backend="pyarrow", since that combination was what caused the
5+ GB fixed-width-unicode memory crash during EDA (mixing dtype=str
with dtype_backend made pandas cast to a numpy fixed-width array sized
to the single longest string in the column, times every row).
"""

import time
from pathlib import Path

import pandas as pd


def load_tsv(path: Path, usecols: list = None) -> pd.DataFrame:
    """
    Load a tab-separated source file for cleaning.

    dtype="string[pyarrow]": stores text in a compact Arrow-backed
    string array, avoiding both the numpy object-array overhead of
    plain dtype=str and the fixed-width-cast crash from mixing dtype=str
    with dtype_backend="pyarrow".

    keep_default_na=False: a true missing field in the TSV becomes an
    empty string "", not NaN. This matters directly for the cleaning
    pipeline -- it means every "missing" address is already a plain
    empty string by the time cleaning functions see it, so there is no
    separate NaN-handling path needed alongside the placeholder-token
    handling in clean_pipeline.py.

    usecols: restrict to only the columns actually needed. Cleaning
    needs all four columns (entity_id, business_name, business_address,
    country) since the output must be a complete cleaned file, but this
    parameter is kept for scripts (like validation) that only need a
    subset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected input file not found: {path}")

    start = time.time()
    read_kwargs = dict(
        sep="\t",
        dtype="string[pyarrow]",
        keep_default_na=False,
        encoding="utf-8",
    )
    if usecols is not None:
        read_kwargs["usecols"] = usecols

    try:
        df = pd.read_csv(path, engine="pyarrow", **read_kwargs)
    except (ImportError, ValueError):
        # Falls back to pandas' default engine if pyarrow is missing, or
        # if pyarrow rejects a read option it doesn't support -- keeps
        # this usable on machines without pyarrow installed, just slower.
        df = pd.read_csv(path, **read_kwargs)

    df.columns = [c.strip() for c in df.columns]
    elapsed = time.time() - start
    print(f"  loaded {path.name}: {len(df):,} rows in {elapsed:.1f}s")
    return df


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    """
    Write a cleaned dataframe back out as tab-separated, matching the
    input format exactly (tabs, not commas, since business names and
    addresses may still contain commas even after cleaning).
    """
    start = time.time()
    df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    elapsed = time.time() - start
    print(f"  wrote {path.name}: {len(df):,} rows in {elapsed:.1f}s")