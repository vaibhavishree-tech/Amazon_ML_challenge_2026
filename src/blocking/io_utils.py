"""
io_utils.py

Load/save utilities for the blocking phase. Reuses the same
dtype="string[pyarrow]" approach established in the cleaning phase,
rather than dtype=str combined with dtype_backend="pyarrow" -- the
latter combination causes pandas to cast to a numpy fixed-width Unicode
array sized to the single longest string in the column, times every
row, which is both slow and memory-intensive at multi-million-row
scale.
"""

import time
from pathlib import Path

import pandas as pd


def load_tsv(path: Path, usecols: list = None) -> pd.DataFrame:
    """
    Load a tab-separated source file for the blocking phase.

    dtype="string[pyarrow]": stores text in a compact Arrow-backed
    string array rather than plain dtype=str, avoiding the numpy
    object-array overhead that comes with storing millions of
    individual Python string objects.

    usecols: restricts the load to only the columns a given script
    actually needs, reducing memory and load time proportionally --
    important here since blocking repeatedly reloads the same
    multi-million-row cleaned source files across several scripts.
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
        # Falls back to pandas' default engine if pyarrow is unavailable
        # or rejects an option it doesn't support, so the script still
        # runs (just slower) on machines without pyarrow installed.
        df = pd.read_csv(path, **read_kwargs)

    df.columns = [c.strip() for c in df.columns]
    elapsed = time.time() - start
    print(f"  loaded {path.name}: {len(df):,} rows in {elapsed:.1f}s")
    return df