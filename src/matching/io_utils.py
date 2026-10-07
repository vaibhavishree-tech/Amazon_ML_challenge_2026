"""
io_utils.py

Shared load utility for the matching phase. Uses
dtype="string[pyarrow]" consistently with the earlier EDA, cleaning,
and blocking phases -- not dtype=str combined with
dtype_backend="pyarrow", since that combination forces a cast to a
numpy fixed-width Unicode array sized to the single longest string in
the column, times every row, which is both slow and memory-intensive
at this project's multi-million-row scale.
"""

import time
from pathlib import Path

import pandas as pd


def load_tsv(path: Path, usecols: list = None) -> pd.DataFrame:
    """
    Loads a tab-separated file for the matching phase.

    usecols: restricts the load to only the columns a given script
    actually needs, reducing memory and load time proportionally when
    working with the large cleaned source files.
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
        # or rejects an unsupported option, so loading still succeeds
        # (just more slowly) on machines without pyarrow installed.
        df = pd.read_csv(path, **read_kwargs)

    df.columns = [c.strip() for c in df.columns]
    elapsed = time.time() - start
    print(f"  loaded {path.name}: {len(df):,} rows in {elapsed:.1f}s")
    return df