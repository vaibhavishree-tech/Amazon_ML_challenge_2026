"""
io_utils.py

Shared loading utilities for all EDA scripts.

Rewritten for large files (millions of rows):

1. engine="pyarrow" is tried first. Pyarrow's CSV/TSV reader is
   substantially faster than pandas' default C engine at this scale,
   since it parses in parallel using compiled code. If pyarrow is not
   installed, the loader falls back to the standard engine automatically
   so the script does not hard-fail on machines without it.

2. Load timing is printed for every file, so you can see exactly which
   file is slow, instead of a script that silently sits at "Loading
   files..." with no feedback for minutes.

The sep="\t" requirement, dtype=str requirement, and
keep_default_na=False requirement from the original version are
unchanged -- those are correctness requirements, not performance
choices, and removing them would corrupt the data.
"""

import time
from pathlib import Path

import pandas as pd

def load_tsv(path: Path, usecols: list = None) -> pd.DataFrame:
    """
    Load a single tab-separated source or ground-truth file, with the
    same column-handling rules applied consistently everywhere this
    function is called across the EDA suite.

    usecols: optional list of column names to load. Restricting to only
    the columns a given script actually needs (e.g. skipping
    business_name/business_address when only counts are required) cuts
    load time and memory proportionally at multi-million-row scale.

    dtype="string[pyarrow]": stores every column as a pyarrow-backed
    string type directly at read time. This must be used instead of
    plain dtype=str -- combining dtype=str with dtype_backend="pyarrow"
    causes pandas to read with pyarrow, then cast to a numpy
    fixed-width unicode array afterward, which allocates space for the
    single longest string in the column times every row. That cast is
    what caused the 5+ GB allocation crash; specifying the pyarrow
    string dtype directly avoids the cast entirely.

    keep_default_na=False: ensures a genuinely empty cell is read as an
    empty string, not silently converted to NaN -- this matters because
    several scripts explicitly measure "missing" as str.strip() == "",
    and pandas' default NA handling would hide some of that signal
    before it could be counted.
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
        # the script usable on machines without pyarrow installed,
        # just slower.
        df = pd.read_csv(path, **read_kwargs)

    df.columns = [c.strip() for c in df.columns]
    elapsed = time.time() - start
    print(f"  loaded {path.name}: {len(df):,} rows in {elapsed:.1f}s")
    return df