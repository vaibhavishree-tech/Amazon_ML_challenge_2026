"""
tokenizers.py

Text-to-tokens functions used to build and query the blocking indexes.
Two variants of each tokenizer are provided deliberately:

  - a Series-based (vectorized where possible) version, used when
    building an index over a full multi-million-row column
  - a single-string version, used inside the per-entity query loop in
    blocking_pipeline.py, where a full Series does not exist (each
    Source 1 entity is handled one at a time)

Both variants share the same underlying regex pattern / n-gram logic,
so a token produced at index-build time will always match the same
token produced at query time -- keeping these two code paths in sync is
essential, since any divergence would silently break every lookup
against the index.
"""

import re

import pandas as pd

from config import WORD_MIN_LEN, NGRAM_N

# Word pattern with the minimum length encoded directly in the regex
# (e.g. {2,} for WORD_MIN_LEN=2), so length filtering happens as part
# of tokenization itself -- no separate per-row filtering step needed
# after the fact.
_WORD_PATTERN = re.compile(rf"[a-z0-9]{{{WORD_MIN_LEN},}}")


def tokenize_words_batch(text_series: pd.Series) -> pd.Series:
    """
    Vectorized word tokenization for a full column at once. Uses
    str.findall(), which runs in pandas' compiled C code rather than a
    Python-level loop -- the fast path used when building indexes over
    millions of rows at once.
    """
    return text_series.str.findall(_WORD_PATTERN.pattern)


def tokenize_words_single(text: str) -> list:
    """Single-string equivalent of tokenize_words_batch, used in the per-entity query loop."""
    return _WORD_PATTERN.findall(text)


def generate_char_ngrams(text: str, n: int = NGRAM_N) -> list:
    """
    Character n-grams for one string. This is a plain Python loop over
    the string's characters -- there is no vectorized pandas primitive
    for n-gram extraction, so this per-string cost is unavoidable
    wherever it runs. It remains fast per individual string given
    typical name/address lengths in this dataset, but at millions of
    rows this step must be BATCHED (see inverted_index.py) to bound
    peak memory, the same discipline used for unidecode batching in the
    cleaning phase.
    """
    if len(text) < n:
        return [text] if text else []
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def tokenize_ngrams_batch(text_series: pd.Series, n: int = NGRAM_N) -> pd.Series:
    """
    Series-based n-gram tokenization. Uses .map() rather than a
    vectorized string method, since pandas has no built-in vectorized
    n-gram primitive -- this is inherently a per-row Python operation,
    the same category of cost as the unidecode step in cleaning.
    Always call this through the batching loop in inverted_index.py,
    never directly on a full multi-million-row column.
    """
    return text_series.map(lambda s: generate_char_ngrams(s, n))