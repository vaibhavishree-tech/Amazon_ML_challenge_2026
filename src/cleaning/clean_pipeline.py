"""
clean_pipeline.py

Core cleaning functions. Every function takes and returns a pandas
Series, and is vectorized (pandas .str methods, which run in C, not
Python-level loops) except for the unidecode transliteration step,
which is unavoidably a per-string Python function call -- pandas has
no vectorized equivalent. That single step is batched and progress-
printed for exactly the same memory reasons as the token-frequency
batching used in eda/04_text_vocab_encoding.py.

Processing order matters and is NOT arbitrary:

  1. Placeholder-token neutralization -- must run first, before
     anything else treats placeholder text as real content.
  2. Curly quote normalization -- raw text, before any other
     transformation touches punctuation.
  3. NFKC unicode normalization -- collapses composed/decomposed
     character variants into one consistent form before anything
     else inspects individual characters.
  4. Native-script state substitution (address only) -- must run
     BEFORE unidecode, since unidecode's phonetic transliteration of
     Devanagari/Kannada/Gurmukhi/Bengali state names is unreliable for
     proper nouns; a direct dictionary hit is safer than trusting
     phonetic reconstruction.
  5. Unidecode transliteration -- folds all remaining non-ASCII
     (French accents, business names in native script, anything the
     state map didn't catch) into ASCII.
  6. Lowercase + whitespace normalization.
  7. Period-stripping after letters -- turns "L.L.C." into "LLC" and
     "Inc." into "Inc" BEFORE suffix mapping, or the suffix dictionary
     (keyed on clean tokens like "llc") would never match the
     punctuated forms.
  8. Legal suffix / street-type / state mapping -- now safe to run
     since punctuation and casing are already normalized.
  9. Domain/hashtag/honorific stripping (name only).
 10. Postal code extraction (address only) -- last, since it reads
     digit sequences that earlier steps do not alter.
"""

import gc
import re

import pandas as pd
from unidecode import unidecode

from text_cleaning_rules import (
    PLACEHOLDER_TOKENS,
    CURLY_DOUBLE_QUOTES,
    CURLY_SINGLE_QUOTES,
    LEGAL_SUFFIX_MAP,
    STREET_TYPE_MAP,
    US_STATE_MAP,
    INDIA_STATE_MAP,
    NATIVE_SCRIPT_STATE_MAP,
    LEADING_HONORIFICS_PATTERN,
    URL_OR_DOMAIN_PATTERN,
    POSTAL_CODE_PATTERN,
)


def neutralize_placeholder_tokens(series: pd.Series) -> pd.Series:
    """
    Replaces a cell with an empty string if, after stripping and
    lowercasing, its ENTIRE content matches a known placeholder token
    (e.g. "NA", "null", "<NULL>"). This is a whole-cell match, not a
    substring match, so it will not touch legitimate text that merely
    contains "na" as part of a longer word.
    """
    stripped_lower = series.str.strip().str.lower()
    is_placeholder = stripped_lower.isin(PLACEHOLDER_TOKENS)
    return series.mask(is_placeholder, "")


def normalize_curly_quotes(series: pd.Series) -> pd.Series:
    """
    Curly double quotes become a space (they carry no content).
    Curly single quotes become a straight apostrophe, preserving
    possessive names like "Denny's" instead of breaking them into
    "Denny s".
    """
    double_pattern = f"[{CURLY_DOUBLE_QUOTES}]"
    single_pattern = f"[{CURLY_SINGLE_QUOTES}]"
    series = series.str.replace(double_pattern, " ", regex=True)
    series = series.str.replace(single_pattern, "'", regex=True)
    return series


def nfkc_normalize(series: pd.Series) -> pd.Series:
    """
    Applies Unicode NFKC normalization, which collapses characters that
    can be represented multiple ways (e.g. a composed accented
    character vs. a base letter plus a separate combining-accent mark)
    into one canonical form. Without this, two visually identical
    strings can fail an exact-match or even a naive similarity check
    because their underlying byte sequences differ.

    This is a per-string operation (Python's unicodedata module has no
    vectorized pandas equivalent), so it uses .map(), same as
    unidecode -- but NFKC normalization is much faster per string than
    unidecode's lookup-table transliteration, so this does not need the
    same batching treatment.
    """
    import unicodedata
    return series.map(lambda s: unicodedata.normalize("NFKC", s))


def substitute_native_script_states(series: pd.Series) -> pd.Series:
    """
    Address-only. Direct substring replacement of known native-script
    state names with their English equivalent, run before unidecode.
    Vectorized via a single alternation regex built from the
    dictionary keys, rather than looping the dictionary in Python.
    """
    if not NATIVE_SCRIPT_STATE_MAP:
        return series
    # Builds one big "keyA|keyB|keyC|..." regex from every dictionary
    # key (escaped, since native-script characters can include regex
    # metacharacters) so the whole map is applied in a single pass per
    # string instead of one .replace() call per dictionary entry.
    pattern = "|".join(re.escape(k) for k in NATIVE_SCRIPT_STATE_MAP.keys())
    compiled = re.compile(pattern)

    def replace_all(text: str) -> str:
        return compiled.sub(lambda m: NATIVE_SCRIPT_STATE_MAP[m.group(0)], text)

    return series.map(replace_all)


def transliterate_unidecode(series: pd.Series, batch_size: int, label: str) -> pd.Series:
    """
    Folds remaining non-ASCII characters to their closest ASCII
    equivalent (accented French characters, and any native-script text
    the state-name substitution did not catch, such as business names
    themselves written in Telugu/Hindi/Malayalam/etc, as seen directly
    in the manual review).

    Batched deliberately: unidecode() is a pure Python function called
    once per string, with no vectorized equivalent, so at 5 million
    rows this is unavoidably the slowest step in the whole pipeline.
    Batching keeps peak memory bounded and prints progress so you can
    see it is working rather than appearing to hang, exactly like the
    tokenization batching used in the EDA phase.
    """
    n_rows = len(series)
    n_batches = (n_rows // batch_size) + 1
    print(f"    unidecode on {label}: {n_rows:,} rows in {n_batches} batch(es)")

    results = []
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        batch = series.iloc[start:end]
        cleaned_batch = batch.map(unidecode)
        results.append(cleaned_batch)
        # Drop references to this batch's data and force garbage
        # collection before starting the next batch, so peak memory
        # stays bounded to roughly one batch's worth rather than
        # accumulating across the whole column.
        del batch, cleaned_batch
        gc.collect()

    combined = pd.concat(results, ignore_index=False)
    del results
    gc.collect()
    return combined


def standardize_case_and_whitespace(series: pd.Series) -> pd.Series:
    """Lowercase, strip leading/trailing whitespace, collapse internal double spaces."""
    series = series.str.lower().str.strip()
    series = series.str.replace(r"\s+", " ", regex=True)
    return series


def strip_periods_after_letters(series: pd.Series) -> pd.Series:
    """
    Removes any period immediately following a letter: "l.l.c." becomes
    "llc", "inc." becomes "inc", "p.c." becomes "pc". This targets the
    single-letter-token fragmentation directly visible in the EDA's top
    name-token list (l: 799,570; s: 778,726; c: 570,277 -- all far too
    frequent to be real one-letter business-name words, and consistent
    with the punctuated suffix forms seen in manual review rows 21, 46,
    66, 80). Must run before legal suffix mapping, or "l.l.c." never
    matches the "llc" key in LEGAL_SUFFIX_MAP.
    """
    return series.str.replace(r"(?<=[A-Za-z])\.", "", regex=True)


def _apply_word_map(series: pd.Series, mapping: dict) -> pd.Series:
    """
    Shared helper: replaces whole-word tokens matching a dictionary key
    with their mapped value, using one compiled alternation regex with
    word boundaries rather than looping the dictionary per row. Used
    for legal suffixes, street types, and state abbreviations alike.
    """
    if not mapping:
        return series
    # \b word boundaries ensure only whole tokens match a key -- e.g.
    # the "co" -> "company" entry in LEGAL_SUFFIX_MAP must not match
    # the "co" inside "coconut" or "corp".
    pattern = r"\b(" + "|".join(re.escape(k) for k in mapping.keys()) + r")\b"
    compiled = re.compile(pattern)

    def replace_match(m):
        return mapping[m.group(0)]

    return series.str.replace(compiled, replace_match, regex=True)


def map_legal_suffixes(series: pd.Series) -> pd.Series:
    """Business-name only. Expands abbreviated legal suffixes to a standard form."""
    return _apply_word_map(series, LEGAL_SUFFIX_MAP)


def map_street_types_and_states(series: pd.Series) -> pd.Series:
    """
    Business-address only. Expands street-type abbreviations and both
    US and Indian state abbreviations to their full names in one pass.
    """
    series = _apply_word_map(series, STREET_TYPE_MAP)
    series = _apply_word_map(series, US_STATE_MAP)
    series = _apply_word_map(series, INDIA_STATE_MAP)
    return series


def strip_name_artifacts(series: pd.Series) -> pd.Series:
    """
    Business-name only. Removes domain/URL/hashtag-style artifacts and
    leading honorific prefixes, then collapses any resulting double
    spaces.
    """
    series = series.str.replace(URL_OR_DOMAIN_PATTERN, " ", regex=True)
    series = series.str.replace(LEADING_HONORIFICS_PATTERN, "", regex=True)
    series = series.str.replace(r"\s+", " ", regex=True).str.strip()
    return series


def extract_postal_code(series: pd.Series) -> pd.Series:
    """
    Business-address only. Extracts the first 5-6 digit standalone
    number found in the address into a separate feature column, per
    the locked plan. The address text itself is left unchanged --
    extraction is additive, not destructive, since the plan calls for
    this as a separate scoring feature, not a text-cleaning transform.
    """
    extracted = series.str.extract(POSTAL_CODE_PATTERN, expand=False)
    return extracted.fillna("")


def clean_business_name(series: pd.Series, batch_size: int) -> pd.Series:
    """Full ordered pipeline for the business_name column. See module docstring for why this order."""
    series = neutralize_placeholder_tokens(series)
    series = normalize_curly_quotes(series)
    series = nfkc_normalize(series)
    series = transliterate_unidecode(series, batch_size, label="business_name")
    series = standardize_case_and_whitespace(series)
    series = strip_periods_after_letters(series)
    series = map_legal_suffixes(series)
    series = strip_name_artifacts(series)
    series = standardize_case_and_whitespace(series)  # final collapse after artifact removal
    return series


def clean_business_address(series: pd.Series, batch_size: int) -> tuple:
    """
    Full ordered pipeline for the business_address column. Returns
    (cleaned_address_series, postal_code_series) since address cleaning
    also produces the new postal-code feature column.
    """
    series = neutralize_placeholder_tokens(series)
    series = normalize_curly_quotes(series)
    series = nfkc_normalize(series)
    series = substitute_native_script_states(series)
    series = transliterate_unidecode(series, batch_size, label="business_address")
    series = standardize_case_and_whitespace(series)
    series = strip_periods_after_letters(series)
    series = map_street_types_and_states(series)
    series = standardize_case_and_whitespace(series)  # final collapse after mapping

    # Extraction happens on the already-cleaned address text, so the
    # extracted digits reflect the normalized format rather than any
    # raw-text noise (stray punctuation, etc.) that earlier steps removed.
    postal_codes = extract_postal_code(series)

    return series, postal_codes