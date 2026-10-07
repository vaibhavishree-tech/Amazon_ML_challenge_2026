"""
text_similarity_features.py

The single shared feature-computation function for the matching model.
Imported UNCHANGED by both the training feature-computation script and
the final test-inference script -- this is deliberate: if training and
inference ever computed features even slightly differently, the model
would be scoring data it was never actually trained to understand.

Feature rationale:
  - Levenshtein / token Jaccard / char n-gram Jaccard (name and
    address): these lexical similarity metrics were validated during
    the EDA and cleaning phases, where positive (true-match) and
    negative (non-match) pair distributions were measured directly and
    showed clear separation on all three metrics.
  - postal_code_exact_match: uses the postal/PIN code extracted during
    cleaning as a standalone feature -- sparse in the data overall, but
    a strong, high-confidence signal when present on both sides of a
    pair.
  - matched_side_was_non_ascii: flags whether the candidate's original
    (pre-cleaning) business name contained non-ASCII characters. This
    was found during blocking validation to correlate with a real
    recall/similarity gap (names originally written in non-Latin
    scripts score lower on lexical similarity even after cleaning), so
    it is given to the model directly as a feature rather than left for
    the model to infer indirectly from noisy text.

FEATURE_COLUMNS defines the fixed output column order -- both the
training script and the test-inference script write/read features in
exactly this order, eliminating any risk of a column-mismatch between
what the model was trained on and what it scores at inference time.
"""

import re

from rapidfuzz.distance import Levenshtein

NON_ASCII_PATTERN = re.compile(r"[^\x20-\x7E]")

FEATURE_COLUMNS = [
    "name_levenshtein",
    "name_token_jaccard",
    "name_char3gram_jaccard",
    "address_levenshtein",
    "address_token_jaccard",
    "address_char3gram_jaccard",
    "postal_code_exact_match",
    "name_length_diff",
    "address_length_diff",
    "matched_side_was_non_ascii",
]


def char_ngrams(text: str, n: int = 3) -> set:
    """Character n-gram set for one string (default trigrams); returns the whole string as a single "n-gram" if it is shorter than n."""
    text = text.strip()
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def token_set(text: str) -> set:
    """Lowercase alphanumeric token (word) set for one string."""
    return set(re.findall(r"[a-z0-9]+", text))


def jaccard(set_a: set, set_b: set) -> float:
    """Standard Jaccard similarity; two empty sets are treated as a perfect match (1.0) rather than undefined."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity (0 = completely different, 1 = identical), via rapidfuzz's C-accelerated implementation."""
    return Levenshtein.normalized_similarity(a, b)


def compute_pair_features(
    s1_name: str, s1_addr: str, s1_postal: str,
    other_name: str, other_addr: str, other_postal: str,
    other_name_original: str,
) -> list:
    """
    Computes all features for ONE (Source 1, candidate) pair. Returns a
    plain list, in the exact order of FEATURE_COLUMNS, so it can be
    written straight to a tab-separated line without an intermediate
    dict -- this function runs once per candidate pair, potentially tens
    of millions of times across training and inference, so avoiding
    dict overhead here matters at that scale.

    other_name_original: the candidate's business_name from the RAW
    (uncleaned) source file, used ONLY for the non-ASCII flag -- every
    other argument should be the CLEANED text, since those are the
    values both the model and the actual matching task operate on.
    """
    name_lev = levenshtein_ratio(s1_name, other_name)
    name_tok_jac = jaccard(token_set(s1_name), token_set(other_name))
    name_char_jac = jaccard(char_ngrams(s1_name), char_ngrams(other_name))

    addr_lev = levenshtein_ratio(s1_addr, other_addr)
    addr_tok_jac = jaccard(token_set(s1_addr), token_set(other_addr))
    addr_char_jac = jaccard(char_ngrams(s1_addr), char_ngrams(other_addr))

    postal_match = 1 if (s1_postal and other_postal and s1_postal == other_postal) else 0

    name_len_diff = abs(len(s1_name) - len(other_name))
    addr_len_diff = abs(len(s1_addr) - len(other_addr))

    is_non_ascii = 1 if NON_ASCII_PATTERN.search(other_name_original) else 0

    return [
        round(name_lev, 4), round(name_tok_jac, 4), round(name_char_jac, 4),
        round(addr_lev, 4), round(addr_tok_jac, 4), round(addr_char_jac, 4),
        postal_match, name_len_diff, addr_len_diff, is_non_ascii,
    ]