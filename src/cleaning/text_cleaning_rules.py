"""
text_cleaning_rules.py

All lookup dictionaries and regex patterns used by the cleaning
pipeline, kept in one file so every rule's evidence source is visible
in one place rather than scattered across processing functions.

A note on the competition's "no external data lookup" rule: the
dictionaries below (legal suffix expansions, US/Indian state names,
street-type abbreviations) are general linguistic and geographic
knowledge, not business-identity data -- they do not look up any
specific business's registration, identity, or records. This is
different in kind from querying a business registry or geocoding API.
"""

import re

# ---------------------------------------------------------------------
# 1. Placeholder tokens that appear as literal cell content rather than
#    a true empty string.
#
#    Evidence: you personally found 13 rows in train_source3's
#    business_name where Data Wrangler flagged missing values that our
#    EDA script counted as present, because they were literal text like
#    "NA" rather than a blank cell. The manual review sample also shows
#    literal "<NULL>" (row 24) and "null" (rows 73, 81) inside
#    business_address text.
#
#    Matching is done as a WHOLE-CELL match after stripping and
#    lowercasing, not a substring match -- this is important, since a
#    substring match would incorrectly strip "na" out of legitimate
#    text like "National" or "Panama".
# ---------------------------------------------------------------------
# Used by neutralize_placeholder_tokens() in clean_pipeline.py -- the
# set membership check there is a whole-cell match against this set,
# not a substring search.
PLACEHOLDER_TOKENS = {
    "na", "n/a", "n.a", "n.a.", "none", "null", "<null>", "nan",
    "-", "--", "not available", "not applicable",
}

# ---------------------------------------------------------------------
# 2. Curly quote handling.
#
#    Evidence: 01_volume_integrity_report.txt shows curly quotes only
#    in the TEST set addresses (2,091 in test_source1, 5,381 in
#    test_source2, 5,508 in test_source3 -- absent from all training
#    files). Double and single curly quotes need different treatment:
#    double quotes carry no semantic content and become a space; single
#    curly quotes are almost always a stylized apostrophe and should
#    become a straight apostrophe, not a space -- collapsing an
#    apostrophe to a space would break possessive names like "Denny's"
#    into "Denny s", which changes the tokenization of a real business
#    name for no reason.
# ---------------------------------------------------------------------
CURLY_DOUBLE_QUOTES = "\u201c\u201d"   # " "
CURLY_SINGLE_QUOTES = "\u2018\u2019"   # ' '

# ---------------------------------------------------------------------
# 3. Legal suffix standardization.
#
#    Evidence: 04_text_vocab_encoding_report.txt top-50 name tokens
#    show pvt (1,117,065), ltd (2,007,217), inc (1,696,402),
#    corp (570,620), llp (403,859), lp (175,194), pc (164,862),
#    co (597,205) as extremely high-frequency separate tokens from
#    their expanded forms (private: 3,345,585, limited: 3,647,872).
#    Keys here are matched AFTER lowercasing and AFTER period-stripping
#    (see clean_pipeline.py), so "Pvt", "PVT.", "Pvt." all reduce to
#    the same "pvt" key before this mapping runs.
#
#    sarl and sas are deliberately NOT included here -- the EDA
#    confirms these are high-frequency (358,680 and 254,297
#    occurrences) and correspond to French entity types, tied directly
#    to France's 157,598 occurrences in the same token list. Since test
#    set France volume (per 03_country_analysis_report.txt) is
#    substantial (259,452 / 703,378 / 731,615 rows across the three
#    sources), collapsing sarl/sas into an English equivalent would
#    destroy a real, discriminative signal for an entire country's
#    worth of test data.
# ---------------------------------------------------------------------
# Applied to business_name only, via map_legal_suffixes() ->
# _apply_word_map() in clean_pipeline.py.
LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "llp": "limited liability partnership",
    "lp": "limited partnership",
    "pc": "professional corporation",
    "co": "company",
}

# ---------------------------------------------------------------------
# 4. Street-type abbreviation standardization.
#
#    Evidence: 04_text_vocab_encoding_report.txt address token list
#    shows both forms of several street types at high frequency side by
#    side: road (3,330,789) vs rd (1,138,922), street (1,674,567) vs
#    st (1,149,237), avenue (1,084,277) vs ave (753,896), drive
#    (1,075,980) vs dr (944,259), lane (588,170) appears without a
#    common short form in the top 50 but is included for consistency.
# ---------------------------------------------------------------------
# Applied to business_address only, via map_street_types_and_states()
# in clean_pipeline.py.
STREET_TYPE_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "dr": "drive",
    "ln": "lane",
    "blvd": "boulevard",
    "ct": "court",
    "pl": "place",
}

# ---------------------------------------------------------------------
# 5. US state abbreviation -> full name.
#
#    Evidence: address token list shows state abbreviations at high
#    frequency (mh: 700,334, tx: 696,984, ny: 535,855, nc: 498,815,
#    tn: 540,277) mixed with spelled-out forms across sources -- exactly
#    the pattern your own manual review flagged repeatedly (e.g. row 33:
#    Maharashtra vs MH; row 70: state spelled out only on one side).
# ---------------------------------------------------------------------
US_STATE_MAP = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut",
    "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii",
    "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine",
    "md": "maryland", "ma": "massachusetts", "mi": "michigan",
    "mn": "minnesota", "ms": "mississippi", "mo": "missouri",
    "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio",
    "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming",
}

# ---------------------------------------------------------------------
# 6. Indian state/UT abbreviation -> full name, and known regional
#    English-spelling variants.
#
#    Evidence: manual review shows MH (Maharashtra), TG (Telangana),
#    GJ (Gujarat), TN (Tamil Nadu), DL (Delhi) as abbreviation forms,
#    plus "Keralam" (row 82) and "Odisha" vs "Orissa" (row 41,
#    official renaming rather than noise) as spelling variants of the
#    same state appearing on different sides of a true match.
# ---------------------------------------------------------------------
# Note: "keralam"/"orissa" map to the SAME target value a corresponding
# abbreviation key would (kl -> kerala is not listed above but keralam
# is a direct spelling-variant entry here) -- these two entries handle
# spelling variants, not abbreviations, which is why they coexist with
# the ap/tg/etc. abbreviation entries above them in the same dict.
INDIA_STATE_MAP = {
    "mh": "maharashtra",
    "tg": "telangana",
    "ap": "andhra pradesh",
    "gj": "gujarat",
    "tn": "tamil nadu",
    "dl": "delhi",
    "up": "uttar pradesh",
    "wb": "west bengal",
    "ka": "karnataka",
    "kl": "kerala",
    "rj": "rajasthan",
    "pb": "punjab",
    "hr": "haryana",
    "keralam": "kerala",
    "orissa": "odisha",
}

# ---------------------------------------------------------------------
# 7. Native-script state name -> English, applied BEFORE unidecode.
#
#    Evidence: manual review shows state names appearing directly in
#    native script rather than transliterated Latin text: Devanagari
#    for Uttar Pradesh (row 1), Maharashtra (rows 14, 38, 61, 62), and
#    Delhi (rows 71, 99); Kannada for Karnataka (row 15); Gurmukhi for
#    Punjab (row 58); Bengali for West Bengal (row 96).
#
#    This runs BEFORE the general unidecode step deliberately, because
#    unidecode's phonetic transliteration of these scripts is often
#    imprecise for proper nouns -- a direct dictionary substitution on
#    the exact known state-name string is more reliable than trusting
#    unidecode to phonetically reconstruct "maharashtra" from
#    "महाराष्ट्र" on its own.
#
#    This dictionary is NOT exhaustive -- it covers the scripts and
#    states actually observed in your 100-row manual review, plus a
#    small number of additional common states added by extension.
#    Extend it if 02_validate_cleaning.py's post-cleaning non-ASCII
#    scan shows a state-name-shaped token still surviving.
# ---------------------------------------------------------------------
# Keys are the exact native-script strings; applied via a compiled
# alternation regex in substitute_native_script_states() so all entries
# are checked in a single vectorized pass rather than one dict lookup
# per state per row.
NATIVE_SCRIPT_STATE_MAP = {
    "महाराष्ट्र": "maharashtra",
    "उत्तर प्रदेश": "uttar pradesh",
    "दिल्ली": "delhi",
    "गुजरात": "gujarat",
    "राजस्थान": "rajasthan",
    "ಕರ್ನಾಟಕ": "karnataka",
    "ਪੰਜਾਬ": "punjab",
    "পশ্চিমবঙ্গ": "west bengal",
    "తెలంగాణ": "telangana",
    "ఆంధ్రప్రదేశ్": "andhra pradesh",
    "കേരളം": "kerala",
    "તેલંગાણા": "telangana",
}

# ---------------------------------------------------------------------
# 8. Honorific / entity-prefix tokens to strip from the START of a
#    business name only (not mid-string, to avoid stripping a real word
#    that happens to match).
#
#    Evidence: manual review rows 18 (Sri), 19 (Smt), 29 (Mr), 99 (M/s)
#    all show an honorific prefix present on one side of a true match
#    and absent on the other.
# ---------------------------------------------------------------------
# Anchored to the START of the string (^) with a word boundary, so this
# only strips a genuine leading honorific, never a mid-name occurrence.
LEADING_HONORIFICS_PATTERN = re.compile(
    r"^(sri|smt|mr|mrs|m/s)\b\s*", flags=re.IGNORECASE
)

# ---------------------------------------------------------------------
# 9. Domain / social-media-style name artifacts.
#
#    Evidence: manual review rows 43 (heliospetroleum.com), 59
#    (#ghartrading), 63 (shyamsystems.com), 79 (metoyerurban.com), 96
#    (vasupujyaenterprises.com), 98 (LLC Insight | www.hernandez.com)
#    all show a domain-name-style or hashtag-style string used as the
#    business name on one side instead of a normal name.
# ---------------------------------------------------------------------
# Matches either a URL/domain-like token (optionally preceded by a pipe
# separator, as in row 98's "LLC Insight | www.hernandez.com") OR a
# leading hashtag -- the two noise patterns actually observed in the
# manual review, combined into one pattern via alternation.
URL_OR_DOMAIN_PATTERN = re.compile(
    r"(\|?\s*(www\.)?\S+\.(com|net|org|in|co)\b)|(^#\S+)",
    flags=re.IGNORECASE,
)

# ---------------------------------------------------------------------
# 10. Postal / PIN code extraction pattern.
#
#     Evidence: 04_text_vocab_encoding_report.txt shows 5-6 digit
#     postal-like codes present in only 4.37% to 7.33% of addresses
#     depending on source/split -- too sparse to block on, but a
#     useful standalone feature when present, per the locked plan.
# ---------------------------------------------------------------------
# Captures the digit run (not just matches it), since
# extract_postal_code() in clean_pipeline.py uses str.extract() to pull
# out the captured group as a new feature column.
POSTAL_CODE_PATTERN = re.compile(r"\b(\d{5,6})\b")