"""
08_sanity_check_embeddings.py

GO/NO-GO sanity check performed before building anything further on top
of the embedding model. If this doesn't show real separation between
true matches and non-matches, the embedding-feature approach needs
rethinking before more time is spent on it.

Test pairs are NOT synthetic -- they are drawn directly from data
already seen and manually reviewed earlier in this project:

  - Devanagari/Tamil true-match pairs: from the blocking phase's
    transliteration validation sample, which already showed these pairs
    fail badly on token-based lexical similarity after unidecode (e.g.
    a heavily garbled transliteration bearing little resemblance to the
    English name). This script tests whether a semantic embedding,
    given the ORIGINAL native-script text, does better.

  - English typo/garbled true-match pairs: from the EDA phase's manual
    review sample, covering character-level corruption (e.g. "Sthoen"
    for "Stone", "Certhificatidons" for "Certifications") that
    token-overlap matching also struggles with.

  - Negative pairs: constructed by mismatching names/addresses across
    the positive examples above, so the model is tested on pairs it has
    no reason to consider similar.

PASS criterion: mean cosine similarity for true-match pairs should be
meaningfully higher than for negative pairs. No fixed universal
threshold is claimed here -- what matters is clear SEPARATION between
the two groups, consistent with how the existing lexical features were
validated earlier in this project.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from embedding_utils import load_embedding_model, build_tagged_text, encode_texts, cosine_sim


# --- True-match pairs: Devanagari/Tamil script, from the blocking
# phase's transliteration validation sample. Addresses were not
# captured in that sample (name-only comparison), so address is left
# empty here -- the tag format still works with an empty field. ---
DEVANAGARI_POSITIVE_PAIRS = [
    {
        "note": "All Energy Private Limited (Hindi script match)",
        "s1_name": "All Energy Private Limited", "s1_addr": "", "s1_country": "India",
        "other_name": "ऑल एनर्जी प्राइवेट लिमिटेड", "other_addr": "", "other_country": "India",
    },
    {
        "note": "Guru Trading Private Limited (mixed Latin+Hindi)",
        "s1_name": "Guru Trading Private Limited", "s1_addr": "", "s1_country": "India",
        "other_name": "Guru Trading प्राइवेट लिमिटेड", "other_addr": "", "other_country": "India",
    },
    {
        "note": "City Management Private Limited (Tamil script match)",
        "s1_name": "City Management Private Limited", "s1_addr": "", "s1_country": "India",
        "other_name": "சிட்டி மேனேஜ்மெண்ட் பிரைவேட் லிமிடெட்", "other_addr": "", "other_country": "India",
    },
    {
        "note": "Sky Infotech Private Limited (Gujarati script match)",
        "s1_name": "Sky Infotech Private Limited", "s1_addr": "", "s1_country": "India",
        "other_name": "સ્કાય ઇન્ફોટેક પ્રાઇવેટ લિમિટેડ", "other_addr": "", "other_country": "India",
    },
]

# --- True-match pairs: English typo/garbled text, from the EDA
# phase's manual review sample. ---
ENGLISH_TYPO_POSITIVE_PAIRS = [
    {
        "note": "Bay Creative Stone LLC (character-level typo: Stone -> Sthoen)",
        "s1_name": "Bay Creative Stone LLC",
        "s1_addr": "4018 Annapolis Road, Unit APARTMENT B, Halethorpe, MD", "s1_country": "US",
        "other_name": "Bay Creative Sthoen LLC",
        "other_addr": "4018- Annapolis Rd, HALETHORPE, MD", "other_country": "US",
    },
    {
        "note": "Construction Vr Certifications (heavy character corruption)",
        "s1_name": "Construction Vr Certifications Pvt Ltd",
        "s1_addr": "H.No1-8-67/P Apiic Kamalanagar, Kushaiguda, Hyderabad, Rangareddy, Telangana",
        "s1_country": "India",
        "other_name": "C0nstruction Vr Certhificatidons Pvt Ltd",
        "other_addr": "#1-8-67/P APIIC KAMALANAGAR, KUSHAIGUDA, RANGAREDDY, Telangana",
        "other_country": "India",
    },
]

ALL_POSITIVE_PAIRS = DEVANAGARI_POSITIVE_PAIRS + ENGLISH_TYPO_POSITIVE_PAIRS


def build_negative_pairs(positive_pairs: list) -> list:
    """
    Constructs negative (non-matching) pairs by pairing each entity's
    S1 side with a DIFFERENT entity's "other" side -- real businesses
    that should NOT be considered similar. Rotates by one position so
    every pair is a genuine mismatch, not a pair matched with itself.
    """
    n = len(positive_pairs)
    negatives = []
    for i in range(n):
        s1 = positive_pairs[i]
        other = positive_pairs[(i + 1) % n]
        negatives.append({
            "note": f"MISMATCH: {s1['note']} vs {other['note']}",
            "s1_name": s1["s1_name"], "s1_addr": s1["s1_addr"], "s1_country": s1["s1_country"],
            "other_name": other["other_name"], "other_addr": other["other_addr"],
            "other_country": other["other_country"],
        })
    return negatives


def main():
    model = load_embedding_model()

    negative_pairs = build_negative_pairs(ALL_POSITIVE_PAIRS)

    all_pairs = [(p, True) for p in ALL_POSITIVE_PAIRS] + [(p, False) for p in negative_pairs]

    # Encode everything in ONE batched call -- far more efficient than
    # encoding each pair's two strings separately, and establishes the
    # batching pattern used at much larger scale in the full
    # embedding-build scripts.
    s1_texts = [build_tagged_text(p["s1_name"], p["s1_addr"], p["s1_country"]) for p, _ in all_pairs]
    other_texts = [build_tagged_text(p["other_name"], p["other_addr"], p["other_country"]) for p, _ in all_pairs]

    print(f"\nEncoding {len(s1_texts)} pairs ({len(s1_texts) * 2} total texts)...")
    s1_embeddings = encode_texts(model, s1_texts)
    other_embeddings = encode_texts(model, other_texts)

    positive_sims = []
    negative_sims = []

    print("\n=== Pairwise Cosine Similarity ===\n")
    for i, (pair, is_positive) in enumerate(all_pairs):
        sim = cosine_sim(s1_embeddings[i], other_embeddings[i])
        label = "TRUE MATCH" if is_positive else "MISMATCH  "
        print(f"  [{label}] sim={sim:.4f}  -- {pair['note']}")
        if is_positive:
            positive_sims.append(sim)
        else:
            negative_sims.append(sim)

    mean_pos = float(np.mean(positive_sims))
    mean_neg = float(np.mean(negative_sims))
    gap = mean_pos - mean_neg

    print(f"\n=== Summary ===")
    print(f"  True-match pairs:  mean cosine sim = {mean_pos:.4f}  (n={len(positive_sims)})")
    print(f"  Mismatch pairs:    mean cosine sim = {mean_neg:.4f}  (n={len(negative_sims)})")
    print(f"  Separation (positive - negative):    {gap:.4f}")

    print(f"\n=== GO / NO-GO ===")
    if gap > 0.05:
        print(f"  GO -- clear separation (gap={gap:.4f}). Proceed to build full embeddings.")
    elif gap > 0:
        print(f"  MARGINAL -- some separation (gap={gap:.4f}), but weaker than hoped. "
              f"Proceed with caution; the embedding feature may contribute less than expected.")
    else:
        print(f"  NO-GO -- no separation, or negative pairs scored HIGHER (gap={gap:.4f}). "
              f"Stop and reconsider the model choice or tagging format before building further.")


if __name__ == "__main__":
    main()