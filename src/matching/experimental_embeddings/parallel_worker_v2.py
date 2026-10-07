"""
parallel_worker_v2.py

Worker module extending parallel_worker.py's multiprocessing design
with the 11th feature (embedding cosine similarity).

Embeddings are loaded from a FILE-BACKED memmap (not an anonymous
multiprocessing.shared_memory block): each worker opens the same
on-disk embeddings file independently in read-only mode. All workers
therefore share the OS's underlying page cache for that file rather
than each holding a separate in-process copy, while avoiding the
pagefile-backed virtual memory ceiling that an anonymous shared-memory
allocation of this size would be subject to on Windows.
"""

import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from text_similarity_features import compute_pair_features

_S1_LOOKUP = None
_OTHER_LOOKUP = None
_RAW_NAME_LOOKUP = None
_MODEL = None
_MAX_CANDIDATES = None
_THRESHOLD = None
_EMBED_ID_TO_ROW = None
_EMBED_SHM = None
_EMBED_ARRAY = None


def init_worker_v2(s1_lookup_path, other_lookup_path, raw_name_lookup_path, model_path,
                    max_candidates, threshold, embed_id_to_row_path,
                    embed_path, embed_shape):
    """
    Runs once per worker process at pool startup. Lookups and the
    embedding-ID index are loaded from pickle files on disk rather than
    passed directly through initargs, so that only small file-path
    strings (not large in-memory dictionaries) need to be transmitted
    to each spawned worker process.
    """
    global _S1_LOOKUP, _OTHER_LOOKUP, _RAW_NAME_LOOKUP, _MODEL
    global _MAX_CANDIDATES, _THRESHOLD, _EMBED_ID_TO_ROW, _EMBED_ARRAY

    import pickle
    with open(s1_lookup_path, "rb") as f:
        _S1_LOOKUP = pickle.load(f)
    with open(other_lookup_path, "rb") as f:
        _OTHER_LOOKUP = pickle.load(f)
    with open(raw_name_lookup_path, "rb") as f:
        _RAW_NAME_LOOKUP = pickle.load(f)
    with open(embed_id_to_row_path, "rb") as f:
        _EMBED_ID_TO_ROW = pickle.load(f)

    _MAX_CANDIDATES = max_candidates
    _THRESHOLD = threshold

    _MODEL = xgb.XGBClassifier()
    _MODEL.load_model(model_path)

    # Loaded fully into RAM rather than kept as a lazy memmap: with a
    # single worker process, random-access reads directly against a
    # memory-mapped file on disk are markedly slower per lookup than
    # reading from an in-RAM array, since each access can trigger a
    # separate disk read rather than a cached memory access. Loading
    # once into RAM up front trades a one-time load cost for
    # consistently fast per-pair lookups during scoring.
    _EMBED_ARRAY = np.array(np.memmap(embed_path, dtype="float16", mode="r", shape=embed_shape))


def _embedding_sim(s1_id, cand_id):
    """Looks up both entities' precomputed embedding rows and returns their cosine similarity (dot product, since embeddings are pre-normalized)."""
    s1_row = _EMBED_ID_TO_ROW.get(s1_id)
    cand_row = _EMBED_ID_TO_ROW.get(cand_id)
    if s1_row is None or cand_row is None:
        return 0.0
    vec_a = _EMBED_ARRAY[s1_row].astype("float32")
    vec_b = _EMBED_ARRAY[cand_row].astype("float32")
    return float(np.dot(vec_a, vec_b))


def score_and_predict_batch(lines: list):
    """
    Processes a batch of raw candidate-file lines. Returns
    (s1_id, matched_ids_string) tuples -- one per S1 entity in the batch,
    ALREADY thresholded, so the main process only needs to write results
    directly, no further scoring logic required there.
    """
    results = []

    for line in lines:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        s1_id = parts[0]
        cand_str = parts[1] if len(parts) > 1 else ""

        if not cand_str:
            results.append((s1_id, ""))
            continue

        candidate_ids = cand_str.split(",")[:_MAX_CANDIDATES]
        s1_name, s1_addr, s1_postal = _S1_LOOKUP.get(s1_id, ("", "", ""))

        feature_rows = []
        valid_candidate_ids = []
        for cand_id in candidate_ids:
            other_fields = _OTHER_LOOKUP.get(cand_id)
            if other_fields is None:
                other_name, other_addr, other_postal = "", "", ""
            else:
                other_name, other_addr, other_postal = other_fields
            other_name_original = _RAW_NAME_LOOKUP.get(cand_id, "")

            base_features = compute_pair_features(
                s1_name, s1_addr, s1_postal,
                other_name, other_addr, other_postal,
                other_name_original,
            )
            embed_sim = _embedding_sim(s1_id, cand_id)
            feature_rows.append(base_features + [embed_sim])
            valid_candidate_ids.append(cand_id)

        if not feature_rows:
            results.append((s1_id, ""))
            continue

        feature_matrix = np.array(feature_rows, dtype="float64")
        probs = _MODEL.predict_proba(feature_matrix)[:, 1]

        matched = [cid for cid, p in zip(valid_candidate_ids, probs) if p >= _THRESHOLD]
        results.append((s1_id, ",".join(matched)))

    return results