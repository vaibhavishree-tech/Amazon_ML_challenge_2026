"""
parallel_worker.py

Module-level worker functions for multiprocessing feature computation.

Windows multiprocessing detail: on Windows, multiprocessing uses the
"spawn" start method rather than "fork". Every worker process is a
genuinely fresh Python interpreter with no inherited memory from the
parent -- it does not see any variable, dataframe, or dict that existed
in the main process before the pool was created. Everything a worker
needs must be explicitly passed through Pool's initializer/initargs
(once, at worker startup) or through the task arguments themselves
(once per task). Functions passed to Pool.map/imap must also be defined
at module level (not as a nested function or lambda) so they can be
pickled and sent to the worker process.

Memory design: workers do not each load the full cleaned/raw source
files, which would multiply memory use by the worker count. Instead,
the calling script filters the lookups down to only the entity IDs
actually referenced in the specific candidate file being processed,
before creating the pool, and passes those small, filtered dicts via
initargs. Each worker receives its own copy (pickled once at pool
startup, not per task), far smaller than the full source-file lookups.
"""

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.append(str(Path(__file__).resolve().parent))
from text_similarity_features import compute_pair_features

# Populated once per worker process by init_worker() -- module-level
# globals are how data persists across multiple score_batch() calls
# within the SAME worker, without needing to reload anything per task.
_S1_LOOKUP = None
_OTHER_LOOKUP = None
_RAW_NAME_LOOKUP = None
_MODEL = None
_MAX_CANDIDATES = None


def init_worker(s1_lookup, other_lookup, raw_name_lookup, model_path, max_candidates):
    """
    Runs ONCE when each worker process starts (via Pool's initializer
    argument), not once per task. This is where the filtered lookups
    and the model get loaded into this worker's own memory space.
    """
    global _S1_LOOKUP, _OTHER_LOOKUP, _RAW_NAME_LOOKUP, _MODEL, _MAX_CANDIDATES
    _S1_LOOKUP = s1_lookup
    _OTHER_LOOKUP = other_lookup
    _RAW_NAME_LOOKUP = raw_name_lookup
    _MAX_CANDIDATES = max_candidates

    _MODEL = xgb.XGBClassifier()
    _MODEL.load_model(model_path)


def score_batch(lines: list) -> list:
    """
    Processes one batch (chunk) of raw candidate-file lines. Each line
    is "source1_entity_id\\tcandidate_entity_ids" (comma-separated,
    possibly empty).

    Returns a flat list of (s1_id, candidate_id_or_None, probability_or_None)
    tuples. A zero-candidate entity still produces exactly one tuple
    with candidate_id=None, probability=None -- this ensures the main
    process knows that entity existed even though it contributed no
    scoreable pairs, which matters for correctly computing macro
    F_0.5 later (an entity must be counted even if blocking gave it
    nothing to score).
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
            results.append((s1_id, None, None))
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

            features = compute_pair_features(
                s1_name, s1_addr, s1_postal,
                other_name, other_addr, other_postal,
                other_name_original,
            )
            feature_rows.append(features)
            valid_candidate_ids.append(cand_id)

        if not feature_rows:
            results.append((s1_id, None, None))
            continue

        feature_matrix = np.array(feature_rows, dtype="float64")
        probs = _MODEL.predict_proba(feature_matrix)[:, 1]

        for cand_id, prob in zip(valid_candidate_ids, probs):
            results.append((s1_id, cand_id, float(prob)))

    return results