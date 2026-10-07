"""
18_scan_test_candidate_refs.py

Step A: scans candidate_pairs.tsv (the real test blocking output)
respecting FINAL_CAP, to find every unique referenced entity ID. A
cheap, fast pass -- same discipline as 12_scan_train_embedding_refs.py
-- so the real scope is known before committing to the full inference
run.

Note: test-side embeddings (all test entities) were already built in
full by 11_build_test_embeddings.py, so no filtering is needed for
those -- this scan is only used to filter the NAME/ADDRESS lookups
(which hold full text, much heavier per entry than an embedding row
index).
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import CANDIDATES_TEST_PATH, FINAL_CAP, TEST_EMBED_REFS_S1_PATH, TEST_EMBED_REFS_OTHER_PATH


def main():
    s1_ids = set()
    other_ids = set()
    n_rows = 0

    print(f"Scanning {CANDIDATES_TEST_PATH.name} (cap={FINAL_CAP})...")
    start = time.time()
    with open(CANDIDATES_TEST_PATH, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            s1_ids.add(parts[0])
            cand_str = parts[1] if len(parts) > 1 else ""
            if cand_str:
                other_ids.update(cand_str.split(",")[:FINAL_CAP])
            n_rows += 1
            if n_rows % 200_000 == 0:
                elapsed = time.time() - start
                print(f"    {n_rows:,} rows scanned ({elapsed:.1f}s elapsed)", flush=True)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. {n_rows:,} rows scanned.")
    print(f"  Unique S1 entities:    {len(s1_ids):,}")
    print(f"  Unique S2/S3 entities: {len(other_ids):,}")

    with open(TEST_EMBED_REFS_S1_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(s1_ids))
    with open(TEST_EMBED_REFS_OTHER_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(other_ids))
    print(f"\nWritten to {TEST_EMBED_REFS_S1_PATH} and {TEST_EMBED_REFS_OTHER_PATH}")


if __name__ == "__main__":
    main()