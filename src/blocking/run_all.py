"""
run_all.py

Runs the SAFE portion of the blocking phase: build train candidates,
then measure recall ceiling against ground truth.

    python run_all.py

03_build_candidates_test.py is deliberately NOT included here -- it
should only be run manually, after reviewing 02's recall ceiling report
and being satisfied with it. Running the full test build before
confirming train recall wastes time re-running an expensive step that
may still need tuning (document-frequency thresholds, rare-K, or the
per-entity candidate cap in config.py) based on what 02 shows.
"""

import runpy
from pathlib import Path

SCRIPT_ORDER = [
    "01_build_candidates_train.py",
    "02_measure_recall_ceiling.py",
]


def main():
    here = Path(__file__).resolve().parent
    for script_name in SCRIPT_ORDER:
        script_path = here / script_name
        print("\n" + "=" * 80)
        print(f"Running {script_name}")
        print("=" * 80 + "\n")
        runpy.run_path(str(script_path), run_name="__main__")

    print("\n" + "=" * 80)
    print("Train candidates built and recall ceiling measured.")
    print("Review blocking_outputs/reports/02_recall_ceiling_report.txt.")
    print("If satisfied, run 03_build_candidates_test.py manually.")
    print("=" * 80)


if __name__ == "__main__":
    main()