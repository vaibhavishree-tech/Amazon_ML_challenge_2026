"""
run_all.py

Convenience runner that executes all five EDA scripts in order, so the
full Phase 1 / Step 1 EDA can be regenerated with a single command:

    python run_all.py

Each script also writes its own report file into eda_outputs/, so any
single script can still be run independently later (e.g. after a data
refresh) without re-running the others.
"""

import runpy
from pathlib import Path

# Execution order matters only in that each script is self-contained and
# writes its own output file -- there is no dependency between them, but
# running them in this numbered order keeps console output readable and
# matches the order the EDA phase was designed to be read in.
SCRIPT_ORDER = [
    "01_volume_integrity.py",
    "02_ground_truth_topology.py",
    "03_country_analysis.py",
    "04_text_vocab_encoding.py",
    "05_pairwise_noise_mining.py",
]


def main():
    here = Path(__file__).resolve().parent
    for script_name in SCRIPT_ORDER:
        script_path = here / script_name
        print("\n" + "=" * 80)
        print(f"Running {script_name}")
        print("=" * 80 + "\n")
        # runpy.run_path executes the target script as if it were run
        # directly from the command line (run_name="__main__" makes its
        # own "if __name__ == '__main__':" block fire), rather than
        # importing it as a module -- this avoids needing each script to
        # expose a separately-callable function just for this runner.
        runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()