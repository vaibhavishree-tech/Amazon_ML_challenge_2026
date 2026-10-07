"""
run_all.py

Runs the full cleaning phase in order: clean all six files, validate
the cleaning against the original EDA numbers, then validate
transliteration quality on a real non-ASCII sample.

    python run_all.py

Each script can still be run individually afterward -- e.g. if you
adjust a rule in text_cleaning_rules.py and only need to re-run
01 and 02.
"""

import runpy
from pathlib import Path

# Order matters: 01 must run first since 02/03/04 all read from
# cleaned_outputs/, which only exists after 01 has written it. 02, 03,
# 04 are otherwise independent validation passes and could technically
# run in any order relative to each other, but are kept in this
# numbered sequence for readable console output.
SCRIPT_ORDER = [
    "01_clean_source_files.py",
    "02_validate_cleaning.py",
    "03_validate_transliteration_sample.py",
    "04_quantify_transliteration_impact.py",
]


def main():
    here = Path(__file__).resolve().parent
    for script_name in SCRIPT_ORDER:
        script_path = here / script_name
        print("\n" + "=" * 80)
        print(f"Running {script_name}")
        print("=" * 80 + "\n")
        runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()