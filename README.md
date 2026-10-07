
# Business Entity Resolution — Amazon ML Challenge 2026

**Public leaderboard: F_0.5 = 0.85** · Solo-built end-to-end pipeline (EDA → cleaning → blocking → XGBoost matching)

## Results

| Metric | Score |
|---|---|
| Public leaderboard F_0.5 | **0.85** |
| Local validation estimate (cap=300) | 0.8809 |
| Embedding-augmented variant (local only, not deployed) | 0.8853 |

```text
This README describes how to reproduce `matching_results.tsv` and `candidate_pairs.tsv` from the provided dataset.

For the full methodology, findings, and reasoning behind each design decision, see `DOCUMENTATION.md`.
```

---

## Requirements

- Python 3.10+
- Install dependencies:
```bash
  pip install -r requirements.txt
```

The `experimental_embeddings/` subfolder under `matching/` has its own separate `requirements_experimental.txt` — not required to reproduce the submitted result (see note below).

---

## Runtime & Resource Expectations

This pipeline processes multi-million-row datasets. Exact runtimes depend on hardware, but expect **cleaning, blocking, and matching to each take meaningfully long** (potentially 30+ minutes to multiple hours per phase) on a 16GB-class machine — the matching phase's final inference step alone took several hours during development.

**Minimum recommended**: 16GB RAM. Scripts print progress regularly — a quiet terminal for several minutes during a known-slow step (e.g. unidecode transliteration, full test inference) is expected behavior, not a hang.

---

## Directory Structure

```text
amazon_ml_challenge_2026/
├── src/ 
│ ├── eda/
│ ├── cleaning/
│ ├── blocking/
│ └── matching/
│       └── experimental_embeddings/ # not used in final submission
├── README.md
└── DOCUMENTATION.md #Refer to this for complete understanding of my approach
```

The dataset is **not included** in this repository. Before running anything, place the competition dataset at:
src/dataset/train/
src/dataset/test/

---

## How to Reproduce the Submission

Run the following scripts in order from inside `src/`.

### 1. EDA (optional, audit only)

```bash
cd eda
python run_all.py
```

Produces audit reports only. Not required by later phases, but documents the findings referenced throughout `DOCUMENTATION.md`.

### 2. Cleaning

```bash
cd cleaning
python run_all.py
```

Produces cleaned train/test source files. Required before blocking.

### 3. Blocking

```bash
cd blocking
python run_all.py
```

Builds training candidates and measures recall ceiling against ground truth.

Once satisfied with the recall ceiling report, build the real test candidate file:

```bash
python 03_build_candidates_test.py
```

This produces `candidate_pairs.tsv`.

### 4. Matching

```bash
cd matching
python 00_prepare_training_sample.py
python 01_compute_training_features.py
python 02_check_split_stratification.py
python 03_train_baseline_model.py
python 04_extract_eval_candidates.py
python 05_score_and_select_threshold.py
python 06_run_test_inference.py
```

`06_run_test_inference.py` produces `matching_results.tsv`.

---

## About `experimental_embeddings/`

This subfolder contains a separate, exploratory extension — a semantic embedding feature that measurably improved local validation performance but was **not used in the final submission** due to memory constraints at full test scale. 
Not required to reproduce the submitted result.
See `experimental_embeddings/README_EXPERIMENTAL.md` for details, and Section 6 of `DOCUMENTATION.md` for the full story.

---

## Final Output

After running all steps above:

| File | Produced by |
|---|---|
| `candidate_pairs.tsv` | `blocking/03_build_candidates_test.py` |
| `matching_results.tsv` | `matching/06_run_test_inference.py` |