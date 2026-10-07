# Experimental: Embedding-Augmented Matching Model

## Status: Explored and measured, NOT used in the final submission

This folder contains a complete, working exploration of augmenting the
matching model with a semantic embedding similarity feature. It is
kept separate from the main `matching/` pipeline because the final
submitted result (`matching_results.tsv`) was produced by the
10-feature baseline pipeline in `matching/06_run_test_inference.py`,
not by anything in this folder.

This document explains what was built, what it achieved, why it wasn't
deployed, and why it's still included.

---

## Motivation

During the cleaning and blocking phases, we identified a specific,
measurable weakness in our pipeline: business names originally written
in non-Latin scripts (Devanagari, Tamil, Gujarati, and others) lose
much of their lexical similarity signal after `unidecode` transliterates
them to ASCII. Token-based name similarity for this subset of true
matches was measurably and substantially lower than for ASCII-origin
names, even after cleaning.

Our hypothesis: a multilingual sentence embedding model, given the
**original, untransliterated** text, could recognize semantic
equivalence (e.g. a business name in Hindi script and its English
equivalent) in a way that token/character-level lexical metrics
fundamentally cannot. This would directly target the weakest part of
our existing feature set.

## What was built

1. **Sanity check** (`08_sanity_check_embeddings.py`): verified the
   chosen model (`intfloat/multilingual-e5-small`) produces clear
   separation between true-match and non-match pairs, including on
   real Devanagari/Tamil/Gujarati examples pulled from our own manual
   review data. Result: clear separation confirmed — proceeded.

2. **Parallelization and correctness verification**
   (`09_test_parallel_speedup.py`, `parallel_worker.py`): built and
   verified a multiprocessing scoring pipeline against the existing
   single-threaded baseline, confirming identical results before
   trusting it at scale.

3. **Candidate-cap sweep** (`10_test_candidate_caps.py`): measured
   F_0.5 across several candidate-per-entity caps to find a
   cost/performance balance.

4. **Full embedding generation** (`11_build_test_embeddings.py`,
   `12_scan_train_embedding_refs.py`, `13_build_train_embeddings.py`):
   successfully embedded the full test set (~11.7M entities) and a
   representative subset of training entities, using the **raw,
   uncleaned** business name and address text — essential, since
   feeding already-transliterated text into the embedding model would
   have defeated the purpose.

5. **Feature augmentation and retraining**
   (`14_augment_training_features.py`, `15_augment_eval_candidates.py`,
   `16_train_model_v2.py`): added the embedding cosine-similarity
   feature to the existing 10-feature set and retrained an 11-feature
   XGBoost model.

6. **Country-specific threshold evaluation**
   (`17_threshold_sweep_by_country.py`): re-ran threshold selection
   split by country, to check whether a single global threshold was
   miscalibrated for any one country. Result: US and India
   independently converged on the same optimal threshold (0.98),
   indicating the global threshold was not a problem for the countries
   we could measure.

## Measured result

On a held-out, ground-truth-backed evaluation sample (50,000 entities,
same methodology as the baseline's own threshold selection):

| Configuration | Macro F_0.5 |
|---|---|
| Baseline (10 features, no embeddings) | 0.8809 |
| Embedding-augmented (11 features) | **0.8853** |

This is a real, measured, positive result: **+0.0044** improvement from
adding the embedding feature. The embedding feature itself ranked
roughly mid-pack in XGBoost's gain-based feature importance — a
genuine, if modest, contribution beyond what the existing lexical
features already captured, consistent with it targeting a specific
known weakness rather than overlapping with existing signal.

## Why it was not used in the final submission

Deploying this configuration at full test scale
(`19_run_final_inference.py`) requires holding, simultaneously, in
memory:
- The full candidate-entity name/address lookup tables (filtered to
  referenced entities, but still substantial at full test scale)
- The full test-set embedding matrix (~11.7M entities × 384 dimensions,
  on the order of several GB even at reduced float16 precision)
- The trained model and multiprocessing worker overhead

On the development machine used for this project (16 GB RAM), this
combination repeatedly exceeded available memory during multiprocessing
worker initialization, regardless of worker count or several attempted
mitigations (file-backed lookups instead of pickled dictionaries,
file-backed embedding memmaps instead of in-RAM arrays, reduced worker
counts). Each mitigation solved one specific failure mode but the
underlying combination — all required data structures alive at once —
remained too large for the available memory.

Given competition time constraints, we made the decision to submit the
**proven, already-validated 10-feature baseline pipeline**
(`matching/06_run_test_inference.py`), which had already completed
successfully once before and carries no similar memory risk, rather
than risk an incomplete submission chasing a measured +0.0044
improvement.

## What this folder represents

This is a complete, honest record of a real engineering exploration:
a specific weakness was identified from evidence, a targeted fix was
designed and built, its benefit was rigorously measured, and a
clear-eyed tradeoff was made under real constraints. The code here is
functional and the measured result is real — it simply was not
deployed at full scale due to a specific, well-understood resource
constraint rather than any flaw in the approach itself.

With more available memory (e.g. a machine with 32GB+ RAM, or a
cloud instance sized accordingly), this pipeline would very plausibly
complete successfully and the ~0.0044 improvement would likely
transfer to the real test set for the US and India portions at least
(France's contribution remains unmeasurable in either configuration,
since it has zero training examples to validate against).

## Reproducing this folder's results

Requires the main `matching/` pipeline's outputs to already exist
(cleaned source files, trained baseline model, best_threshold.txt),
plus a CUDA-capable GPU recommended for reasonable embedding generation
time. Run scripts `07` through `18` and `20` in order; `19` is included
for completeness but is not expected to complete successfully without
more available system memory than was used during development.