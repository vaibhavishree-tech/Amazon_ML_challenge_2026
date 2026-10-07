"""
03_train_baseline_model.py

Trains the baseline XGBoost binary classifier on training_features.tsv,
using the existing (verified representative, per
02_check_split_stratification.py) train/val split.

Design choices:
  - Loads the feature file with explicit numeric dtypes for the feature
    columns (not the pyarrow string dtype used elsewhere in this
    project for raw text) -- here the relevant columns are genuinely
    numeric, and XGBoost requires numeric input. The file is a
    manageable size for pandas to handle directly without the batching
    discipline needed for the much larger raw text files earlier in
    this project.
  - Uses xgboost's sklearn-style API (XGBClassifier) with
    early_stopping_rounds, so training stops once validation logloss
    stops improving, rather than requiring a hand-picked exact number
    of boosting rounds.
  - Reports feature importance and train-vs-validation metrics as a
    basic diagnostic check -- sufficient to catch something obviously
    broken (e.g. severe overfitting), not intended as a deep tuning
    pass for this baseline.
  - Saves the trained model to disk in XGBoost's native JSON format, so
    later threshold-selection and inference scripts can load it without
    retraining.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, log_loss,
)

sys.path.append(str(Path(__file__).resolve().parent))
from config import TRAINING_FEATURES_PATH, BASELINE_MODEL_PATH, XGB_PARAMS
from text_similarity_features import FEATURE_COLUMNS


def load_features_split(path):
    """
    Loads the feature file with explicit dtypes: FEATURE_COLUMNS as
    float64 (XGBoost requires numeric input), label as int, split/IDs
    as plain strings. Specifying dtypes in one read_csv call is simpler
    and faster than loading generically and casting columns afterward.
    """
    dtype_map = {col: "float64" for col in FEATURE_COLUMNS}
    dtype_map["label"] = "int8"
    dtype_map["source1_entity_id"] = "str"
    dtype_map["candidate_entity_id"] = "str"
    dtype_map["split"] = "str"

    print(f"Loading {path.name}...")
    start = time.time()
    df = pd.read_csv(path, sep="\t", dtype=dtype_map)
    print(f"  loaded {len(df):,} rows in {time.time() - start:.1f}s")
    return df


def main():
    df = load_features_split(TRAINING_FEATURES_PATH)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    print(f"\nTrain rows: {len(train_df):,}  |  Validation rows: {len(val_df):,}")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["label"]

    print(f"\nTrain label balance: {y_train.mean()*100:.2f}% positive")
    print(f"Val label balance:   {y_val.mean()*100:.2f}% positive")

    print("\nTraining XGBoost baseline model...")
    start = time.time()

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=25,
    )

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed:.1f}s "
          f"(stopped at boosting round {model.best_iteration}, "
          f"of {XGB_PARAMS['n_estimators']} max)")

    # --- Diagnostics: train vs validation gap ---
    print("\n=== Train vs Validation Diagnostics ===\n")
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        probs = model.predict_proba(X)[:, 1]
        # threshold=0.5 here is only a diagnostic default -- the actual
        # decision threshold used for the real submission is selected
        # separately (see 05_score_and_select_threshold.py), chosen to
        # maximize F_0.5 rather than plain accuracy.
        preds = (probs >= 0.5).astype(int)
        print(f"-- {name} (threshold=0.5, diagnostic only) --")
        print(f"  logloss:   {log_loss(y, probs):.4f}")
        print(f"  roc_auc:   {roc_auc_score(y, probs):.4f}")
        print(f"  precision: {precision_score(y, preds):.4f}")
        print(f"  recall:    {recall_score(y, preds):.4f}")
        print(f"  f1:        {f1_score(y, preds):.4f}")
        print()

    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    gap = train_auc - val_auc
    print(f"Train/val ROC-AUC gap: {gap:.4f} "
          f"({'looks fine' if gap < 0.03 else 'WORTH A LOOK -- possible overfitting'})")

    # --- Feature importance ---
    print("\n=== Feature Importance (gain-based) ===\n")
    importance = model.get_booster().get_score(importance_type="gain")
    # get_score() returns keys using the actual feature column names
    # here, since training used a named pandas DataFrame rather than a
    # raw numpy array -- no index-to-name translation is needed.
    importance_named = importance
    for feat, score in sorted(importance_named.items(), key=lambda x: -x[1]):
        print(f"  {feat:30s}: {score:.2f}")

    # --- Save model ---
    model.save_model(str(BASELINE_MODEL_PATH))
    print(f"\nModel saved to: {BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    main()