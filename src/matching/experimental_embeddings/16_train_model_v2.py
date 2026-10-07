"""
16_train_model_v2.py

Step A: retrains XGBoost on the 11-feature set (original 10 +
name_address_embedding_cosine_sim), using training_features_v2.tsv
(the embedding-augmented feature table). Uses the same XGB_PARAMS as
the original baseline -- this is still a fast, sensible-defaults
retrain, not a new tuning pass, since the only change is one added
feature.
"""

import sys
import time
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, log_loss

sys.path.append(str(Path(__file__).resolve().parent))
from config import TRAINING_FEATURES_AUGMENTED_PATH, MODEL_V2_PATH, XGB_PARAMS, V2_FEATURE_COLUMNS
from io_utils import load_tsv


def load_features_split(path):
    """Loads the augmented feature file with explicit dtypes: all feature columns as float64, label as int, IDs/split as strings."""
    dtype_map = {col: "float64" for col in V2_FEATURE_COLUMNS}
    dtype_map["label"] = "int8"
    dtype_map["source1_entity_id"] = "str"
    dtype_map["candidate_entity_id"] = "str"
    dtype_map["split"] = "str"

    print(f"Loading {path.name}...")
    start = time.time()
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=dtype_map)
    print(f"  loaded {len(df):,} rows in {time.time() - start:.1f}s")
    return df


def main():
    df = load_features_split(TRAINING_FEATURES_AUGMENTED_PATH)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    print(f"\nTrain rows: {len(train_df):,}  |  Validation rows: {len(val_df):,}")

    X_train = train_df[V2_FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_val = val_df[V2_FEATURE_COLUMNS]
    y_val = val_df["label"]

    print(f"\nTrain label balance: {y_train.mean()*100:.2f}% positive")
    print(f"Val label balance:   {y_val.mean()*100:.2f}% positive")

    print("\nTraining XGBoost v2 (11 features)...")
    start = time.time()

    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=25,
    )

    elapsed = time.time() - start
    print(f"\nTraining finished in {elapsed:.1f}s "
          f"(stopped at boosting round {model.best_iteration}, of {XGB_PARAMS['n_estimators']} max)")

    print("\n=== Train vs Validation Diagnostics ===\n")
    for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val)]:
        probs = model.predict_proba(X)[:, 1]
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

    print("\n=== Feature Importance (gain-based) ===\n")
    importance = model.get_booster().get_score(importance_type="gain")
    for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {feat:35s}: {score:.2f}")

    print(f"\nCompare 'name_address_embedding_cosine_sim' ranking here against the "
          f"v1 (10-feature) model's top features. If the embedding feature ranks low, "
          f"it may be contributing less than hoped; if it ranks high, it's genuinely "
          f"adding new signal beyond the lexical features.")

    model.save_model(str(MODEL_V2_PATH))
    print(f"\nModel saved to: {MODEL_V2_PATH}")


if __name__ == "__main__":
    main()