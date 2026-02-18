#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM binary classifier for next-hop selection.

Reads a feature CSV, splits into train/validation, trains an LGBMClassifier,
reports metrics and saves the trained model plus feature-importance and confusion
matrix plots.
"""

from __future__ import annotations

import os
import time
from typing import List

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Configuration
INPUT_CSV = os.path.join("Feature_Calc", "results", "train_features_normalized.csv")
RESULTS_DIR = os.path.join("Modeltraining", "results")
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODEL_OUT_JOBLIB = os.path.join(RESULTS_DIR, "lgbm_model.joblib")
MODEL_OUT_TXT = os.path.join(RESULTS_DIR, "lgbm_model.txt")
FIG_IMPORTANCE_GAIN = os.path.join(RESULTS_DIR, "lgbm_feature_importance_gain.png")
FIG_IMPORTANCE_SPLIT = os.path.join(RESULTS_DIR, "lgbm_feature_importance_split.png")
FIG_CM = os.path.join(RESULTS_DIR, "lgbm_confusion_matrix.png")


FEATURE_LABEL = {
    "advance": r"$GA_i$",
    "bc_y": r"$BC_i$",
    "cc_y": r"$CC_i$",
    "BC": r"$BC_i$",
    "CC": r"$CC_i$",
    "turn_ang": r"$\theta_i^t$",
    "cosang": r"$\theta_i^r$",
    "angle_yd": r"$\theta_i^a$",
    "dist_xy": r"$d_i$",
    "deg_y": r"$deg_i$",
}


def main() -> None:
    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    # Remove non-feature columns if present
    df = df.drop(columns=["snap", "N", "prev", "x", "y"], errors="ignore")

    X = df.drop(columns=["label"])  # all columns except 'label'
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Model
    lgbm = lgb.LGBMClassifier(
        objective="binary",
        # boosting_type="gbdt",
        n_estimators=300,
        # learning_rate=0.06,
        # num_leaves=127,
        # max_depth=-1,
        # subsample=0.8,
        # colsample_bytree=0.8,
        # reg_lambda=0.0,
        # reg_alpha=0.0,
        # class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining classifier...")
    train_start = time.time()
    lgbm.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.2f} s")

    print("\nPredicting on validation set...")
    pred_start = time.time()
    y_pred = lgbm.predict(X_val)
    pred_time = time.time() - pred_start
    print(f"Prediction time: {pred_time:.2f} s")

    cm = confusion_matrix(y_val, y_pred, normalize="all")
    print("\nConfusion matrix (normalized):\n", cm)
    print("\nClassification report:\n", classification_report(y_val, y_pred, digits=3))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(lgbm, MODEL_OUT_JOBLIB)
    lgbm.booster_.save_model(MODEL_OUT_TXT)
    print(f"\nSaved model (joblib): {MODEL_OUT_JOBLIB}")
    print(f"Saved booster model (.txt): {MODEL_OUT_TXT}")

    booster = lgbm.booster_
    feat_names: List[str] = X.columns.to_list()
    gain_importance = booster.feature_importance(importance_type="gain")
    split_importance = booster.feature_importance(importance_type="split")
    imp_df = pd.DataFrame(
        {"feature": feat_names, "importance_gain": gain_importance, "importance_split": split_importance}
    )

    imp_gain_sorted = imp_df.sort_values("importance_gain", ascending=True)
    labels_gain = [FEATURE_LABEL.get(f, f) for f in imp_gain_sorted["feature"]]
    ypos_gain = np.arange(len(imp_gain_sorted))
    plt.figure(figsize=(10, max(6, 0.3 * len(imp_gain_sorted))))
    plt.barh(ypos_gain, imp_gain_sorted["importance_gain"], color="seagreen")
    plt.yticks(ypos_gain, labels_gain)
    plt.xlabel("Feature Importance (Gain)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_IMPORTANCE_GAIN, dpi=150)
    plt.close()
    print(f"Saved importance plot (gain): {FIG_IMPORTANCE_GAIN}")

    imp_split_sorted = imp_df.sort_values("importance_split", ascending=True)
    labels_split = [FEATURE_LABEL.get(f, f) for f in imp_split_sorted["feature"]]
    ypos_split = np.arange(len(imp_split_sorted))
    plt.figure(figsize=(10, max(6, 0.3 * len(imp_split_sorted))))
    plt.barh(ypos_split, imp_split_sorted["importance_split"], color="steelblue")
    plt.yticks(ypos_split, labels_split)
    plt.xlabel("Feature Importance (Split)")
    plt.ylabel("Feature")
    # plt.title("LightGBM Feature Importance (Split)")
    plt.tight_layout()
    plt.savefig(FIG_IMPORTANCE_SPLIT, dpi=150)
    plt.close()
    print(f"Saved importance plot (split): {FIG_IMPORTANCE_SPLIT}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lgbm.classes_).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format=".4f"
    )
    plt.tight_layout()
    plt.savefig(FIG_CM, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix (normalized): {FIG_CM}")

    print("\n--- Runtime Summary ---")
    print(f"Training time: {train_time:.2f} s")
    print(f"Prediction time: {pred_time:.2f} s")
    print(f"Total time: {train_time + pred_time:.2f} s")


if __name__ == "__main__":
    main()

