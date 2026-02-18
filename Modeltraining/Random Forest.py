#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest classifier for next-hop selection.

Reads a feature CSV, splits into train/validation, trains a RandomForestClassifier,
reports metrics and saves the trained model plus feature-importance and confusion
matrix plots.
"""

from __future__ import annotations

import os
import time
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Configuration
INPUT_CSV = os.path.join("Feature_Calc", "results", "train_features_normalized.csv")
RESULTS_DIR = os.path.join("Modeltraining", "results")
MODEL_OUT = os.path.join(RESULTS_DIR, "rf_model.joblib")
FIG_IMPORTANCE = os.path.join(RESULTS_DIR, "rf_feature_importance.png")
FIG_CM = os.path.join(RESULTS_DIR, "rf_confusion_matrix.png")

RANDOM_STATE = 42
TEST_SIZE = 0.2


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
    rf = RandomForestClassifier(
        n_estimators=300,
        # max_depth=None,
        # min_samples_split=2,
        # min_samples_leaf=1,
        # max_features="sqrt",
        # bootstrap=True,
        # class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining classifier...")
    train_start = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.2f} s")

    print("\nPredicting on validation set...")
    pred_start = time.time()
    y_pred = rf.predict(X_val)
    pred_time = time.time() - pred_start
    print(f"Prediction time: {pred_time:.2f} s")

    cm = confusion_matrix(y_val, y_pred, normalize="all")
    print("\nConfusion matrix (normalized):\n", cm)
    print("\nClassification report:\n", classification_report(y_val, y_pred, digits=3))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(rf, MODEL_OUT)
    print(f"\nSaved model: {MODEL_OUT}")

    importances = rf.feature_importances_
    feat_names = X.columns.to_list()
    label_names = [FEATURE_LABEL.get(f, f) for f in feat_names]
    imp_df = pd.DataFrame({"feature": feat_names, "label": label_names, "importance": importances})

    imp_sorted = imp_df.sort_values("importance", ascending=True)
    plt.figure(figsize=(10, max(6, 0.3 * len(imp_sorted))))
    plt.barh(imp_sorted["label"], imp_sorted["importance"], color="steelblue")
    plt.xlabel("Feature Importance (Gini)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_IMPORTANCE, dpi=150)
    plt.close()
    print(f"Saved importance plot: {FIG_IMPORTANCE}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_).plot(
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

