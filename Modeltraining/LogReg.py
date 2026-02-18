#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logistic Regression classifier for next-hop selection.

Reads a feature CSV, splits into train/validation, trains a LogisticRegression,
reports metrics and saves the trained model plus coefficient-based feature-importance
plots and a confusion matrix.
"""

from __future__ import annotations

import os
import time
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# Configuration
INPUT_CSV = os.path.join("Feature_Calc", "results", "train_features_normalized.csv")
RESULTS_DIR = os.path.join("Modeltraining", "results")
MODEL_OUT = os.path.join(RESULTS_DIR, "logreg_model.joblib")
FIG_IMPORTANCE_ABS = os.path.join(RESULTS_DIR, "logreg_feature_importance_abs.png")
FIG_IMPORTANCE_SIGNED = os.path.join(RESULTS_DIR, "logreg_feature_importance_signed.png")
FIG_CM = os.path.join(RESULTS_DIR, "logreg_confusion_matrix.png")

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
    logreg = LogisticRegression(
        # penalty="l2",
        # C=1.0,
        # solver="lbfgs",
        # max_iter=200,
        # class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining classifier...")
    train_start = time.time()
    logreg.fit(X_train, y_train)
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.2f} s")

    print("\nPredicting on validation set...")
    pred_start = time.time()
    y_pred = logreg.predict(X_val)
    pred_time = time.time() - pred_start
    print(f"Prediction time: {pred_time:.2f} s")

    cm = confusion_matrix(y_val, y_pred, normalize="all")
    print("\nConfusion matrix (normalized):\n", cm)
    print("\nClassification report:\n", classification_report(y_val, y_pred, digits=3))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(logreg, MODEL_OUT)
    print(f"\nSaved model (joblib): {MODEL_OUT}")

    coef = logreg.coef_.ravel()
    feat_names = X.columns.tolist()
    label_names = [FEATURE_LABEL.get(f, f) for f in feat_names]
    coef_df = pd.DataFrame({"feature": feat_names, "label": label_names, "coef": coef})
    coef_df["abs_coef"] = np.abs(coef_df["coef"])

    coef_abs_sorted = coef_df.sort_values("abs_coef", ascending=True)
    plt.figure(figsize=(10, max(6, 0.3 * len(coef_abs_sorted))))
    plt.barh(coef_abs_sorted["label"], coef_abs_sorted["abs_coef"], color="steelblue")
    plt.xlabel("Absolute Coefficient |coef|")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_IMPORTANCE_ABS, dpi=150)
    plt.close()
    print(f"Saved importance plot (abs coef): {FIG_IMPORTANCE_ABS}")

    coef_signed_sorted = coef_df.sort_values("coef", ascending=True)
    colors = ["salmon" if v < 0 else "skyblue" for v in coef_signed_sorted["coef"]]
    plt.figure(figsize=(10, max(6, 0.3 * len(coef_signed_sorted))))
    plt.barh(coef_signed_sorted["label"], coef_signed_sorted["coef"], color=colors)
    plt.axvline(0, color="k", lw=1, ls="--", alpha=0.7)
    plt.xlabel("Signed Coefficient (coef)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIG_IMPORTANCE_SIGNED, dpi=150)
    plt.close()
    print(f"Saved importance plot (signed coef): {FIG_IMPORTANCE_SIGNED}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_).plot(
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

