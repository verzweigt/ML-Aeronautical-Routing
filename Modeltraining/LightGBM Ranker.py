#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LightGBM ranking model (Lambdarank) for neighbor selection.

Trains a group-wise ranker over candidates y for each scenario (group defined by
columns such as snap/N/prev/x). Evaluates top-1 accuracy per scenario and saves the
model and feature importance plots.
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
from sklearn.model_selection import train_test_split


# === Configuration (edit as needed) ===
INPUT_CSV = os.path.join("Feature_Calc", "results", "train_features_normalized.csv")
RESULTS_DIR = os.path.join("Modeltraining", "results")
TEST_SIZE = 0.2
RANDOM_STATE = 42
FILTER_GROUPS = False  # filter training groups with all-0 or all-1 labels


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


def prepare_groups(df: pd.DataFrame) -> pd.DataFrame:
    group_cols_candidates: List[str] = ["snap", "N", "prev", "x"]
    group_cols = [c for c in group_cols_candidates if c in df.columns]
    if not group_cols:
        raise ValueError("No grouping columns found (expected e.g., 'snap','N','prev','x').")
    group_key = df[group_cols].astype(str).agg("|".join, axis=1)
    df["__group_id__"] = pd.Categorical(group_key).codes
    return df


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading CSV: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    if "label" not in df.columns:
        raise ValueError("Column 'label' not found in input CSV.")

    df = prepare_groups(df)

    group_cols_candidates = ["snap", "N", "prev", "x"]
    group_cols = [c for c in group_cols_candidates if c in df.columns]
    feature_cols = [c for c in df.columns if c not in (set(["label", "y", "__group_id__"]) | set(group_cols))]

    unique_groups = np.unique(df["__group_id__"])
    g_train, g_val = train_test_split(unique_groups, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    train_mask = df["__group_id__"].isin(g_train)
    val_mask = df["__group_id__"].isin(g_val)

    train_df = df.loc[train_mask, ["__group_id__", "label"] + feature_cols].copy()
    val_df = df.loc[val_mask, ["__group_id__", "label"] + feature_cols].copy()

    train_df.sort_values(["__group_id__"], kind="stable", inplace=True)
    val_df.sort_values(["__group_id__"], kind="stable", inplace=True)

    train_group_sizes = train_df.groupby("__group_id__").size().tolist()
    val_group_sizes = val_df.groupby("__group_id__").size().tolist()

    if FILTER_GROUPS:
        g_stats = train_df.groupby("__group_id__")["label"].agg(["sum", "count"]).reset_index()
        valid_g = g_stats[(g_stats["sum"] > 0) & (g_stats["sum"] < g_stats["count"])]["__group_id__"].tolist()
        if valid_g:
            before = len(train_df)
            train_df = train_df[train_df["__group_id__"].isin(valid_g)].copy()
            train_df.sort_values(["__group_id__"], kind="stable", inplace=True)
            train_group_sizes = train_df.groupby("__group_id__").size().tolist()
            print(f"Filtered training rows to groups with mixed labels: {len(train_df)}/{before}")
        else:
            print("Warning: No mixed-label training groups found; learning signal may be weak.")

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["label"].astype(int)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df["label"].astype(int)

    # Ranker definition
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        # num_leaves=127,
        # max_depth=7,
        # learning_rate=0.05,
        # min_child_samples=20,
        # subsample=0.9,
        # colsample_bytree=0.8,
        # class_weight="balanced",
    )

    print("\nTraining group-wise ranking model...")
    train_start = time.time()
    ranker.fit(
        X_train,
        y_train,
        group=train_group_sizes,
        eval_set=[(X_val, y_val)],
        eval_group=[val_group_sizes],
        eval_at=[1],
    )
    train_time = time.time() - train_start
    print(f"Training time: {train_time:.2f} s")

    print("\nScoring validation groups...")
    pred_start = time.time()
    val_scores = ranker.predict(X_val)
    pred_time = time.time() - pred_start
    print(f"Prediction time: {pred_time:.2f} s")

    top1_correct = 0
    idx = 0
    for gsize in val_group_sizes:
        group_scores = val_scores[idx : idx + gsize]
        group_labels = y_val.iloc[idx : idx + gsize].to_numpy()
        best_pos = int(np.argmax(group_scores))
        if group_labels[best_pos] == 1:
            top1_correct += 1
        idx += gsize
    top1_acc = top1_correct / len(val_group_sizes) if len(val_group_sizes) > 0 else float("nan")
    print(f"Top-1 accuracy per scenario (val): {top1_acc:.4f}")

    model_out_joblib = os.path.join(RESULTS_DIR, "lgbm_ranker_model.joblib")
    model_out_txt = os.path.join(RESULTS_DIR, "lgbm_ranker_model.txt")
    joblib.dump(ranker, model_out_joblib)
    ranker.booster_.save_model(model_out_txt)
    print(f"\nSaved model (joblib): {model_out_joblib}")
    print(f"Saved booster model (.txt): {model_out_txt}")

    booster = ranker.booster_
    feat_names = feature_cols
    gain_importance = booster.feature_importance(importance_type="gain")
    split_importance = booster.feature_importance(importance_type="split")

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance_gain": gain_importance,
        "importance_split": split_importance,
    })

    fig_gain = os.path.join(RESULTS_DIR, "lgbm_ranker_feature_importance_gain.png")
    fig_split = os.path.join(RESULTS_DIR, "lgbm_ranker_feature_importance_split.png")

    imp_gain_sorted = imp_df.sort_values("importance_gain", ascending=True)
    labels_gain = [FEATURE_LABEL.get(f, f) for f in imp_gain_sorted["feature"]]
    ypos_gain = np.arange(len(imp_gain_sorted))
    plt.figure(figsize=(10, max(6, 0.3 * len(imp_gain_sorted))))
    plt.barh(ypos_gain, imp_gain_sorted["importance_gain"], color="seagreen")
    plt.yticks(ypos_gain, labels_gain)
    plt.xlabel("Feature Importance (Gain)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(fig_gain, dpi=150)
    plt.close()
    print(f"Saved importance plot (gain): {fig_gain}")

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
    plt.savefig(fig_split, dpi=150)
    plt.close()
    print(f"Saved importance plot (split): {fig_split}")

    print("\n--- Runtime Summary ---")
    print(f"Training time: {train_time:.2f} s")
    print(f"Prediction time: {pred_time:.2f} s")
    print(f"Total time: {train_time + pred_time:.2f} s")


if __name__ == "__main__":
    main()
