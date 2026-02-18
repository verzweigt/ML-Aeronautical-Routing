#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "simulation_metrics.csv")


def _try_set_style():
    """Apply a clean Matplotlib style with LaTeX/lmodern if available."""
    try:
        plt.style.use("default")
    except Exception:
        pass

    plt.rcParams.update({
        'font.family': 'lmodern',
        'font.size': 30,
        'text.usetex': True,
        'pgf.rcfonts': False,
        'savefig.dpi': 300,
        'text.latex.preamble': r'\usepackage{lmodern}',
    })
    # Make legends smaller globally
    plt.rcParams['legend.fontsize'] = 23


def _equipage_fraction(n: pd.Series, denom: float = 500.0) -> pd.Series:
    """Convert node count to equipage fraction using ``denom``."""
    return n.astype(float) / denom


def _ci95(std: pd.Series, n: pd.Series) -> pd.Series:
    """95% CI of the mean: ``1.96 * std / sqrt(n)`` (elementwise)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return 1.96 * (std.astype(float) / (n.astype(float).pow(0.5)))


def _load_data(csv_path: str) -> pd.DataFrame:
    """Load results CSV, harmonize column names and add plotting keys."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "PDR_mean" in df.columns:
        df = df.rename(columns={
            "PDR_mean": "SR_mean",
            "PDR_std": "SR_std",
        })

    # Add equipage fraction (x-axis) and group key
    df["equipage_frac"] = _equipage_fraction(df["N"])  # N/500
    df["key"] = df.apply(lambda r: ("Greedy" if r.get("method") == "Greedy" else f"ML:{r.get('model')}") , axis=1)
    return df


def _ordered_keys(df: pd.DataFrame) -> List[str]:
    """Preferred legend order with graceful fallback for any additional keys."""
    keys = []
    for k in ["Greedy", "ML:lgbm_rank", "ML:lgbm_cls", "ML:logreg"]:
        if (df["key"] == k).any():
            keys.append(k)
    # Append any remaining keys not in the preferred order
    for k in df["key"].unique():
        if k not in keys:
            keys.append(k)
    return keys


def _color_cycle(keys: List[str]) -> Dict[str, str]:
    """Return a stable color palette mapping for known keys, fallback otherwise."""
    # Fixed palette for consistency across plots
    palette = {
        "Greedy": "#1f77b4",       # blue
        "ML:lgbm_rank": "#2ca02c", # green
        "ML:lgbm_cls": "#d62728",  # red
        "ML:logreg": "#9467bd",    # purple
    }
    # Fallback to default colors if unknown
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    out = {}
    for i, k in enumerate(keys):
        out[k] = palette.get(k, default_colors[i % len(default_colors)] if default_colors else None)
    return out


def _label_for_key(key: str) -> str:
    mapping = {
        "Greedy": "GGR",
        "ML:lgbm_rank": "LightGBM Ranker",
        "ML:lgbm_cls": "LightGBM Classifier",
        "ML:rf": "Random Forest",
        "ML:logreg": "LogReg",
    }
    return mapping.get(key, key.replace("ML:", ""))


def _make_figure():
    """Create the base figure and axes used by all plots."""
    return plt.subplots(figsize=(12, 9), dpi=200)


def _apply_axis_style(ax, set_ylim: bool = False, ylim=(0.68, 1.015)):
    """Apply consistent grid and tick styling. Labels are set by callers."""
    # Y locators
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # X locator at 0.1 steps
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

    ax.xaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.xaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)
    ax.yaxis.grid(True, which='major', linestyle=(0, (5, 10)), linewidth=0.2)
    ax.yaxis.grid(True, which='minor', linestyle=(0, (5, 20)), linewidth=0.1)

    if set_ylim:
        ax.set_ylim(ylim)


def _nice_step(span: float) -> float:
    """Choose a reasonable tick step given a y-span."""
    if span <= 0 or not np.isfinite(span):
        return 0.1
    exp = np.floor(np.log10(span))
    base = 10 ** exp
    for m in [1, 2, 5, 10]:
        if span / (base * m) <= 6:
            return base * m / 5  # target around ~5 ticks
    return base


def _compute_ylim_for_errorbars(x: np.ndarray, y: np.ndarray, ci: np.ndarray,
                                lower_clip: float | None = None,
                                upper_clip: float | None = None,
                                pad: float = 0.005) -> tuple[float, float]:
    """Compute y-limits that include error bars with optional clipping and padding."""
    y_low = np.nanmin(y - ci)
    y_high = np.nanmax(y + ci)
    if lower_clip is not None:
        y_low = max(lower_clip, y_low)
    if upper_clip is not None:
        y_high = min(upper_clip, y_high)
    # add small margins
    span = y_high - y_low
    if not np.isfinite(span) or span <= 0:
        span = 0.1
    y_low -= pad
    y_high += pad
    return y_low, y_high



def plot_time_per_path(df: pd.DataFrame, out_path: str) -> None:
    """Plot time per path over equipage fraction for each method/model."""
    keys = _ordered_keys(df)
    # Map to provided color palette
    base_colors = {
        "Greedy": "#7A7A7A",          # gray
        "ML:lgbm_rank": "#2ca02c",    # green
        "ML:lgbm_cls": "#d62728",     # red
        "ML:logreg": "#298C8C",       # teal
    }
    colors = {k: base_colors.get(k, v) for k, v in _color_cycle(keys).items()}
    linestyles = ['dotted', 'dashed', 'solid', 'dashdot']

    fig, ax = _make_figure()

    for k in keys:
        d = df[df["key"] == k].sort_values("equipage_frac")
        ax.plot(
            d["equipage_frac"],
            d["Time_per_path_ms"],
            marker="o",
            markersize=5,
            linewidth=3,
            linestyle=linestyles[keys.index(k) % len(linestyles)],
            markeredgecolor='black',
            label=_label_for_key(k),
            color=colors.get(k),
        )

    ax.set_xlabel("Equipage Fraction")
    ax.set_ylabel("Computational Cost [ms]")
    _apply_axis_style(ax)
    ax.legend(loc='best', ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_hop_stretch(df: pd.DataFrame, out_path: str) -> None:
    """Plot mean hop stretch with 95% CI as error bars."""
    keys = _ordered_keys(df)
    base_colors = {
        "Greedy": "#7A7A7A",
        "ML:lgbm_rank": "#2ca02c",
        "ML:lgbm_cls": "#d62728",
        "ML:logreg": "#298C8C",
    }
    colors = {k: base_colors.get(k, v) for k, v in _color_cycle(keys).items()}
    linestyles = ['dotted', 'dashed', 'solid', 'dashdot']
    # Pre-compute dynamic y-limits across all series
    series = []
    for k in keys:
        d = df[df["key"] == k].sort_values("equipage_frac")
        y = d["HopStretch_mean"].to_numpy(dtype=float)
        ci = _ci95(d["HopStretch_std"], d["snapshots"]).to_numpy(dtype=float)
        x = d["equipage_frac"].to_numpy(dtype=float)
        series.append((k, x, y, ci))

    if series:
        all_y = np.concatenate([s[2] for s in series])
        all_ci = np.concatenate([s[3] for s in series])
        x_dummy = np.concatenate([s[1] for s in series])
        y_min, y_max = _compute_ylim_for_errorbars(x_dummy, all_y, all_ci,
                                                   lower_clip=None, upper_clip=None, pad=0.01)
    else:
        y_min, y_max = 0.95, 1.05

    fig, ax = _make_figure()
    for k, x, y, ci in series:
        # Asymmetric yerr clipped to computed limits
        y_lower = np.minimum(ci, y - y_min)
        y_upper = np.minimum(ci, y_max - y)
        ax.errorbar(
            x, y,
            yerr=[y_lower, y_upper],
            fmt="o",
            markersize=5,
            linewidth=3,
            elinewidth=1.5,
            capsize=4,
            capthick=1.2,
            linestyle=linestyles[keys.index(k) % len(linestyles)],
            markeredgecolor='black',
            label=_label_for_key(k),
            color=colors.get(k),
        )

    ax.set_xlabel("Equipage Fraction")
    ax.set_ylabel("Hop Stretch")
    ax.set_ylim(y_min, y_max)
    # Choose a clean y tick step
    step = _nice_step(y_max - y_min)
    if step > 0:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    _apply_axis_style(ax)
    ax.legend(loc='best', ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_success_ratio(df: pd.DataFrame, out_path: str) -> None:
    """Plot success ratio (mean) with 95% CI as error bars."""
    keys = _ordered_keys(df)
    base_colors = {
        "Greedy": "#7A7A7A",
        "ML:lgbm_rank": "#2ca02c",
        "ML:lgbm_cls": "#d62728",
        "ML:logreg": "#298C8C",
    }
    colors = {k: base_colors.get(k, v) for k, v in _color_cycle(keys).items()}
    linestyles = ['dotted', 'dashed', 'solid', 'dashdot']
    # Pre-compute dynamic y-limits across all series (clip to [0, 1.02])
    series = []
    for k in keys:
        d = df[df["key"] == k].sort_values("equipage_frac")
        y = d["SR_mean"].to_numpy(dtype=float)
        ci = _ci95(d["SR_std"], d["snapshots"]).to_numpy(dtype=float)
        x = d["equipage_frac"].to_numpy(dtype=float)
        series.append((k, x, y, ci))

    if series:
        all_y = np.concatenate([s[2] for s in series])
        all_ci = np.concatenate([s[3] for s in series])
        x_dummy = np.concatenate([s[1] for s in series])
        # No hard clipping: choose axes so that y±CI is fully visible
        y_min, y_max = _compute_ylim_for_errorbars(x_dummy, all_y, all_ci,
                                                   lower_clip=None, upper_clip=None, pad=0.01)
        # Ensure the span is not visually degenerate
        if y_max - y_min < 0.02:
            mid = 0.5 * (y_min + y_max)
            y_min = mid - 0.01
            y_max = mid + 0.01
    else:
        y_min, y_max = 0.9, 1.02

    fig, ax = _make_figure()
    for k, x, y, ci in series:
        y_lower = np.minimum(ci, y - y_min)
        y_upper = np.minimum(ci, y_max - y)
        ax.errorbar(
            x, y,
            yerr=[y_lower, y_upper],
            fmt="o",
            markersize=5,
            linewidth=3,
            elinewidth=1.5,
            capsize=4,
            capthick=1.2,
            linestyle=linestyles[keys.index(k) % len(linestyles)],
            markeredgecolor='black',
            label=_label_for_key(k),
            color=colors.get(k),
        )

    ax.set_xlabel("Equipage Fraction")
    ax.set_ylabel("Success Ratio")
    ax.set_ylim(y_min, y_max)
    step = _nice_step(y_max - y_min)
    if step > 0:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    _apply_axis_style(ax)
    # Success Ratio legend vertically stacked, bottom-right, smaller font via rcParams
    ax.legend(loc='lower right', ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    """Load CSV, generate all plots and save them under results/."""
    _try_set_style()
    df = _load_data(CSV_PATH)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    out_time = os.path.join(RESULTS_DIR, "plot_time_per_path.png")
    out_hs = os.path.join(RESULTS_DIR, "plot_hop_stretch.png")
    out_sr = os.path.join(RESULTS_DIR, "plot_success_ratio.png")

    plot_time_per_path(df, out_time)
    plot_hop_stretch(df, out_hs)
    plot_success_ratio(df, out_sr)

    print("Saved:")
    print(out_time)
    print(out_hs)
    print(out_sr)


if __name__ == "__main__":
    main()
