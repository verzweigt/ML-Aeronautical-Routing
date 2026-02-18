#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation of ML-based routing and a GGR on random snapshots of an
airborne ad-hoc network. The script generates per-density metrics, logs and a
CSV summary.
"""

# ========= Imports =========
import os                              # paths/directories
import math                            # trigonometric functions
import numpy as np                     # numerics, RNG
import time                            # timing
import pandas as pd                    # CSV output
import networkx as nx                  # graphs, path algorithms
import joblib                          # load trained models
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# ========= Configuration =========
# --- Geometry / ranges ---
A2A_RANGE_KM  = 100.0                  # air-to-air range
A2G_RANGE_KM  = 370.4                  # air-to-ground range
AREA_X, AREA_Y = 1250, 800             # simulation rectangle size
GS_POS = (1150, 400)                   # ground station position (x, y)

# --- Densities and snapshots (with distinct seeds) ---
DENSITIES = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # number of aircraft (GS added separately)
SIM_SNAPSHOTS = 500                         # snapshots per density (tune for runtime)
SEED_BASE = 100000                          # seed offset to avoid overlap with training


_BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(_BASE_DIR, "results")
OUT_CSV = os.path.join(RESULTS_DIR, "simulation_metrics.csv")

# Models:
_MODEL_DIR = os.path.normpath(os.path.join(_BASE_DIR, "..", "Modeltraining", "results"))
MODELS = [
    ("lgbm_rank", os.path.join(_MODEL_DIR, "lgbm_ranker_model.joblib")),
    ("lgbm_cls",  os.path.join(_MODEL_DIR, "lgbm_model.joblib")),
    # ("rf", os.path.join(_MODEL_DIR, "rf_model.joblib")),
    ("logreg",    os.path.join(_MODEL_DIR, "logreg_model.joblib")),
]
LOG_PREFIX = os.path.join(RESULTS_DIR, "simulation_log_")


def _scale_minmax_value(val: float, vmin: float, vmax: float) -> float:
    """Clamp and scale a value to [0, 1] based on [vmin, vmax]."""
    if vmax <= vmin:
        return 0.0
    if val < vmin:
        val = vmin
    elif val > vmax:
        val = vmax
    return (val - vmin) / (vmax - vmin)

# --- Routing guard ---
MAX_HOPS_FACTOR = 3.0   # OUTDATED


# ========= Network construction =========
def create_network(num_nodes: int, rng: np.random.RandomState):
    """Create a network snapshot.

    - Place ``num_nodes`` aircraft uniformly in the rectangle.
    - Add a ground station (node id = ``num_nodes``) at ``GS_POS``.
    - Add A2A and A2G links based on ranges.

    Returns a tuple ``(G, pos, gs_id)``.
    """
    gs_id = num_nodes
    pos = {i: (rng.uniform(0, AREA_X), rng.uniform(0, AREA_Y)) for i in range(num_nodes)}
    pos[gs_id] = GS_POS

    G = nx.Graph()
    G.add_nodes_from(pos)

    # A2A edges
    for i in range(num_nodes):
        xi, yi = pos[i]
        for j in range(i + 1, num_nodes):
            xj, yj = pos[j]
            if math.hypot(xi - xj, yi - yj) <= A2A_RANGE_KM:
                G.add_edge(i, j)

    # A2G edges
    for i in range(num_nodes):
        xi, yi = pos[i]
        if math.hypot(xi - GS_POS[0], yi - GS_POS[1]) <= A2G_RANGE_KM:
            G.add_edge(i, gs_id)

    return G, pos, gs_id


# ========= Feature computation for candidate (prev, x, y) =========
def features_for_candidate(prev, x, y, pos, gs_id, G, bc_cache, cc_cache):
    """Compute the input features as used during training.

    Feature set includes: ``dist_y, advance, cosang, angle_yd, deg_y, bc_y, cc_y, turn_ang, dist_xy``.
    ``prev`` can be ``None`` (first hop), in which case ``turn_ang = 0.0``.
    """
    GSx, GSy = pos[gs_id]
    px, py = pos[x], pos[y]

    # Distances x->GS and y->GS
    dist_x = math.hypot(px[0] - GSx, px[1] - GSy)
    dist_y = math.hypot(py[0] - GSx, py[1] - GSy)

    # Progress (positive if y is closer to GS than x)
    advance = dist_x - dist_y

    # Direction vectors
    v1 = (GSx - px[0], GSy - px[1])         # x -> GS
    v2 = (py[0] - px[0], py[1] - px[1])     # x -> y

    # Cosine of angle between v1 and v2 (stabilized)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cosang = float(np.dot(v1, v2) / denom)

    # Absolute angle of y towards GS
    angle_yd = math.atan2(GSy - py[1], GSx - py[0])

    # Turn angle (prev->x vs. x->y)
    if prev is not None:
        pp = pos[prev]
        v_prev = (px[0] - pp[0], px[1] - pp[1])           # prev -> x
        bearing_px = math.atan2(v_prev[1], v_prev[0])
        bearing_xy = math.atan2(v2[1], v2[0])             # x -> y
        turn_ang = ((bearing_xy - bearing_px + math.pi) % (2 * math.pi)) - math.pi
    else:
        turn_ang = 0.0

    # Topology features of y
    deg_y = G.degree(y)
    # Betweenness and clustering are precomputed per snapshot and fetched from cache
    bc_y = bc_cache[y]
    cc_y = cc_cache[y]

    # Distance x->y (optional, depending on model)
    dist_xy = math.hypot(py[0] - px[0], py[1] - px[1])

    # Manual normalization to match training
    advance  = _scale_minmax_value(advance,   -100.0, 100.0)
    cosang   = _scale_minmax_value(cosang,    -1.0,     1.0)
    angle_yd = _scale_minmax_value(angle_yd,  -math.pi, math.pi)
    turn_ang = _scale_minmax_value(turn_ang,  -math.pi, math.pi)
    dist_xy  = _scale_minmax_value(dist_xy,     0.0,   100.0)

    # Return feature vector as a dict
    return {
        "dist_y": dist_y,
        "advance": advance,
        "cosang": cosang,
        "angle_yd": angle_yd,
        "deg_y": deg_y,
        "bc_y": bc_y,
        "cc_y": cc_y,
        "turn_ang": turn_ang,
        "dist_xy": dist_xy
    }


# ========= Model scoring (ranker/classifier) =========
def _scores_for_candidates(model, X: np.ndarray) -> np.ndarray:
    """Return a score per candidate row.

    - Classifier: probability of class 1 via ``predict_proba``.
    - Ranker/regressor: direct scores via ``predict``.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass
    return np.asarray(model.predict(X)).ravel()


# ========= Next hop via ML model =========
def choose_next_hop(model, x, prev, neighbors, pos, gs_id, G, bc_cache, cc_cache, feature_order):
    """Choose neighbor ``y`` with maximal score P(label=1 | features).

    ``feature_order`` typically comes from ``model.feature_names_in_``.
    """
    if not neighbors:
        return None

    rows = []
    ys = []
    for y in neighbors:
        feats = features_for_candidate(prev, x, y, pos, gs_id, G, bc_cache, cc_cache)
        ys.append(y)
        # Ensure training-compatible feature order/subset
        row = [feats[f] for f in feature_order]
        rows.append(row)

    X = np.array(rows, dtype=float)
    scores = _scores_for_candidates(model, X)  # works for ranker & classifier
    best_idx = int(np.argmax(scores))
    return ys[best_idx]


# ========= Routing simulation for a single snapshot =========
def simulate_snapshot(model, G, pos, gs_id):
    """Simulate one packet from every source to the GS.

    - Forward using ML next-hop selection (argmax score).
    - Metrics: delivered, attempted, hop-stretches (successful paths only).

    Returns: ``delivered, attempted, hop_stretches``.
    """
    # Precompute centralities once per snapshot
    bc = nx.betweenness_centrality(G, normalized=True)
    cc = nx.clustering(G)

    # Feature order for model input (robust against column order)
    if hasattr(model, "feature_names_in_"):
        feature_order = list(model.feature_names_in_)
    else:
        # Fallback order if model does not expose names
        feature_order = [
            "dist_xy",
            "advance",
            "cosang",
            "angle_yd",
            "deg_y",
            "bc_y",
            "cc_y",
            "turn_ang",
        ]

    sources = [n for n in G.nodes if n != gs_id]
    attempted = 0
    delivered = 0
    hop_stretches = []

    # TTL limit
    max_hops = int(MAX_HOPS_FACTOR * (len(G.nodes)))

    for s in sources:
        # Count only sources that have a path to GS (for optimal hop baseline)
        try:
            optimal_hops = nx.shortest_path_length(G, source=s, target=gs_id)  # optimal hop count
        except nx.NetworkXNoPath:
            continue  # no path -> not attempted

        attempted += 1

        visited = set([s])
        prev = None
        x = s
        hops_used = 0
        success = False

        while hops_used < max_hops:
            if x == gs_id:
                success = True
                break

            # Direct A2G: if current node is within A2G range, deliver to GS next
            GSx, GSy = pos[gs_id]
            px = pos[x]
            if math.hypot(px[0] - GSx, px[1] - GSy) <= A2G_RANGE_KM:
                hops_used += 1
                success = True
                break

            neigh = list(G.neighbors(x))
            # No neighbor -> dead end
            if not neigh:
                break

            # Choose next hop via ML
            y = choose_next_hop(model, x, prev, neigh, pos, gs_id, G, bc, cc, feature_order)
            if y is None:
                break

            # Loop detection
            if y in visited:
                break

            # Advance
            prev, x = x, y
            visited.add(x)
            hops_used += 1

        if success:
            delivered += 1
            # Hop stretch = used hops / optimal hops
            hop_stretches.append(hops_used / optimal_hops)

    return delivered, attempted, hop_stretches


def choose_next_hop_greedy(x, neighbors, pos, gs_id):
    """Choose the neighbor with the smallest distance to the GS."""
    if not neighbors:
        return None
    GSx, GSy = pos[gs_id]
    px = pos[x]
    dist_x = math.hypot(px[0] - GSx, px[1] - GSy)

    best_y = None
    best_dist = dist_x  # must be strictly smaller to make progress
    for y in neighbors:
        py = pos[y]
        dist_y = math.hypot(py[0] - GSx, py[1] - GSy)
        if dist_y < best_dist:
            best_dist = dist_y
            best_y = y
    return best_y  # None if no closer neighbor exists


def simulate_snapshot_greedy(G, pos, gs_id):
    """Greedy routing baseline on one snapshot.

    Chooses the neighbor strictly closer to GS at each step. Metrics are
    computed analogously to the ML simulation.
    """
    sources = [n for n in G.nodes if n != gs_id]
    attempted = 0
    delivered = 0
    hop_stretches = []

    max_hops = int(MAX_HOPS_FACTOR * (len(G.nodes)))

    for s in sources:
        try:
            optimal_hops = nx.shortest_path_length(G, source=s, target=gs_id)
        except nx.NetworkXNoPath:
            continue

        attempted += 1

        visited = set([s])
        x = s
        hops_used = 0
        success = False

        while hops_used < max_hops:
            if x == gs_id:
                success = True
                break

            neigh = list(G.neighbors(x))
            if not neigh:
                break

            # Direct A2G: if current node is within A2G range, deliver to GS next
            GSx, GSy = pos[gs_id]
            px = pos[x]
            if math.hypot(px[0] - GSx, px[1] - GSy) <= A2G_RANGE_KM:
                hops_used += 1
                success = True
                break

            y = choose_next_hop_greedy(x, neigh, pos, gs_id)
            if y is None:
                break

            if y in visited:
                break

            x = y
            visited.add(x)
            hops_used += 1

        if success:
            delivered += 1
            hop_stretches.append(hops_used / optimal_hops)

    return delivered, attempted, hop_stretches


# ========= Main program (model + greedy) =========
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def log(msg: str, fh):
        print(msg)
        if fh is not None:
            fh.write(msg + "\n")
            fh.flush()

    all_records = []

    # 1) Simulate ML models one by one
    for model_name, model_path in MODELS:
        log_path = f"{LOG_PREFIX}{model_name}.txt"
        with open(log_path, "w", encoding="utf-8") as lf:
            log(f"Loading model: {model_path}", lf)
            model = joblib.load(model_path)

            for N in DENSITIES:
                SR_list = []
                stretch_list = []
                total_time_sec = 0.0
                total_attempts = 0
                for k in range(SIM_SNAPSHOTS):
                    seed = SEED_BASE + (N * 10_000) + k
                    rng = np.random.RandomState(seed)

                    G, pos, gs_id = create_network(N, rng)

                    t0 = time.time()
                    delivered, attempted, hop_stretches = simulate_snapshot(model, G, pos, gs_id)
                    dt = time.time() - t0
                    total_time_sec += dt
                    total_attempts += attempted

                    SR = (delivered / attempted) if attempted > 0 else np.nan
                    SR_list.append(SR)

                    if len(hop_stretches) > 0:
                        stretch_mean = float(np.mean(hop_stretches))
                    else:
                        stretch_mean = np.nan
                    stretch_list.append(stretch_mean)

                    log(f"[ML:{model_name}] N={N:4d} | snap {k+1:03d}/{SIM_SNAPSHOTS} | attempted={attempted:4d} delivered={delivered:4d} | SR={SR:.3f} | stretch_mean={stretch_mean:.3f} | time={dt*1000:.1f} ms", lf)

                SR_mean  = np.nanmean(SR_list) if len(SR_list) else np.nan
                SR_std   = np.nanstd(SR_list)  if len(SR_list) else np.nan
                str_mean  = np.nanmean(stretch_list) if len(stretch_list) else np.nan
                str_std   = np.nanstd(stretch_list)  if len(stretch_list) else np.nan

                time_total = total_time_sec
                time_per_path_ms = (time_total / total_attempts * 1000.0) if total_attempts > 0 else np.nan

                all_records.append({
                    "method": "ML",
                    "model": model_name,
                    "N": N,
                    "snapshots": SIM_SNAPSHOTS,
                    "SR_mean": SR_mean,
                    "SR_std": SR_std,
                    "HopStretch_mean": str_mean,
                    "HopStretch_std": str_std,
                    "Time_total_sec": time_total,
                    "Time_per_path_ms": time_per_path_ms
                })

            # Model summary in log
            df_model = pd.DataFrame([r for r in all_records if r.get("method") == "ML" and r.get("model") == model_name]).sort_values("N")
            log("\n===== Summary per density (ML:" + model_name + ") =====", lf)
            log(df_model.to_string(index=False), lf)
            log("", lf)

    # 2) Simulate greedy routing separately
    greedy_log_path = f"{LOG_PREFIX}greedy.txt"
    with open(greedy_log_path, "w", encoding="utf-8") as lf:
        for N in DENSITIES:
            SR_list = []
            stretch_list = []
            total_time_sec = 0.0
            total_attempts = 0
            for k in range(SIM_SNAPSHOTS):
                seed = SEED_BASE + (N * 10_000) + k
                rng = np.random.RandomState(seed)
                G, pos, gs_id = create_network(N, rng)

                t0 = time.time()
                delivered, attempted, hop_stretches = simulate_snapshot_greedy(G, pos, gs_id)
                dt = time.time() - t0
                total_time_sec += dt
                total_attempts += attempted

                SR = (delivered / attempted) if attempted > 0 else np.nan
                SR_list.append(SR)

                if len(hop_stretches) > 0:
                    stretch_mean = float(np.mean(hop_stretches))
                else:
                    stretch_mean = np.nan
                stretch_list.append(stretch_mean)

                log(f"[Greedy] N={N:4d} | snap {k+1:03d}/{SIM_SNAPSHOTS} | attempted={attempted:4d} delivered={delivered:4d} | SR={SR:.3f} | stretch_mean={stretch_mean:.3f} | time={dt*1000:.1f} ms", lf)

            SR_mean  = np.nanmean(SR_list) if len(SR_list) else np.nan
            SR_std   = np.nanstd(SR_list)  if len(SR_list) else np.nan
            str_mean  = np.nanmean(stretch_list) if len(stretch_list) else np.nan
            str_std   = np.nanstd(stretch_list)  if len(stretch_list) else np.nan

            time_total = total_time_sec
            time_per_path_ms = (time_total / total_attempts * 1000.0) if total_attempts > 0 else np.nan

            all_records.append({
                "method": "Greedy",
                "model": "greedy",
                "N": N,
                "snapshots": SIM_SNAPSHOTS,
                "SR_mean": SR_mean,
                "SR_std": SR_std,
                "HopStretch_mean": str_mean,
                "HopStretch_std": str_std,
                "Time_total_sec": time_total,
                "Time_per_path_ms": time_per_path_ms
            })

        df_greedy = pd.DataFrame([r for r in all_records if r.get("method") == "Greedy"]).sort_values("N")
        log("\n===== Summary per density (Greedy) =====", lf)
        log(df_greedy.to_string(index=False), lf)
        log("", lf)

    # 3) Write all results to a CSV
    df_all = pd.DataFrame(all_records).sort_values(["method", "model", "N"]).reset_index(drop=True)
    df_all.to_csv(OUT_CSV, index=False)
    print("\n===== Overall summary (all methods/models) =====")
    print(df_all.to_string(index=False))
    print(f"\nResults written to: {OUT_CSV}")
