#!/usr/bin/env python3
"""
Enhanced Hybrid GPU Performance Prediction Model

Combines physics-based analytical modeling with machine learning:
1. Roofline model + occupancy (analytical features)
2. Enhanced feature engineering (ratios, cache awareness, memory patterns)
3. Multiple ML models (Random Forest, XGBoost if available)
4. Log-transform for better scale handling
5. Comprehensive evaluation across all 3 experiments
"""

import json
import math
import itertools
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Try to import XGBoost for better performance
try:
    import xgboost as xgb
    HAS_XGBOOST = True
    print("✓ XGBoost available - will use for modeling")
except ImportError:
    HAS_XGBOOST = False
    print("✗ XGBoost not found - will use Random Forest (install with: pip install xgboost)")

# ============================================================
# CONFIG
# ============================================================

KERNEL_CSVS = [
    "../data/runs_2080ti_final.csv",
    "../data/runs_4070_final.csv",
    "../data/runs_titanv_final.csv",
    "../data/runs_titanx_final.csv",
]

GPU_JSON = "../data/gpu_metrics.json"

# Test GPU for Experiment 1 (new GPU generalization)
TEST_GPU_NAME = "NVIDIA TITAN V"

# Kernel splits for Experiment 3 (new kernel generalization)
TRAIN_KERNELS = [
    "vector_add",
    "saxpy",
    "strided_copy_8",
    "random_access",
    "reduce_sum",
    "dot_product",
    "histogram",
    "matmul_naive",
    "naive_transpose",
    "conv2d_3x3",
    "conv2d_7x7",
    "shared_bank_conflict",
]

TEST_KERNELS = [
    "matmul_tiled",
    "shared_transpose",
    "atomic_hotspot",
    "vector_add_divergent",
]

# Model configuration
USE_LOG_TRANSFORM = True  # Log-transform target for better scale handling
MODEL_TYPE = "xgboost" if HAS_XGBOOST else "random_forest"

# ============================================================
# Load Data
# ============================================================

print("\n" + "="*70)
print("LOADING DATA")
print("="*70)

dfs = []
for path in KERNEL_CSVS:
    print(f"Loading {path}...")
    df_part = pd.read_csv(path)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}")
print(f"GPUs: {df['gpu_device_name'].unique()}")
print(f"Kernels: {df['kernel'].unique()}")

# Normalize column names
if "device_name" not in df.columns and "gpu_device_name" in df.columns:
    df["device_name"] = df["gpu_device_name"]

# Extract block dimensions
if "block" in df.columns:
    df["bx"] = df["block"].astype(int)
    df["by"] = 1
    df["bz"] = 1
else:
    raise ValueError("Expected 'block' column in CSV")

# Ensure numeric columns
for col in ["FLOPs", "BYTES"]:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in CSV")
    df[col] = pd.to_numeric(df[col], errors="coerce")

if "mean_ms" not in df.columns:
    raise ValueError("Expected 'mean_ms' column in CSV")

# Fill missing columns
for col in ["regs", "shmem", "N", "rows", "cols", "iters", "working_set_bytes"]:
    if col not in df.columns:
        df[col] = 0

# If working_set_bytes missing, use BYTES
if "working_set_bytes" not in df.columns or df["working_set_bytes"].isna().all():
    df["working_set_bytes"] = df["BYTES"]

# Load GPU specifications
with open(GPU_JSON, "r") as f:
    gpu_list: List[Dict[str, Any]] = json.load(f)

gpu_by_name: Dict[str, Dict[str, Any]] = {g["device_name"]: g for g in gpu_list}

print(f"\nGPU specs loaded for: {list(gpu_by_name.keys())}")

# ============================================================
# Config ID and Roles
# ============================================================

def add_config_id(df_in: pd.DataFrame) -> pd.DataFrame:
    """Create unique config identifier for each kernel configuration"""
    df_out = df_in.copy()
    for col in ["N", "rows", "cols", "block", "iters"]:
        if col not in df_out.columns:
            df_out[col] = 0

    df_out["config_id"] = (
        df_out["kernel"].astype(str)
        + "|N=" + df_out["N"].fillna(0).astype(int).astype(str)
        + "|rows=" + df_out["rows"].fillna(0).astype(int).astype(str)
        + "|cols=" + df_out["cols"].fillna(0).astype(int).astype(str)
        + "|block=" + df_out["block"].fillna(0).astype(int).astype(str)
        + "|iters=" + df_out["iters"].fillna(0).astype(int).astype(str)
    )
    return df_out

df = add_config_id(df)

def compute_config_roles(df_with_cfg: pd.DataFrame) -> pd.DataFrame:
    """Assign roles to configs: baseline, train_extra, test_extra"""
    df = df_with_cfg.copy()
    size_col = "working_set_bytes"

    cfg = df.groupby(["kernel", "config_id"], as_index=False)[size_col].max()

    roles = []
    for kernel, sub in cfg.groupby("kernel"):
        sub = sub.sort_values(size_col).reset_index(drop=True)
        n = len(sub)

        for idx, row in sub.iterrows():
            if n == 1:
                role = "baseline"
            elif n == 2:
                role = "baseline" if idx == 0 else "test_extra"
            else:
                # Use ALL intermediate sizes for training
                if idx == 0:
                    role = "baseline"
                elif idx == n - 1:
                    role = "test_extra"
                else:
                    role = "train_extra"  # ALL middle configs
            roles.append({
                "kernel": kernel,
                "config_id": row["config_id"],
                "config_role": role,
            })

    return pd.DataFrame(roles).drop_duplicates()

roles_df = compute_config_roles(df)

# ============================================================
# Analytical Model Functions (Physics-Based Features)
# ============================================================

def compute_arithmetic_intensity(row):
    """FLOPs per byte transferred"""
    F = row["FLOPs"]
    B = row["BYTES"]
    if B <= 0:
        return 0.0
    return F / B

def roofline_bound_gflops(row, gpu_name: str):
    """Compute roofline performance ceiling"""
    I = row["arith_intensity"]
    if not np.isfinite(I) or I <= 0:
        return np.nan

    g = gpu_by_name[gpu_name]

    # Prefer sustained/calibrated over peak
    C_sust = (
        g.get("sustained_compute_gflops")
        or g.get("calibrated_compute_gflops")
        or g.get("peak_fp32_gflops")
    )
    BW_sust = (
        g.get("sustained_bandwidth_gbps")
        or g.get("calibrated_mem_bandwidth_gbps")
        or g.get("peak_mem_bandwidth_gbps")
    )

    if C_sust is None or BW_sust is None or C_sust <= 0 or BW_sust <= 0:
        return np.nan

    mem_bound = I * BW_sust  # GFLOPS if memory-bound
    return min(C_sust, mem_bound)

def compute_occupancy(row, gpu_name: str):
    """Compute theoretical occupancy based on resource limits"""
    g = gpu_by_name[gpu_name]

    regs_per_thread = row["regs"]
    shmem_per_block = row["shmem"]

    bx = int(row["bx"])
    by = int(row["by"])
    bz = int(row["bz"])
    threads_per_block = bx * by * bz
    if threads_per_block == 0:
        return 0.0

    warp_size = g["warp_size"]
    max_threads_per_sm = g["max_threads_per_sm"]
    max_blocks_per_sm = g["max_blocks_per_sm"]
    regs_per_sm = g["registers_per_sm"]
    shared_mem_per_sm = g["shared_mem_per_sm"]

    # Compute resource limits
    if regs_per_thread > 0:
        blocks_reg_limit = regs_per_sm // (regs_per_thread * threads_per_block)
        blocks_reg_limit = max(blocks_reg_limit, 0)
    else:
        blocks_reg_limit = max_blocks_per_sm

    if shmem_per_block > 0:
        blocks_smem_limit = shared_mem_per_sm // shmem_per_block
        blocks_smem_limit = max(blocks_smem_limit, 0)
    else:
        blocks_smem_limit = max_blocks_per_sm

    blocks_thread_limit = max_threads_per_sm // threads_per_block
    blocks_sm_limit = max_blocks_per_sm

    # Active blocks is minimum of all limits
    limits = [blocks_reg_limit, blocks_smem_limit, blocks_thread_limit, blocks_sm_limit]
    limits_pos = [x for x in limits if x > 0]
    active_blocks = min(limits_pos) if limits_pos else 0

    if active_blocks <= 0:
        return 0.0

    warps_per_block = math.ceil(threads_per_block / warp_size)
    max_warps_per_sm = max_threads_per_sm / warp_size
    active_warps = active_blocks * warps_per_block

    occ = active_warps / max_warps_per_sm
    return max(0.0, min(1.0, occ))

def measured_gflops(row):
    """Compute achieved GFLOPS from measurement"""
    F = row["FLOPs"]
    T_ms = row["mean_ms"]
    if T_ms <= 0 or F <= 0:
        return 0.0
    return F / (T_ms / 1000.0) / 1e9

def measured_bandwidth_gbps(row):
    """Compute achieved bandwidth from measurement"""
    B = row["BYTES"]
    T_ms = row["mean_ms"]
    if T_ms <= 0 or B <= 0:
        return 0.0
    return (B / 1e9) / (T_ms / 1000.0)

# Compute baseline analytical features
df["arith_intensity"] = df.apply(compute_arithmetic_intensity, axis=1)

# ============================================================
# Enhanced Feature Engineering
# ============================================================

def build_enhanced_pair_dataset(df_in: pd.DataFrame) -> pd.DataFrame:
    """Build cross-GPU pairs with comprehensive features"""

    print("\n" + "="*70)
    print("BUILDING ENHANCED FEATURE SET")
    print("="*70)

    rows = []
    for config_id, df_conf in df_in.groupby("config_id"):
        gpu_names = df_conf["device_name"].unique()
        if len(gpu_names) < 2:
            continue

        for src_name, tgt_name in itertools.permutations(gpu_names, 2):
            row_src = df_conf[df_conf["device_name"] == src_name].iloc[0]
            row_tgt = df_conf[df_conf["device_name"] == tgt_name].iloc[0]

            T_src = float(row_src["mean_ms"])
            T_tgt_true = float(row_tgt["mean_ms"])

            g_src = gpu_by_name.get(src_name, {})
            g_tgt = gpu_by_name.get(tgt_name, {})

            # === BASIC FEATURES ===
            features = {
                "kernel": row_src["kernel"],
                "config_id": config_id,
                "src_gpu": src_name,
                "tgt_gpu": tgt_name,
                "T_src_ms": T_src,
                "T_tgt_true_ms": T_tgt_true,

                # Kernel characteristics
                "FLOPs": float(row_src["FLOPs"]),
                "BYTES": float(row_src["BYTES"]),
                "arith_intensity": float(row_src["arith_intensity"]) if np.isfinite(row_src["arith_intensity"]) else 0.0,
                "regs": float(row_src.get("regs", 0.0)),
                "shmem": float(row_src.get("shmem", 0.0)),
                "block_size": float(row_src.get("block", 0.0)),
                "N": float(row_src.get("N", 0.0)),
                "rows": float(row_src.get("rows", 0.0)),
                "cols": float(row_src.get("cols", 0.0)),
                "iters": float(row_src.get("iters", 0.0)),
                "working_set_bytes": float(row_src.get("working_set_bytes", 0.0)),
            }

            # Memory pattern (categorical)
            mem_pattern = row_src.get("mem_pattern", "unknown")
            features["mem_pattern"] = mem_pattern

            # === PHYSICS-BASED FEATURES ===
            occ_src = compute_occupancy(row_src, src_name)
            occ_tgt = compute_occupancy(row_src, tgt_name)  # Same kernel config on tgt GPU

            features["occupancy_src"] = occ_src
            features["occupancy_tgt"] = occ_tgt
            features["occupancy_ratio"] = occ_tgt / occ_src if occ_src > 0 else 1.0

            # Roofline bounds
            roof_src = roofline_bound_gflops(row_src, src_name)
            roof_tgt = roofline_bound_gflops(row_src, tgt_name)

            features["roofline_src_gflops"] = roof_src if np.isfinite(roof_src) else 0.0
            features["roofline_tgt_gflops"] = roof_tgt if np.isfinite(roof_tgt) else 0.0
            features["roofline_ratio"] = (roof_tgt / roof_src) if (np.isfinite(roof_src) and roof_src > 0) else 1.0

            # Measured performance
            meas_gflops_src = measured_gflops(row_src)
            meas_bw_src = measured_bandwidth_gbps(row_src)

            features["measured_gflops_src"] = meas_gflops_src
            features["measured_bw_src_gbps"] = meas_bw_src

            # Efficiency (how close to roofline)
            if np.isfinite(roof_src) and roof_src > 0 and occ_src > 0:
                features["compute_efficiency_src"] = meas_gflops_src / (occ_src * roof_src)
            else:
                features["compute_efficiency_src"] = 0.0

            # === GPU SPECIFICATION FEATURES ===
            src_compute = g_src.get("calibrated_compute_gflops") or g_src.get("peak_fp32_gflops") or 1.0
            tgt_compute = g_tgt.get("calibrated_compute_gflops") or g_tgt.get("peak_fp32_gflops") or 1.0
            src_bw = g_src.get("calibrated_mem_bandwidth_gbps") or g_src.get("peak_mem_bandwidth_gbps") or 1.0
            tgt_bw = g_tgt.get("calibrated_mem_bandwidth_gbps") or g_tgt.get("peak_mem_bandwidth_gbps") or 1.0

            features["src_compute_gflops"] = src_compute
            features["tgt_compute_gflops"] = tgt_compute
            features["src_bandwidth_gbps"] = src_bw
            features["tgt_bandwidth_gbps"] = tgt_bw
            features["src_sm_count"] = g_src.get("sm_count", 1)
            features["tgt_sm_count"] = g_tgt.get("sm_count", 1)
            features["src_l2_cache_bytes"] = g_src.get("l2_cache_bytes", 0)
            features["tgt_l2_cache_bytes"] = g_tgt.get("l2_cache_bytes", 0)
            features["src_shared_mem_per_sm"] = g_src.get("shared_mem_per_sm", 0)
            features["tgt_shared_mem_per_sm"] = g_tgt.get("shared_mem_per_sm", 0)
            features["src_max_threads_per_sm"] = g_src.get("max_threads_per_sm", 1)
            features["tgt_max_threads_per_sm"] = g_tgt.get("max_threads_per_sm", 1)

            # === RATIO FEATURES (Key for cross-GPU prediction!) ===
            features["compute_ratio"] = tgt_compute / src_compute
            features["bandwidth_ratio"] = tgt_bw / src_bw
            features["sm_count_ratio"] = features["tgt_sm_count"] / features["src_sm_count"]
            features["l2_cache_ratio"] = (features["tgt_l2_cache_bytes"] / features["src_l2_cache_bytes"]) if features["src_l2_cache_bytes"] > 0 else 1.0

            # === CACHE AWARENESS ===
            ws = features["working_set_bytes"]
            features["working_set_per_l2_src"] = ws / features["src_l2_cache_bytes"] if features["src_l2_cache_bytes"] > 0 else 0.0
            features["working_set_per_l2_tgt"] = ws / features["tgt_l2_cache_bytes"] if features["tgt_l2_cache_bytes"] > 0 else 0.0

            # Cache residency (what fraction fits in L2)
            features["cache_residency_src"] = min(1.0, features["src_l2_cache_bytes"] / ws) if ws > 0 else 1.0
            features["cache_residency_tgt"] = min(1.0, features["tgt_l2_cache_bytes"] / ws) if ws > 0 else 1.0

            # === DERIVED FEATURES ===
            bx = int(row_src.get("bx", row_src.get("block", 256)))
            features["threads_per_block"] = bx
            features["warps_per_block"] = math.ceil(bx / 32.0)

            # Registers per thread pressure
            features["reg_pressure_src"] = features["regs"] / (features["src_max_threads_per_sm"] / features["threads_per_block"]) if features["threads_per_block"] > 0 else 0.0
            features["reg_pressure_tgt"] = features["regs"] / (features["tgt_max_threads_per_sm"] / features["threads_per_block"]) if features["threads_per_block"] > 0 else 0.0

            rows.append(features)

    pair_df = pd.DataFrame(rows)

    if pair_df.empty:
        print("WARNING: No cross-GPU pairs found!")
        return pair_df

    # Filter out invalid targets
    pair_df = pair_df[pair_df["T_tgt_true_ms"] > 0].copy()
    pair_df.reset_index(drop=True, inplace=True)

    # One-hot encode memory patterns
    if "mem_pattern" in pair_df.columns:
        pattern_dummies = pd.get_dummies(pair_df["mem_pattern"], prefix="pattern")
        pair_df = pd.concat([pair_df, pattern_dummies], axis=1)

    print(f"\nTotal cross-GPU pairs: {len(pair_df)}")
    print(f"Unique kernels: {pair_df['kernel'].nunique()}")
    print(f"Unique configs: {pair_df['config_id'].nunique()}")
    print(f"GPU pairs: {len(pair_df.groupby(['src_gpu', 'tgt_gpu']))}")

    return pair_df

pair_df = build_enhanced_pair_dataset(df)

# Merge config roles
pair_df = pair_df.merge(roles_df, on=["kernel", "config_id"], how="left")

# ============================================================
# Feature Selection
# ============================================================

# Define feature columns (exclude metadata and target)
EXCLUDE_COLS = [
    "kernel", "config_id", "src_gpu", "tgt_gpu",
    "T_src_ms", "T_tgt_true_ms", "config_role", "mem_pattern"
]

# Get all numeric columns as features
feature_cols = [col for col in pair_df.columns if col not in EXCLUDE_COLS and pair_df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]

print(f"\nFeature count: {len(feature_cols)}")
print("Sample features:", feature_cols[:10])

# ============================================================
# Model Training & Evaluation
# ============================================================

def make_feature_matrix(df_sub: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """Extract feature matrix and handle missing values"""
    X = df_sub[feature_cols].copy()
    X = X.fillna(0.0)
    X = X.replace([np.inf, -np.inf], 0.0)
    return X.values

def train_model(X_train: np.ndarray, y_train: np.ndarray, model_type: str):
    """Train regression model"""
    if model_type == "xgboost" and HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        print("Training XGBoost model...")
    else:
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        print("Training Random Forest model...")

    model.fit(X_train, y_train)
    return model

def evaluate_predictions(name: str, df_sub: pd.DataFrame, pred_col: str, save_prefix: str):
    """Comprehensive evaluation metrics"""
    df = df_sub.dropna(subset=[pred_col, "T_tgt_true_ms"]).copy()
    if df.empty:
        print(f"\n[{name}] No data after filtering.")
        return None

    true = df["T_tgt_true_ms"].values
    pred = df[pred_col].values

    # Metrics
    errors = np.abs(pred - true) / true
    ratios = pred / true

    mape = errors.mean() * 100.0
    med_ratio = np.median(ratios)
    mae = np.mean(np.abs(pred - true))
    rmse = math.sqrt(np.mean((pred - true) ** 2))

    within_10 = np.mean(errors < 0.10) * 100.0
    within_25 = np.mean(errors < 0.25) * 100.0
    within_50 = np.mean(errors < 0.50) * 100.0

    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    print(f"Pairs:             {len(df):6d}")
    print(f"MAPE:              {mape:6.2f}%")
    print(f"Median pred/true:  {med_ratio:6.3f}")
    print(f"MAE:               {mae:6.4f} ms")
    print(f"RMSE:              {rmse:6.4f} ms")
    print(f"Within 10% error:  {within_10:5.1f}%")
    print(f"Within 25% error:  {within_25:5.1f}%")
    print(f"Within 50% error:  {within_50:5.1f}%")

    # Per-kernel metrics
    kernel_metrics = []
    for kernel, g in df.groupby("kernel"):
        k_true = g["T_tgt_true_ms"].values
        k_pred = g[pred_col].values

        k_errors = np.abs(k_pred - k_true) / k_true
        k_mape = k_errors.mean() * 100.0
        k_max = k_errors.max() * 100.0
        k_med_ae = np.median(np.abs(k_pred - k_true))

        kernel_metrics.append({
            "kernel": kernel,
            "count": len(g),
            "MAPE_%": k_mape,
            "MAX_%": k_max,
            "MedAE_ms": k_med_ae,
        })

    kernel_df = pd.DataFrame(kernel_metrics).sort_values("MAPE_%", ascending=False)
    kernel_df.to_csv(f"{save_prefix}_kernel_metrics.csv", index=False)
    print(f"\nPer-kernel metrics saved to {save_prefix}_kernel_metrics.csv")

    # Save predictions
    df.to_csv(f"{save_prefix}_predictions.csv", index=False)
    print(f"Predictions saved to {save_prefix}_predictions.csv")

    return {
        "name": name,
        "pairs": len(df),
        "mape": mape,
        "median_ratio": med_ratio,
        "mae": mae,
        "rmse": rmse,
        "within_10": within_10,
        "within_25": within_25,
        "within_50": within_50,
    }

def run_experiment(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    model_type: str,
    use_log: bool,
    save_prefix: str,
):
    """Run full train/eval cycle"""

    if train_df.empty or test_df.empty:
        print(f"\n[{name}] Insufficient data: train={len(train_df)}, test={len(test_df)}")
        return None

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples:     {len(test_df)}")

    X_train = make_feature_matrix(train_df, feature_cols)
    y_train = train_df["T_tgt_true_ms"].values

    X_test = make_feature_matrix(test_df, feature_cols)
    y_test = test_df["T_tgt_true_ms"].values

    # Log transform if enabled
    if use_log:
        y_train = np.log1p(y_train)
        print("Using log-transform for target variable")

    # Train
    model = train_model(X_train, y_train, model_type)

    # Predict
    y_pred = model.predict(X_test)

    # Inverse transform if log was used
    if use_log:
        y_pred = np.expm1(y_pred)
        y_pred = np.maximum(y_pred, 0.0)  # Ensure non-negative

    # Evaluate
    test_df = test_df.copy()
    test_df["T_tgt_pred_ms"] = y_pred

    results = evaluate_predictions(name, test_df, "T_tgt_pred_ms", save_prefix)

    # Feature importance (top 20)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)

        print(f"\nTop 20 Features:")
        for idx, row in feat_imp.head(20).iterrows():
            print(f"  {row['feature']:35s} {row['importance']:.4f}")

        feat_imp.to_csv(f"{save_prefix}_feature_importance.csv", index=False)

    return results

# ============================================================
# EXPERIMENTS
# ============================================================

print("\n" + "="*70)
print("RUNNING EXPERIMENTS")
print("="*70)

all_results = []

# -------------------- EXPERIMENT 1 --------------------
# Same kernel, same config, NEW GPU
print("\n\n" + "#"*70)
print("# EXPERIMENT 1: Same kernel + config, NEW GPU")
print("#"*70)

test_mask_exp1 = (
    (pair_df["tgt_gpu"] == TEST_GPU_NAME) &
    (pair_df["src_gpu"] != TEST_GPU_NAME)
)
train_mask_exp1 = pair_df["tgt_gpu"] != TEST_GPU_NAME

result_exp1 = run_experiment(
    name=f"Exp1: New GPU ({TEST_GPU_NAME})",
    train_df=pair_df[train_mask_exp1],
    test_df=pair_df[test_mask_exp1],
    feature_cols=feature_cols,
    model_type=MODEL_TYPE,
    use_log=USE_LOG_TRANSFORM,
    save_prefix="exp1_new_gpu_hybrid",
)
if result_exp1:
    all_results.append(result_exp1)

# -------------------- EXPERIMENT 2 --------------------
# Same kernel, NEW configs, same GPUs
print("\n\n" + "#"*70)
print("# EXPERIMENT 2: Same kernel, NEW config")
print("#"*70)

train_mask_exp2 = pair_df["config_role"].isin(["baseline", "train_extra"])
test_mask_exp2 = pair_df["config_role"] == "test_extra"

result_exp2 = run_experiment(
    name="Exp2: New config",
    train_df=pair_df[train_mask_exp2],
    test_df=pair_df[test_mask_exp2],
    feature_cols=feature_cols,
    model_type=MODEL_TYPE,
    use_log=USE_LOG_TRANSFORM,
    save_prefix="exp2_new_config_hybrid",
)
if result_exp2:
    all_results.append(result_exp2)

# -------------------- EXPERIMENT 3 --------------------
# NEW but related kernels
print("\n\n" + "#"*70)
print("# EXPERIMENT 3: NEW kernels")
print("#"*70)

train_mask_exp3 = pair_df["kernel"].isin(TRAIN_KERNELS)
test_mask_exp3 = pair_df["kernel"].isin(TEST_KERNELS)

result_exp3 = run_experiment(
    name="Exp3: New kernels",
    train_df=pair_df[train_mask_exp3],
    test_df=pair_df[test_mask_exp3],
    feature_cols=feature_cols,
    model_type=MODEL_TYPE,
    use_log=USE_LOG_TRANSFORM,
    save_prefix="exp3_new_kernels_hybrid",
)
if result_exp3:
    all_results.append(result_exp3)

# ============================================================
# Summary
# ============================================================

print("\n\n" + "="*70)
print("FINAL SUMMARY - HYBRID MODEL")
print("="*70)
print(f"Model: {MODEL_TYPE.upper()}")
print(f"Log transform: {USE_LOG_TRANSFORM}")
print(f"Features: {len(feature_cols)}")
print("\n")

summary_df = pd.DataFrame(all_results)
if not summary_df.empty:
    print(summary_df.to_string(index=False))
    summary_df.to_csv("hybrid_model_summary.csv", index=False)
    print("\n✓ Summary saved to hybrid_model_summary.csv")

print("\n" + "="*70)
print("FILES GENERATED:")
print("="*70)
print("  exp1_new_gpu_hybrid_predictions.csv")
print("  exp1_new_gpu_hybrid_kernel_metrics.csv")
print("  exp1_new_gpu_hybrid_feature_importance.csv")
print("  exp2_new_config_hybrid_predictions.csv")
print("  exp2_new_config_hybrid_kernel_metrics.csv")
print("  exp2_new_config_hybrid_feature_importance.csv")
print("  exp3_new_kernels_hybrid_predictions.csv")
print("  exp3_new_kernels_hybrid_kernel_metrics.csv")
print("  exp3_new_kernels_hybrid_feature_importance.csv")
print("  hybrid_model_summary.csv")
print("="*70)
