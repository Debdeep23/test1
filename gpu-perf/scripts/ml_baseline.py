#!/usr/bin/env python3
import json
import math
import itertools
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np

# --- ML imports ---
try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("ERROR: scikit-learn not found. Run `pip install scikit-learn` to use this script.")
    exit(1)

# ============================================================
# CONFIG: paths and GPU naming
# ============================================================

KERNEL_CSVS = [
    "runs_2080ti_final.csv",
    "runs_4070_final.csv",
    "runs_titanv_final.csv",
]

GPU_JSON = "gpu_metrics.json"

REF_GPU_NAME = "NVIDIA GeForce RTX 4070"
TEST_GPU_NAME = "NVIDIA TITAN V"


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

# numeric GPU fields we’ll use as features (if missing -> 0)
GPU_NUMERIC_FIELDS = [
    "peak_fp32_gflops",
    "sustained_compute_gflops",
    "calibrated_compute_gflops",
    "peak_mem_bandwidth_gbps",
    "sustained_bandwidth_gbps",
    "calibrated_mem_bandwidth_gbps",
    "sm_count",
    "max_threads_per_sm",
    "max_blocks_per_sm",
    "registers_per_sm",
    "shared_mem_per_sm",
    "warp_size",
]

# ============================================================
# Load & merge CSVs, normalize schema
# ============================================================

dfs = []
for path in KERNEL_CSVS:
    df_part = pd.read_csv(path)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

# 1) device_name
if "device_name" not in df.columns and "gpu_device_name" in df.columns:
    df["device_name"] = df["gpu_device_name"]

# 2) bx,by,bz from block
if "block" in df.columns:
    df["bx"] = df["block"].astype(int)
    df["by"] = 1
    df["bz"] = 1
else:
    raise ValueError("Expected 'block' column in CSV for threads-per-block info.")

# 3) FLOPs/BYTES numeric
for col in ["FLOPs", "BYTES"]:
    if col not in df.columns:
        raise ValueError(f"Expected column '{col}' in CSV.")
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 4) mean_ms
if "mean_ms" not in df.columns:
    raise ValueError("Expected 'mean_ms' column in CSV.")

# Load GPU JSON
with open(GPU_JSON, "r") as f:
    gpu_list: List[Dict[str, Any]] = json.load(f)

gpu_by_name: Dict[str, Dict[str, Any]] = {g["device_name"]: g for g in gpu_list}

# ============================================================
# Helper: add config_id
# ============================================================

def add_config_id(df_in: pd.DataFrame) -> pd.DataFrame:
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

# Fill missing numeric columns we’ll use later
for col in ["regs", "shmem", "N", "rows", "cols", "iters"]:
    if col not in df.columns:
        df[col] = 0

# ============================================================
# Config roles for Exp2 (baseline/train_extra/test_extra)
# ============================================================

def compute_config_roles(df_with_cfg: pd.DataFrame) -> pd.DataFrame:
    df = df_with_cfg.copy()

    size_col = "working_set_bytes" if "working_set_bytes" in df.columns else "BYTES"
    if size_col not in df.columns:
        raise ValueError("Expected 'working_set_bytes' or 'BYTES' to define config size.")

    cfg = (
        df.groupby(["kernel", "config_id"], as_index=False)[size_col]
        .max()
    )

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
                if idx == 0:
                    role = "baseline"
                elif idx == 1:
                    role = "train_extra"
                elif idx == n - 1:
                    role = "test_extra"
                else:
                    role = "other"
            roles.append(
                {
                    "kernel": kernel,
                    "config_id": row["config_id"],
                    "config_role": role,
                }
            )

    roles_df = pd.DataFrame(roles).drop_duplicates()
    return roles_df

roles_df = compute_config_roles(df)

# ============================================================
# Build pair dataset for ML (src->tgt pairs)
# ============================================================

def compute_arithmetic_intensity(row):
    F = row["FLOPs"]
    B = row["BYTES"]
    if B <= 0:
        return np.nan
    return F / B

df["arith_intensity"] = df.apply(compute_arithmetic_intensity, axis=1)

def build_pair_dataset(df_in: pd.DataFrame) -> pd.DataFrame:
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

            # GPU numeric features
            src_gpu_feats = {f"src_{k}": float(g_src.get(k, 0.0)) for k in GPU_NUMERIC_FIELDS}
            tgt_gpu_feats = {f"tgt_{k}": float(g_tgt.get(k, 0.0)) for k in GPU_NUMERIC_FIELDS}

            rows.append(
                {
                    "kernel": row_src["kernel"],
                    "config_id": config_id,
                    "src_gpu": src_name,
                    "tgt_gpu": tgt_name,
                    "T_src_ms": T_src,
                    "T_tgt_true_ms": T_tgt_true,
                    "FLOPs": float(row_src["FLOPs"]),
                    "BYTES": float(row_src["BYTES"]),
                    "arith_intensity": float(row_src["arith_intensity"])
                    if not math.isnan(row_src["arith_intensity"])
                    else 0.0,
                    "regs": float(row_src.get("regs", 0.0)),
                    "shmem": float(row_src.get("shmem", 0.0)),
                    "block": float(row_src.get("block", 0.0)),
                    "N": float(row_src.get("N", 0.0)),
                    "rows": float(row_src.get("rows", 0.0)),
                    "cols": float(row_src.get("cols", 0.0)),
                    "iters": float(row_src.get("iters", 0.0)),
                    **src_gpu_feats,
                    **tgt_gpu_feats,
                }
            )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        print("WARNING: no cross-GPU pairs found for ML.")
        return pair_df

    # safety: drop rows where target <= 0
    pair_df = pair_df[pair_df["T_tgt_true_ms"] > 0].copy()
    pair_df.reset_index(drop=True, inplace=True)
    return pair_df

pair_df = build_pair_dataset(df)

# merge roles for Exp2
pair_df = pair_df.merge(
    roles_df,
    on=["kernel", "config_id"],
    how="left",
)

# ============================================================
# Features + metrics
# ============================================================

FEATURE_COLS = [
    "FLOPs", "BYTES", "arith_intensity",
    "regs", "shmem", "block",
    "N", "rows", "cols", "iters",
    "T_src_ms",
]

# add GPU feature cols
for k in GPU_NUMERIC_FIELDS:
    FEATURE_COLS.append(f"src_{k}")
    FEATURE_COLS.append(f"tgt_{k}")

def make_feature_matrix(df_sub: pd.DataFrame) -> np.ndarray:
    X = df_sub[FEATURE_COLS].copy()
    X = X.fillna(0.0)
    return X.values

def summarize_experiment(name: str, df_sub: pd.DataFrame, pred_col: str):
    df = df_sub.dropna(subset=[pred_col, "T_tgt_true_ms"]).copy()
    if df.empty:
        print(f"\n[{name}] No data after filtering.")
        return
    true = df["T_tgt_true_ms"].values
    pred = df[pred_col].values

    errors = np.abs(pred - true) / true
    ratios = pred / true

    mape = errors.mean() * 100.0
    med_ratio = np.median(ratios)
    mae = np.mean(np.abs(pred - true))
    rmse = math.sqrt(np.mean((pred - true) ** 2))

    within_10 = np.mean(errors < 0.10) * 100.0
    within_25 = np.mean(errors < 0.25) * 100.0
    within_50 = np.mean(errors < 0.50) * 100.0

    print(f"\n=== {name} ===")
    print("Num pairs:", len(df))
    print(f"MAPE:              {mape:6.2f}%")
    print(f"Median pred/true:  {med_ratio:6.3f}")
    print(f"MAE:               {mae:6.4f} ms")
    print(f"RMSE:              {rmse:6.4f} ms")
    print(f"Within 10% error:  {within_10:5.1f}%")
    print(f"Within 25% error:  {within_25:5.1f}%")
    print(f"Within 50% error:  {within_50:5.1f}%")

def kernel_generalization_metrics(df_sub: pd.DataFrame, pred_col: str, csv_path: str, label: str):
    df = df_sub.dropna(subset=[pred_col, "T_tgt_true_ms"]).copy()
    if df.empty:
        print(f"[{label}] No data for kernel-level metrics.")
        return

    rows = []
    for kernel, g in df.groupby("kernel"):
        true = g["T_tgt_true_ms"].values
        pred = g[pred_col].values
        if len(true) == 0:
            continue

        rel_errors = np.abs(pred - true) / true
        pk_mape = rel_errors.mean() * 100.0
        pk_max = rel_errors.max() * 100.0
        pk_med_ae = np.median(np.abs(pred - true))

        ss_res = np.sum((pred - true) ** 2)
        ss_tot = np.sum((true - true.mean()) ** 2)
        pk_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append(
            {
                "kernel": kernel,
                "count": len(g),
                "PK_MAPE_%": pk_mape,
                "PK_MAX_%": pk_max,
                "PK_MedAE_ms": pk_med_ae,
                "PK_R2": pk_r2,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("PK_MAPE_%", ascending=False)
    metrics_df.to_csv(csv_path, index=False)
    print(f"[{label}] Saved kernel-level metrics to {csv_path}")

# ============================================================
# Training helper
# ============================================================

def train_and_eval_rf(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    pred_col: str,
    metrics_prefix: str,
):
    if train_df.empty or test_df.empty:
        print(f"\n[{name}] Not enough data: train={len(train_df)}, test={len(test_df)}.")
        return

    X_train = make_feature_matrix(train_df)
    y_train = train_df["T_tgt_true_ms"].values

    X_test = make_feature_matrix(test_df)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_df = test_df.copy()
    test_df[pred_col] = y_pred

    # print metrics
    summarize_experiment(name, test_df, pred_col=pred_col)

    # kernel-level
    kernel_generalization_metrics(
        test_df,
        pred_col=pred_col,
        csv_path=f"{metrics_prefix}_kernel_metrics_ml.csv",
        label=name,
    )

    # save predictions
    test_df.to_csv(f"{metrics_prefix}_ml_predictions.csv", index=False)
    print(f"[{name}] Saved predictions to {metrics_prefix}_ml_predictions.csv")

# ============================================================
# EXP 1 / EXP 2 / EXP 3 splits
# ============================================================

def run_exp1_ml(pair_df: pd.DataFrame):
    """
    Exp1: Same kernel, same config, new GPU.

    Analytic design:
      - "New GPU" is the held-out TARGET GPU (TEST_GPU_NAME).
      - We evaluate only pairs where:
            tgt_gpu == TEST_GPU_NAME
            src_gpu != TEST_GPU_NAME

    ML version:
      - Train on all pairs where target is NOT the held-out GPU.
      - Test on pairs where target IS the held-out GPU (and src != tgt).
    """
    # test: predict performance on the new GPU
    test_mask = (
        (pair_df["tgt_gpu"] == TEST_GPU_NAME) &
        (pair_df["src_gpu"] != TEST_GPU_NAME)
    )

    # train: any pair whose target is NOT the new GPU
    train_mask = pair_df["tgt_gpu"] != TEST_GPU_NAME

    train_df = pair_df[train_mask].copy()
    test_df  = pair_df[test_mask].copy()

    train_and_eval_rf(
        name=f"Exp1 ML: same kernel+config, new GPU (tgt={TEST_GPU_NAME})",
        train_df=train_df,
        test_df=test_df,
        pred_col="T_tgt_pred_ms_ml",
        metrics_prefix="exp1_same_config_new_gpu",
    )

def run_exp2_ml(pair_df: pd.DataFrame):
    """
    Exp2: Same kernel, NEW configs, same GPUs.
    - Train on configs with role baseline/train_extra
    - Test on configs with role test_extra
    """
    train_mask = pair_df["config_role"].isin(["baseline", "train_extra"])
    test_mask = pair_df["config_role"] == "test_extra"

    train_df = pair_df[train_mask].copy()
    test_df = pair_df[test_mask].copy()

    train_and_eval_rf(
        name="Exp2 ML: same kernel, new config",
        train_df=train_df,
        test_df=test_df,
        pred_col="T_tgt_pred_ms_ml",
        metrics_prefix="exp2_new_configs_same_gpus",
    )

def run_exp3_ml(pair_df: pd.DataFrame):
    """
    Exp3: New but related kernels.
    - Train on TRAIN_KERNELS
    - Test on TEST_KERNELS (new kernels)
    """
    train_mask = pair_df["kernel"].isin(TRAIN_KERNELS)
    test_mask = pair_df["kernel"].isin(TEST_KERNELS)

    train_df = pair_df[train_mask].copy()
    test_df = pair_df[test_mask].copy()

    train_and_eval_rf(
        name="Exp3 ML: new but related kernels",
        train_df=train_df,
        test_df=test_df,
        pred_col="T_tgt_pred_ms_ml",
        metrics_prefix="exp3_new_kernels",
    )

# ============================================================
# Main
# ============================================================

def main():
    if pair_df.empty:
        print("No data for ML experiments.")
        return

    print("Total ML pair dataset rows:", len(pair_df))

    run_exp1_ml(pair_df)
    run_exp2_ml(pair_df)
    run_exp3_ml(pair_df)

    print("\nDone. Generated ML outputs:")
    print("  - exp1_same_config_new_gpu_ml_predictions.csv")
    print("  - exp1_same_config_new_gpu_kernel_metrics_ml.csv")
    print("  - exp2_new_configs_same_gpus_ml_predictions.csv")
    print("  - exp2_new_configs_same_gpus_kernel_metrics_ml.csv")
    print("  - exp3_new_kernels_ml_predictions.csv")
    print("  - exp3_new_kernels_kernel_metrics_ml.csv")

if __name__ == "__main__":
    main()