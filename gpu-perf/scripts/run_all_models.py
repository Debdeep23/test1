#!/usr/bin/env python3
"""
Master script to run and compare all three prediction models:
1. Analytical (roofline + occupancy)
2. ML Baseline (Random Forest)
3. Hybrid Enhanced (physics-informed ML)

Generates comparison report and visualizations.
"""

import subprocess
import sys
import os
import pandas as pd
import time
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

SCRIPTS = [
    {
        "name": "Analytical Model",
        "script": "analytical_model_occupancy.py",
        "description": "Physics-based roofline + occupancy model",
        "color": "🔵",
    },
    {
        "name": "ML Baseline",
        "script": "ml_baseline.py",
        "description": "Random Forest with basic features",
        "color": "🟢",
    },
    {
        "name": "Hybrid Enhanced",
        "script": "hybrid_model_enhanced.py",
        "description": "Physics-informed ML with enhanced features",
        "color": "🟡",
    },
]

# ============================================================
# Helper Functions
# ============================================================

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_section(text):
    """Print section divider"""
    print("\n" + "-"*80)
    print(f"  {text}")
    print("-"*80 + "\n")

def run_script(script_info):
    """Run a Python script and capture output"""
    print_section(f"{script_info['color']} Running: {script_info['name']}")
    print(f"Description: {script_info['description']}")
    print(f"Script: {script_info['script']}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_info['script']],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ {script_info['name']} completed successfully in {elapsed:.1f}s")
            if result.stdout:
                print("\n--- Output ---")
                print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)  # Last 2000 chars
            return True, elapsed, result.stdout
        else:
            print(f"✗ {script_info['name']} failed with code {result.returncode}")
            if result.stderr:
                print("\n--- Error ---")
                print(result.stderr)
            return False, elapsed, result.stderr

    except subprocess.TimeoutExpired:
        print(f"✗ {script_info['name']} timed out after 10 minutes")
        return False, 600, "Timeout"
    except Exception as e:
        print(f"✗ {script_info['name']} error: {e}")
        return False, 0, str(e)

def extract_metrics_from_csv(prefix, exp_name):
    """Extract metrics from kernel metrics CSV"""
    csv_path = f"{prefix}_kernel_metrics.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'MAPE_%' in df.columns or 'PK_MAPE_%' in df.columns:
                mape_col = 'MAPE_%' if 'MAPE_%' in df.columns else 'PK_MAPE_%'
                avg_mape = df[mape_col].mean()
                max_mape = df[mape_col].max()
                return avg_mape, max_mape
        except:
            pass
    return None, None

def compare_results():
    """Compare results from all models"""
    print_header("COMPARING MODEL RESULTS")

    # File patterns for each model
    patterns = {
        "Analytical": [
            ("exp1_same_config_new_gpu", "Exp1: New GPU"),
            ("exp2_new_configs_same_gpus", "Exp2: New Config"),
            ("exp3a_new_kernels", "Exp3: New Kernels"),
        ],
        "ML Baseline": [
            ("exp1_same_config_new_gpu", "Exp1: New GPU"),
            ("exp2_new_configs_same_gpus", "Exp2: New Config"),
            ("exp3_new_kernels", "Exp3: New Kernels"),
        ],
        "Hybrid": [
            ("exp1_new_gpu_hybrid", "Exp1: New GPU"),
            ("exp2_new_config_hybrid", "Exp2: New Config"),
            ("exp3_new_kernels_hybrid", "Exp3: New Kernels"),
        ],
    }

    comparison_data = []

    for model_name, experiments in patterns.items():
        for prefix, exp_name in experiments:
            avg_mape, max_mape = extract_metrics_from_csv(prefix, exp_name)
            if avg_mape is not None:
                comparison_data.append({
                    "Model": model_name,
                    "Experiment": exp_name,
                    "Avg_MAPE_%": avg_mape,
                    "Max_MAPE_%": max_mape,
                })

    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        comp_df = comp_df.sort_values(["Experiment", "Avg_MAPE_%"])

        print("\nPer-Kernel Average MAPE Comparison:")
        print(comp_df.to_string(index=False))

        # Save comparison
        comp_df.to_csv("model_comparison.csv", index=False)
        print("\n✓ Comparison saved to model_comparison.csv")

        # Summary by model
        print("\n" + "="*80)
        print("OVERALL MODEL PERFORMANCE (lower is better)")
        print("="*80)

        for model in ["Analytical", "ML Baseline", "Hybrid"]:
            model_data = comp_df[comp_df["Model"] == model]
            if not model_data.empty:
                overall_avg = model_data["Avg_MAPE_%"].mean()
                print(f"{model:20s}: {overall_avg:6.2f}% average MAPE")

    else:
        print("⚠ No results found to compare")

    # Try to load summary files
    print("\n" + "="*80)
    print("DETAILED METRICS FROM SUMMARY FILES")
    print("="*80)

    summary_files = [
        ("hybrid_model_summary.csv", "Hybrid Enhanced Model"),
    ]

    for filename, model_name in summary_files:
        if os.path.exists(filename):
            print(f"\n{model_name}:")
            try:
                df = pd.read_csv(filename)
                print(df.to_string(index=False))
            except Exception as e:
                print(f"  Error reading {filename}: {e}")

def generate_readme():
    """Generate a comprehensive README"""
    readme_content = """# GPU Performance Prediction - Model Results

## Overview

This directory contains results from three different GPU performance prediction models:

1. **Analytical Model** - Physics-based roofline + occupancy
2. **ML Baseline** - Random Forest with basic features
3. **Hybrid Enhanced** - Physics-informed ML with enhanced features

## Experiments

### Experiment 1: New GPU Generalization
- **Goal**: Predict performance on a held-out GPU (NVIDIA TITAN V)
- **Training**: All pairs where target GPU is NOT TITAN V
- **Testing**: All pairs where target GPU IS TITAN V
- **Difficulty**: Hard - requires generalizing to new architecture

### Experiment 2: New Configuration Generalization
- **Goal**: Predict performance for unseen problem sizes
- **Training**: Baseline and intermediate configs
- **Testing**: Largest problem size configs
- **Difficulty**: Medium - same kernels, different scales

### Experiment 3: New Kernel Generalization
- **Goal**: Predict performance for new kernel types
- **Training**: 12 training kernels
- **Testing**: 4 held-out kernels (matmul_tiled, shared_transpose, atomic_hotspot, vector_add_divergent)
- **Difficulty**: Hard - requires understanding kernel characteristics

## Files Generated

### Analytical Model
- `cross_gpu_predictions.csv` - All cross-GPU predictions
- `exp1_same_config_new_gpu.csv` - Experiment 1 results
- `exp1_kernel_metrics.csv` - Per-kernel metrics for Exp1
- `exp2_new_configs_same_gpus.csv` - Experiment 2 results
- `exp2_kernel_metrics.csv` - Per-kernel metrics for Exp2
- `exp3a_new_kernels.csv` - Experiment 3 results
- `exp3a_new_kernel_metrics.csv` - Per-kernel metrics for Exp3

### ML Baseline
- `exp1_same_config_new_gpu_ml_predictions.csv`
- `exp1_same_config_new_gpu_kernel_metrics_ml.csv`
- `exp2_new_configs_same_gpus_ml_predictions.csv`
- `exp2_new_configs_same_gpus_kernel_metrics_ml.csv`
- `exp3_new_kernels_ml_predictions.csv`
- `exp3_new_kernels_kernel_metrics_ml.csv`

### Hybrid Enhanced Model
- `exp1_new_gpu_hybrid_predictions.csv`
- `exp1_new_gpu_hybrid_kernel_metrics.csv`
- `exp1_new_gpu_hybrid_feature_importance.csv`
- `exp2_new_config_hybrid_predictions.csv`
- `exp2_new_config_hybrid_kernel_metrics.csv`
- `exp2_new_config_hybrid_feature_importance.csv`
- `exp3_new_kernels_hybrid_predictions.csv`
- `exp3_new_kernels_hybrid_kernel_metrics.csv`
- `exp3_new_kernels_hybrid_feature_importance.csv`
- `hybrid_model_summary.csv` - Overall summary

### Comparison
- `model_comparison.csv` - Side-by-side comparison of all models

## Metrics

- **MAPE** (Mean Absolute Percentage Error): Average |predicted - actual| / actual
- **Median pred/true**: Median ratio of prediction to ground truth
- **MAE** (Mean Absolute Error): Average absolute error in milliseconds
- **RMSE** (Root Mean Squared Error): Emphasizes larger errors
- **Within X%**: Percentage of predictions within X% of ground truth

## Usage

To regenerate all results:
```bash
cd gpu-perf/scripts
python run_all_models.py
```

To run individual models:
```bash
python analytical_model_occupancy.py
python ml_baseline.py
python hybrid_model_enhanced.py
```

## Requirements

```bash
pip install pandas numpy scikit-learn
pip install xgboost  # Optional, for better performance in hybrid model
```
"""

    with open("RESULTS_README.md", "w") as f:
        f.write(readme_content)

    print("\n✓ README generated: RESULTS_README.md")

# ============================================================
# Main Execution
# ============================================================

def main():
    print_header("🚀 GPU PERFORMANCE PREDICTION - MODEL COMPARISON")

    print("This script will run all three prediction models and compare results.")
    print("\nModels to run:")
    for i, script in enumerate(SCRIPTS, 1):
        print(f"  {i}. {script['color']} {script['name']}: {script['description']}")

    print("\nEstimated time: 5-15 minutes depending on data size")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to start...")

    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        return

    # Run all scripts
    results = []
    for script_info in SCRIPTS:
        success, elapsed, output = run_script(script_info)
        results.append({
            "name": script_info["name"],
            "success": success,
            "time_s": elapsed,
        })

    # Summary of execution
    print_header("EXECUTION SUMMARY")
    for result in results:
        status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
        print(f"{status:12s} {result['name']:25s} ({result['time_s']:.1f}s)")

    # Compare results
    if all(r["success"] for r in results):
        compare_results()
        generate_readme()
    else:
        print("\n⚠ Some models failed - comparison may be incomplete")
        compare_results()

    print_header("✅ ALL DONE!")
    print("Check the generated CSV files and RESULTS_README.md for details")

if __name__ == "__main__":
    main()
