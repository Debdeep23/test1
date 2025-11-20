# GPU Performance Prediction Models

## 🎯 Project Overview

**Research Question**: Can we predict kernel performance on a new GPU based on measurements from other GPUs?

This project implements and compares **three different approaches** for cross-GPU performance prediction:

1. **Analytical Model** - Physics-based roofline + occupancy model
2. **ML Baseline** - Random Forest with basic features
3. **Hybrid Enhanced** - Physics-informed ML with advanced features (⭐ **RECOMMENDED**)

## 📊 Dataset

You have collected performance data from **4 GPUs**:
- NVIDIA GeForce RTX 2080 Ti (Turing, 2018)
- NVIDIA GeForce RTX 4070 (Ada Lovelace, 2023)
- NVIDIA TITAN V (Volta, 2017)
- NVIDIA GeForce GTX TITAN X (Maxwell, 2015)

**16 different kernels** with various memory access patterns:
- Memory-bound: `vector_add`, `saxpy`, `strided_copy`, `naive_transpose`
- Compute-bound: `matmul_naive`, `matmul_tiled`, `conv2d_3x3`, `conv2d_7x7`
- Complex patterns: `atomic_hotspot`, `reduce_sum`, `histogram`, `random_access`
- Shared memory: `shared_transpose`, `shared_bank_conflict`, `dot_product`
- Divergent execution: `vector_add_divergent`

## 🚀 Quick Start

### Prerequisites

```bash
# Install required Python packages
pip install pandas numpy scikit-learn

# Optional but recommended for best performance
pip install xgboost
```

### Running the Models

**Option 1: Run all models and compare results (recommended)**
```bash
cd gpu-perf/scripts
python run_all_models.py
```

**Option 2: Run models individually**
```bash
cd gpu-perf/scripts

# Analytical model (fastest, ~30 seconds)
python analytical_model_occupancy.py

# ML baseline (~1-2 minutes)
python ml_baseline.py

# Hybrid enhanced model (~2-5 minutes, best results)
python hybrid_model_enhanced.py
```

## 📁 Generated Files

After running, you'll find these files in `/gpu-perf/scripts/`:

### Analytical Model
- `cross_gpu_predictions.csv` - All 384 cross-GPU pairs with predictions
- `exp1_same_config_new_gpu.csv` - Experiment 1 results
- `exp1_kernel_metrics.csv` - Per-kernel error metrics
- `exp2_new_configs_same_gpus.csv` - Experiment 2 results
- `exp3a_new_kernels.csv` - Experiment 3 results

### ML Baseline
- `exp1_same_config_new_gpu_ml_predictions.csv`
- `exp1_same_config_new_gpu_kernel_metrics_ml.csv`
- (similar pattern for exp2 and exp3)

### Hybrid Enhanced (Best Results)
- `exp1_new_gpu_hybrid_predictions.csv`
- `exp1_new_gpu_hybrid_kernel_metrics.csv`
- `exp1_new_gpu_hybrid_feature_importance.csv` ⭐
- (similar pattern for exp2 and exp3)
- `hybrid_model_summary.csv` - Overall comparison

### Comparison
- `model_comparison.csv` - Side-by-side comparison
- `RESULTS_README.md` - Detailed explanation of results

## 🧪 Experimental Setup

### Experiment 1: **New GPU Generalization** (Hardest)
- **Goal**: Predict performance on NVIDIA TITAN V (held-out GPU)
- **Training**: All pairs where target ≠ TITAN V
- **Testing**: All pairs where target = TITAN V
- **Challenge**: Generalizing to different GPU architecture

### Experiment 2: **New Config Generalization** (Medium)
- **Goal**: Predict performance for larger problem sizes
- **Training**: Baseline and intermediate configs
- **Testing**: Largest problem size configs
- **Challenge**: Scaling predictions across problem sizes

### Experiment 3: **New Kernel Generalization** (Hard)
- **Goal**: Predict performance for unseen kernel types
- **Training**: 12 kernels (vector_add, saxpy, matmul_naive, etc.)
- **Testing**: 4 held-out kernels (matmul_tiled, shared_transpose, atomic_hotspot, vector_add_divergent)
- **Challenge**: Understanding kernel characteristics from limited examples

## 🔬 Model Details

### 1. Analytical Model

**Approach**: Physics-based roofline model + occupancy analysis

**Key Equations**:
```
Arithmetic Intensity (I) = FLOPs / BYTES
Roofline Bound = min(Peak_Compute, I × Bandwidth)
Occupancy = Active_Warps / Max_Warps_Per_SM
Efficiency = Measured_Perf / (Occupancy × Roofline)
```

**Prediction**:
```python
T_pred = FLOPs / (Efficiency × Occupancy_tgt × Roofline_tgt)
```

**Pros**:
- ✅ Interpretable and physically grounded
- ✅ No training data required
- ✅ Works well for regular memory access patterns

**Cons**:
- ❌ Assumes efficiency transfers perfectly
- ❌ Struggles with irregular patterns (atomics, divergence)
- ❌ Doesn't model cache effects well

**Expected Performance**: 30-50% MAPE for new GPU

---

### 2. ML Baseline

**Approach**: Random Forest with basic kernel and GPU features

**Features** (~35 total):
- Kernel: FLOPs, BYTES, arithmetic intensity, regs, shmem, block size, N, rows, cols
- Source GPU: compute, bandwidth, SM count, memory limits
- Target GPU: same specs
- Measured: T_src_ms (runtime on source GPU)

**Model**: Random Forest (200 trees)

**Pros**:
- ✅ Learns complex relationships automatically
- ✅ Handles irregular kernels better than analytical

**Cons**:
- ❌ Black box - hard to interpret
- ❌ Needs substantial training data
- ❌ May not generalize to very different GPUs

**Expected Performance**: 20-40% MAPE for new GPU

---

### 3. Hybrid Enhanced Model ⭐ **BEST APPROACH**

**Approach**: Physics-informed ML with enhanced feature engineering

**Key Innovation**: Use analytical model outputs as **features** for ML model!

**Features** (~60+ total):

1. **Basic Features** (from ML baseline)
   - Kernel characteristics, GPU specs, measured runtime

2. **Physics-Based Features** (NEW!):
   - `occupancy_src`, `occupancy_tgt` - from analytical model
   - `roofline_src_gflops`, `roofline_tgt_gflops`
   - `compute_efficiency_src` - how close to theoretical peak
   - `measured_gflops_src`, `measured_bw_src_gbps`

3. **Ratio Features** (NEW!):
   - `compute_ratio = tgt_compute / src_compute`
   - `bandwidth_ratio = tgt_bw / src_bw`
   - `sm_count_ratio`, `l2_cache_ratio`
   - `occupancy_ratio`, `roofline_ratio`

4. **Cache Awareness** (NEW!):
   - `working_set_per_l2_src`, `working_set_per_l2_tgt`
   - `cache_residency_src`, `cache_residency_tgt`

5. **Memory Pattern Encoding** (NEW!):
   - One-hot encoding: coalesced, strided, random, atomics, divergent

6. **Derived Features** (NEW!):
   - `threads_per_block`, `warps_per_block`
   - `reg_pressure_src`, `reg_pressure_tgt`

**Model**: XGBoost (if available) or Random Forest (300 trees, deeper)

**Improvements over baseline**:
- Log-transform for better scale handling
- Better handling of inf/nan values
- Comprehensive feature importance analysis

**Pros**:
- ✅ Best of both worlds: physics intuition + ML flexibility
- ✅ Interpretable via feature importance
- ✅ Handles all kernel types well
- ✅ Explicit cache modeling

**Expected Performance**: 10-25% MAPE for new GPU, 5-15% for same GPU

---

## 📈 Understanding Results

### Metrics Explained

- **MAPE** (Mean Absolute Percentage Error): Average |predicted - actual| / actual
  - Lower is better
  - 10% = excellent, 25% = good, 50% = acceptable, >100% = poor

- **Median pred/true**: Median ratio of predictions to ground truth
  - 1.0 = perfect, <1.0 = underestimate, >1.0 = overestimate

- **MAE** (Mean Absolute Error): Average error in milliseconds
  - Absolute metric, depends on problem scale

- **RMSE** (Root Mean Squared Error): Emphasizes larger errors
  - Penalizes outliers more than MAE

- **Within X%**: What percentage of predictions are within X% error
  - Within 25% is typically considered "good enough" for practical use

### Feature Importance (Hybrid Model Only)

The hybrid model outputs `expN_feature_importance.csv` showing which features matter most:

**Expected top features**:
1. `T_src_ms` - Runtime on source GPU (strong baseline)
2. `compute_ratio` - Relative compute power
3. `bandwidth_ratio` - Relative memory bandwidth
4. `occupancy_tgt` - How well kernel utilizes target GPU
5. `roofline_ratio` - Theoretical speedup/slowdown
6. `arith_intensity` - Compute vs memory bound
7. Memory pattern indicators

This tells you **why** the model makes certain predictions!

## 🎓 Using for Your Project Report

### Key Points to Highlight

1. **Problem Formulation**:
   - Cross-GPU performance prediction is essential for HPC portability
   - Traditional analytical models struggle with diverse kernel types
   - ML-only approaches lack interpretability

2. **Your Contribution** (Hybrid Model):
   - Novel combination of physics-based and data-driven approaches
   - Explicit modeling of cache effects and memory patterns
   - Comprehensive feature engineering based on GPU architecture

3. **Experimental Rigor**:
   - Three well-defined experiments testing different generalization scenarios
   - Held-out GPU (Titan V) for realistic evaluation
   - Per-kernel analysis to understand model strengths/weaknesses

4. **Results to Report**:
   - "Hybrid model achieves X% MAPE on new GPU, improving over analytical (Y%) and ML baseline (Z%)"
   - "Feature importance analysis reveals bandwidth_ratio and occupancy as key predictors"
   - "Model successfully generalizes to unseen kernels with W% error"

5. **Insights**:
   - Which kernels are hardest to predict? (atomic_hotspot, divergent patterns)
   - How much does cache size matter? (from cache_residency features)
   - When does roofline model work well vs fail? (compare analytical vs hybrid)

### Recommended Visualizations

Create these plots for your report (using Python matplotlib/seaborn):

1. **Scatter plot**: Predicted vs Actual runtime (color by kernel type)
2. **Bar chart**: MAPE by kernel (compare 3 models side-by-side)
3. **Error distribution**: Histogram of relative errors
4. **Feature importance**: Top 20 features for hybrid model
5. **GPU comparison**: How well does model transfer between GPU pairs?

## 🔧 Customization

### Changing Test GPU

In `analytical_model_occupancy.py`, `ml_baseline.py`, and `hybrid_model_enhanced.py`:

```python
TEST_GPU_NAME = "NVIDIA TITAN V"  # Change to any GPU in your dataset
```

### Adding More Kernels

Edit the TRAIN_KERNELS and TEST_KERNELS lists to create different experiment 3 splits.

### Trying Different ML Models

In `hybrid_model_enhanced.py`:

```python
MODEL_TYPE = "xgboost"  # or "random_forest"
USE_LOG_TRANSFORM = True  # or False
```

### Tuning Hyperparameters

For XGBoost:
```python
model = xgb.XGBRegressor(
    n_estimators=300,      # More trees = better fit but slower
    max_depth=8,           # Deeper = more complex interactions
    learning_rate=0.05,    # Lower = slower but more careful learning
    subsample=0.8,         # Use 80% of data per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
)
```

For Random Forest:
```python
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_leaf=3,
)
```

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'pandas'`
```bash
pip install pandas numpy scikit-learn
```

**Issue**: "XGBoost not found"
- Hybrid model will automatically fall back to Random Forest
- For best results: `pip install xgboost`

**Issue**: Memory error / Bus error
- Reduce `n_estimators` in model config
- Use `MODEL_TYPE = "random_forest"` instead of XGBoost
- Run on machine with more RAM (>8GB recommended)

**Issue**: No cross-GPU pairs found
- Check that your CSV files have the correct GPU names in `device_name` column
- Verify `gpu_metrics.json` has matching GPU names

**Issue**: Poor performance on new GPU
- This is expected! Cross-architecture prediction is inherently hard
- Try adding more training GPUs to dataset
- Collect more diverse kernels

## 📊 Expected Results Summary

Based on similar research and your data:

| Experiment | Analytical | ML Baseline | Hybrid Enhanced |
|------------|-----------|-------------|-----------------|
| Exp1: New GPU | 30-50% MAPE | 20-40% MAPE | **10-25% MAPE** ✅ |
| Exp2: New Config | 40-100% MAPE | 25-60% MAPE | **15-35% MAPE** ✅ |
| Exp3: New Kernels | 50-150% MAPE | 30-80% MAPE | **20-50% MAPE** ✅ |

Hybrid model should be best across all experiments!

## 🎯 Next Steps

1. **Run all models**: `python run_all_models.py`

2. **Analyze results**: Look at `model_comparison.csv` and kernel-level metrics

3. **Create visualizations**: Use pandas/matplotlib to plot results

4. **Feature analysis**: Which features matter most? (from feature_importance.csv)

5. **Extend the approach**:
   - Add more GPUs to your dataset
   - Try neural networks
   - Implement transfer learning
   - Add profiling metrics (NCU data)

## 📚 References

Key concepts used in this implementation:

- **Roofline Model**: Williams et al., "Roofline: An Insightful Visual Performance Model" (2009)
- **GPU Occupancy**: NVIDIA CUDA C Programming Guide
- **Cross-Platform Performance Modeling**: Hong & Kim, "An Analytical Model for a GPU Architecture with Memory-level and Thread-level Parallelism Awareness" (2009)
- **Physics-Informed ML**: Karniadakis et al., "Physics-informed machine learning" (2021)

## ✅ Summary

You now have:
- ✅ Three working prediction models
- ✅ Comprehensive evaluation framework
- ✅ Feature importance analysis
- ✅ Per-kernel and per-experiment breakdowns
- ✅ Automated comparison scripts

**No CUDA cluster needed** - all models train on existing CSV data!

**Best approach**: Use the **Hybrid Enhanced Model** for best accuracy and interpretability.

Good luck with your project! 🚀
