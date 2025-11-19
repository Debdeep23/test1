# GPU Performance Dataset Generation

## Quick Start

### Option 1: Generate Everything (Recommended)
Run trials and process in one command:
```bash
cd gpu-perf
./scripts/generate_dataset_full.sh
```

This will:
1. Build the CUDA runner
2. Run all 16 kernels with 4 different sizes each (~65 configurations)
3. Process trials into final dataset with all metrics
4. Output: `data/runs_2080ti_final.csv`

**Time:** ~10-20 minutes (depending on GPU)

---

### Option 2: Two-Step Process

**Step 1: Generate trials only**
```bash
cd gpu-perf
./scripts/gen_trials_2080ti.sh
```

**Step 2: Process existing trials**
```bash
./scripts/process_trials_only.sh
```

Use this approach if you want to:
- Generate trials once, process multiple times
- Modify processing logic without re-running kernels
- Debug the processing pipeline

---

## What You Get

### Complete Metrics Coverage ✅

**Kernel Metrics (11/11):**
1. Total FLOPs
2. Total bytes
3. Arithmetic intensity
4. Memory access pattern
5. Working set size
6. Threads per block
7. Grid size
8. Registers per thread
9. Shared memory per block
10. Branch divergence flag
11. Atomic operations count

**GPU Metrics (13/13):**
1. Peak theoretical FP32 GFLOPS
2. Peak theoretical bandwidth
3. Sustained bandwidth (calibrated)
4. Sustained compute (calibrated)
5. Number of SMs
6. Max threads/SM
7. Max blocks/SM
8. Registers/SM
9. Shared memory/SM
10. L2 cache size
11. Architecture name
12. Compute capability
13. Warp size

### Dataset Size
- **Rows:** ~65 (4 sizes × 16 kernels, except shared_bank_conflict)
- **Columns:** 40 organized metrics

---

## Pipeline Details

### New Pipeline (Recommended)

```
gen_trials_2080ti.sh → build_final_dataset.py → runs_2080ti_final.csv
```

**Single script does everything:**
- Aggregates trials
- Calculates kernel metrics
- Adds GPU metrics
- Computes achieved performance
- Adds performance models

### Old Pipeline (DEPRECATED)

```
aggregate_trials.py → static_counts.py → enrich_with_gpu_metrics.py → add_singlethread_baseline.py
```

⚠️ **Do NOT use `run_2080ti_full.sh`** - it uses the old pipeline and doesn't include new metrics!

---

## File Structure

### Scripts
```
scripts/
├── generate_dataset_full.sh    # NEW: Complete generation pipeline
├── process_trials_only.sh      # NEW: Process existing trials only
├── gen_trials_2080ti.sh        # UPDATED: Generate trials with multiple sizes
├── build_final_dataset.py      # UPDATED: All-in-one processing (11+13 metrics)
├── add_singlethread_baseline.py # UPDATED: Filters useless columns
└── run_2080ti_full.sh          # DEPRECATED: Old pipeline (don't use)
```

### Data Files
```
data/
├── trials_*__2080ti.csv              # Raw trial data (16 files)
├── runs_2080ti_final.csv             # Final dataset (40 columns)
├── props_2080ti.out                  # GPU properties
├── stream_like_2080ti.out            # Bandwidth calibration
└── gemm_cublas_2080ti.out            # Compute calibration
```

---

## Kernel Sizes Tested

### 1D Kernels (N parameter)
- Small: 262,144 (256K)
- Medium: 1,048,576 (1M)
- Large: 4,194,304 (4M)
- XLarge: 16,777,216 (16M)

**Kernels:** vector_add, saxpy, strided_copy_8, reduce_sum, dot_product, histogram, random_access, vector_add_divergent

### 2D Kernels (rows × cols)
- Small: 512 × 512
- Medium: 1024 × 1024
- Large: 2048 × 2048
- XLarge: 4096 × 4096

**Kernels:** naive_transpose, shared_transpose, conv2d_3x3, conv2d_7x7

### Matrix Multiply (matN)
- Small: 256 × 256
- Medium: 512 × 512
- Large: 1024 × 1024
- XLarge: 2048 × 2048

**Kernels:** matmul_naive, matmul_tiled

### Special Cases
- **shared_bank_conflict:** Fixed size (1 configuration)
- **atomic_hotspot:** 3 sizes × 2 iteration counts (6 configurations)

---

## Troubleshooting

### "No trial files found"
You need to run `gen_trials_2080ti.sh` first:
```bash
./scripts/gen_trials_2080ti.sh
```

### "Missing calibration files"
Generate calibration data:
```bash
cd calibration
# Run GPU properties
nvcc -o props props.cu && ./props > ../data/props_2080ti.out

# Run memory bandwidth calibration
nvcc -o stream stream_like.cu && ./stream > ../data/stream_like_2080ti.out

# Run compute calibration
nvcc -o gemm gemm_cublas.cu -lcublas && ./gemm > ../data/gemm_cublas_2080ti.out
```

### "build_final_dataset.py not found"
Make sure you're in the `gpu-perf` directory when running scripts.

---

## Column Reference

**Final CSV has 40 columns organized as:**

1. **Kernel Identification (3):** kernel, regs, shmem
2. **Launch Configuration (2):** block, grid_blocks
3. **Performance Results (2):** mean_ms, std_ms
4. **Problem Sizes (7):** N, rows, cols, H, W, matN, size_kind
5. **Kernel Metrics (8):** FLOPs, BYTES, shared_bytes, working_set_bytes, arithmetic_intensity, mem_pattern, has_branch_divergence, atomic_ops_count
6. **GPU Hardware (10):** gpu_device_name, gpu_architecture, gpu_compute_capability, gpu_sms, gpu_max_threads_per_sm, gpu_max_blocks_per_sm, gpu_regs_per_sm, gpu_shared_mem_per_sm, gpu_l2_bytes, gpu_warp_size
7. **GPU Performance Limits (4):** peak_theoretical_gflops, peak_theoretical_bandwidth_gbps, calibrated_mem_bandwidth_gbps, calibrated_compute_gflops
8. **Achieved Performance (2):** achieved_bandwidth_gbps, achieved_compute_gflops
9. **Performance Models (2):** T1_model_ms, speedup_model

---

## Migration from Old Pipeline

If you were using `run_2080ti_full.sh`:

**OLD:**
```bash
./scripts/run_2080ti_full.sh
```

**NEW:**
```bash
./scripts/generate_dataset_full.sh
```

**Changes:**
- ✅ Now uses `build_final_dataset.py` (all metrics in one script)
- ✅ Includes 6 new metrics (branch divergence, atomic ops, architecture, etc.)
- ✅ Generates 4× more data (multiple sizes per kernel)
- ✅ Cleaner column structure (40 instead of 46)
- ✅ Complete 11/11 kernel + 13/13 GPU metrics
