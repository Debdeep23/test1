# Dataset Iteration Columns

## Changes Made

Both `reps` and `iters` columns are now **kept** in the final datasets for both RTX 2080 Ti and RTX 4070.

### Previous Behavior
- Both `reps` and `iters` were excluded from final datasets
- Reasoning: "iters is kernel-specific and mostly empty, reps is a configuration parameter"

### New Behavior
- Both `reps` and `iters` are **included** in final datasets
- Reasoning: These are important iteration metadata for understanding kernel execution

## Column Meanings

### `reps` (repetitions)
- **What**: Number of times kernel is executed for timing measurements
- **Source**: `--reps N` command-line argument
- **Used by**: All kernels (for statistical accuracy)
- **Typical values**: 10-100
- **Example**: `--reps 50` means kernel runs 50 times, time is averaged

### `iters` (iterations)
- **What**: Iterations performed **inside** the kernel
- **Source**: `--iters N` command-line argument
- **Used by**: Only `atomic_hotspot` kernel
- **Typical values**: 0 for most kernels, 100 for atomic_hotspot
- **Example**: atomic_hotspot with `--iters 100` means each thread does 100 atomic operations

## Why Keep Both?

1. **Reproducibility**: Important for reproducing exact kernel execution
2. **Statistical metadata**: `reps` shows how many measurements were averaged
3. **Kernel behavior**: `iters` affects atomic_hotspot's work per thread
4. **Analysis**: Useful for understanding variance and statistical significance

## Example Data

### Without iters parameter (most kernels)
```csv
kernel,reps,iters
vector_add,50,0
saxpy,50,0
matmul_tiled,50,0
```

### With iters parameter (atomic_hotspot)
```csv
kernel,reps,iters
atomic_hotspot,50,100
```
- `reps=50`: Kernel executed 50 times for timing
- `iters=100`: Each thread performs 100 atomic operations per execution

## Updated Schemas

### RTX 2080 Ti Final Dataset
- Script: `add_singlethread_baseline.py`
- Change: Removed `reps` and `iters` from `exclude_cols`
- Columns now include: ..., `reps`, ..., `iters`, ...

### RTX 4070 Final Dataset
- Script: `make_final_4070.py`
- Change: Added `reps` to fieldnames (iters was already there)
- Columns now include: ..., `reps`, `iters`, ...

## Impact

### Files Changed
1. `gpu-perf/scripts/add_singlethread_baseline.py` - 2080 Ti pipeline
2. `gpu-perf/scripts/make_final_4070.py` - 4070 pipeline

### Datasets to Regenerate
1. `data/runs_2080ti_final.csv` - Need to re-run pipeline to add `reps` column
2. `data/runs_4070_final.csv` - Need to re-run pipeline to add `reps` column

## How to Regenerate

```bash
cd gpu-perf

# For RTX 2080 Ti
scripts/run_2080ti_full.sh

# For RTX 4070
scripts/run_4070_full.sh
```

Both final CSVs will now include complete iteration metadata.
