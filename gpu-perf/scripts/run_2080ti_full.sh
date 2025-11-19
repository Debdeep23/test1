#!/usr/bin/env bash
set -euo pipefail

echo "=== 0) Clean ==="
rm -f data/trials_*__2080ti.csv data/runs_2080ti*.csv

echo "=== 1) Trials (10 per kernel) ==="
# keep your current run list; examples:
scripts/run_trials.sh vector_add            "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100"  12 0    10
scripts/run_trials.sh saxpy                 "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100"  12 0    10
scripts/run_trials.sh strided_copy_8        "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100"   8 0    10
scripts/run_trials.sh naive_transpose       "--rows 2048    --cols 2048 --warmup 20 --reps 100"             16 0    10
scripts/run_trials.sh shared_transpose      "--rows 2048    --cols 2048 --warmup 20 --reps 100"             32 4224 10
scripts/run_trials.sh reduce_sum            "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100"  10 1024 10
scripts/run_trials.sh dot_product           "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100"  15 1024 10
scripts/run_trials.sh histogram             "--rows 1048576 --cols 1   --block 256 --warmup 10  --reps 50"  10 1024 10
scripts/run_trials.sh conv2d_3x3            "--rows 1024    --cols 1024 --warmup 10 --reps 50"              30 0    10
scripts/run_trials.sh conv2d_7x7            "--rows 1024    --cols 1024 --warmup 10 --reps 50"              40 0    10
scripts/run_trials.sh random_access         "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100"  10 0    10
scripts/run_trials.sh vector_add_divergent  "--rows 1048576 --cols 1   --block 256 --warmup 20 --reps 100"  15 0    10
scripts/run_trials.sh shared_bank_conflict  "--warmup 20 --reps 100"                                       206 4096 10
scripts/run_trials.sh atomic_hotspot        "--rows 1048576 --cols 1   --block 256 --iters 100 --warmup 10 --reps 50"  7 0 10
scripts/run_trials.sh matmul_naive          "--rows 512 --cols 512 --warmup 10 --reps 50"                   40 0    10
scripts/run_trials.sh matmul_tiled          "--rows 512 --cols 512 --warmup 10 --reps 50"                   37 8192 10

echo "=== 2) Validate trials ==="
for f in data/trials_*__2080ti.csv; do
  scripts/validate_csv.py "$f"
done

echo "=== 3) Aggregate → runs_2080ti.csv ==="
python3 scripts/aggregate_trials.py data/trials_*__2080ti.csv > data/runs_2080ti.csv

echo "=== 4) Static counts → runs_2080ti_with_counts.csv ==="
python3 scripts/static_counts.py data/runs_2080ti.csv data/runs_2080ti_with_counts.csv

echo "=== 5) Enrich with GPU metrics → runs_2080ti_enriched.csv ==="
python3 scripts/enrich_with_gpu_metrics.py \
  data/runs_2080ti_with_counts.csv \
  data/props_2080ti.out \
  data/stream_like_2080ti.out \
  data/gemm_cublas_2080ti.out \
  data/runs_2080ti_enriched.csv

echo "=== 6) Add single-thread baseline → runs_2080ti_final.csv ==="
python3 scripts/add_singlethread_baseline.py \
  data/runs_2080ti_enriched.csv \
  data/device_calibration_2080ti.json \
  data/runs_2080ti_final.csv \
  32

echo "=== 7) Peek ==="
head -5 data/runs_2080ti_final.csv

