#!/usr/bin/env python3
"""
Parse profiling results from nvprof or ncu CSV files and generate summary reports.
Usage: python3 scripts/parse_profiling_results.py <profiling_dir> [output.csv]
"""

import sys
import os
import csv
import glob
import json
from pathlib import Path

def parse_nvprof_metrics(csv_file):
    """Parse nvprof metrics CSV file."""
    metrics = {}

    if not os.path.exists(csv_file):
        return metrics

    try:
        with open(csv_file, 'r') as f:
            # nvprof CSV format has headers in first few rows
            lines = f.readlines()

            # Find the actual data rows (skip header rows)
            data_start = 0
            for i, line in enumerate(lines):
                if line.startswith('"Metric Name"') or line.startswith('Metric Name'):
                    data_start = i + 1
                    break

            # Parse metrics
            for line in lines[data_start:]:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',')
                if len(parts) >= 2:
                    metric_name = parts[0].strip('"')
                    metric_value = parts[1].strip('"')

                    # Try to convert to float if possible
                    try:
                        metric_value = float(metric_value)
                    except ValueError:
                        pass

                    metrics[metric_name] = metric_value

    except Exception as e:
        print(f"Warning: Could not parse {csv_file}: {e}", file=sys.stderr)

    return metrics

def parse_ncu_csv(csv_file):
    """Parse ncu metrics CSV file."""
    metrics = {}

    if not os.path.exists(csv_file):
        return metrics

    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    # Clean up metric names
                    clean_key = key.strip('"').strip()
                    clean_value = value.strip('"').strip()

                    # Try to convert to float
                    try:
                        clean_value = float(clean_value)
                    except ValueError:
                        pass

                    metrics[clean_key] = clean_value

    except Exception as e:
        print(f"Warning: Could not parse {csv_file}: {e}", file=sys.stderr)

    return metrics

def extract_key_metrics(metrics, tool='nvprof'):
    """Extract the most important metrics for analysis."""

    key_metrics = {}

    if tool == 'nvprof':
        # Map of important metrics
        important = {
            'achieved_occupancy': 'occupancy',
            'sm_efficiency': 'sm_efficiency',
            'warp_execution_efficiency': 'warp_efficiency',
            'branch_efficiency': 'branch_efficiency',
            'dram_utilization': 'dram_utilization',
            'dram_read_throughput': 'dram_read_gb_s',
            'dram_write_throughput': 'dram_write_gb_s',
            'gld_efficiency': 'gld_efficiency',
            'gst_efficiency': 'gst_efficiency',
            'gld_throughput': 'gld_throughput',
            'gst_throughput': 'gst_throughput',
            'l1_cache_global_hit_rate': 'l1_hit_rate',
            'l2_l1_read_hit_rate': 'l2_hit_rate',
            'shared_efficiency': 'shared_efficiency',
            'flop_count_sp': 'flops',
            'flop_sp_efficiency': 'flop_efficiency',
            'inst_per_warp': 'inst_per_warp',
        }
    else:  # ncu
        important = {
            'gpu__time_duration.sum': 'duration_ns',
            'sm__throughput.avg.pct_of_peak_sustained_elapsed': 'sm_throughput_pct',
            'dram__throughput.avg.pct_of_peak_sustained_elapsed': 'dram_throughput_pct',
            'sm__warps_active.avg.pct_of_peak_sustained_active': 'warp_activity_pct',
        }

    # Extract available metrics
    for metric_key, output_name in important.items():
        if metric_key in metrics:
            key_metrics[output_name] = metrics[metric_key]

    return key_metrics

def analyze_profiling_dir(profiling_dir, tool='nvprof'):
    """Analyze all profiling results in a directory."""

    results = {}

    # Pattern for metric files
    if tool == 'nvprof':
        pattern = os.path.join(profiling_dir, 'nvprof_*_metrics.csv')
    else:
        pattern = os.path.join(profiling_dir, 'ncu_*_metrics.csv')

    metric_files = glob.glob(pattern)

    print(f"Found {len(metric_files)} profiling result files")

    for metric_file in sorted(metric_files):
        # Extract kernel name from filename
        basename = os.path.basename(metric_file)
        if tool == 'nvprof':
            kernel = basename.replace('nvprof_', '').replace('_metrics.csv', '')
        else:
            kernel = basename.replace('ncu_', '').replace('_metrics.csv', '')

        print(f"  Parsing {kernel}...")

        # Parse metrics
        if tool == 'nvprof':
            metrics = parse_nvprof_metrics(metric_file)
        else:
            metrics = parse_ncu_csv(metric_file)

        # Extract key metrics
        key_metrics = extract_key_metrics(metrics, tool)

        results[kernel] = {
            'all_metrics': metrics,
            'key_metrics': key_metrics
        }

    return results

def write_summary_csv(results, output_file):
    """Write summary of key metrics to CSV."""

    if not results:
        print("No results to write")
        return

    # Collect all unique metric names
    all_metric_names = set()
    for kernel_data in results.values():
        all_metric_names.update(kernel_data['key_metrics'].keys())

    metric_names = sorted(all_metric_names)

    # Write CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['kernel'] + metric_names)

        # Data rows
        for kernel in sorted(results.keys()):
            row = [kernel]
            for metric in metric_names:
                value = results[kernel]['key_metrics'].get(metric, '')
                row.append(value)
            writer.writerow(row)

    print(f"\nSummary written to: {output_file}")

def print_summary(results):
    """Print a human-readable summary."""

    print("\n" + "="*80)
    print("PROFILING SUMMARY")
    print("="*80)

    for kernel in sorted(results.keys()):
        print(f"\n{kernel}:")
        print("-" * 40)

        key_metrics = results[kernel]['key_metrics']

        if not key_metrics:
            print("  No key metrics found")
            continue

        for metric, value in sorted(key_metrics.items()):
            if isinstance(value, float):
                print(f"  {metric:25s}: {value:>12.4f}")
            else:
                print(f"  {metric:25s}: {value:>12}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <profiling_dir> [output.csv] [--tool nvprof|ncu]")
        print()
        print("Examples:")
        print(f"  {sys.argv[0]} data/profiling_2080")
        print(f"  {sys.argv[0]} data/profiling_2080 summary.csv")
        print(f"  {sys.argv[0]} data/profiling_2080 summary.csv --tool ncu")
        sys.exit(1)

    profiling_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None

    # Determine tool
    tool = 'nvprof'  # default
    if '--tool' in sys.argv:
        tool_idx = sys.argv.index('--tool')
        if tool_idx + 1 < len(sys.argv):
            tool = sys.argv[tool_idx + 1]

    # Auto-detect tool if not specified
    if tool == 'nvprof' and not glob.glob(os.path.join(profiling_dir, 'nvprof_*_metrics.csv')):
        if glob.glob(os.path.join(profiling_dir, 'ncu_*_metrics.csv')):
            tool = 'ncu'
            print("Auto-detected ncu output files")

    print(f"Analyzing profiling results in: {profiling_dir}")
    print(f"Using tool: {tool}")
    print()

    # Analyze
    results = analyze_profiling_dir(profiling_dir, tool)

    if not results:
        print("No profiling results found!")
        sys.exit(1)

    # Print summary
    print_summary(results)

    # Write CSV if requested
    if output_file:
        write_summary_csv(results, output_file)
    else:
        # Default output filename
        default_output = os.path.join(profiling_dir, f'profiling_summary_{tool}.csv')
        write_summary_csv(results, default_output)

if __name__ == '__main__':
    main()
