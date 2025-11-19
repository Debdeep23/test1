#!/usr/bin/env python3
"""
Parse ncu text output files and extract key metrics.
Usage: python3 scripts/parse_ncu_text.py <ncu_details.txt> [output.csv]
       python3 scripts/parse_ncu_text.py data/profiling/*.txt [output.csv]
"""

import sys
import re
import os
from pathlib import Path

def parse_ncu_text_file(filepath):
    """Parse a single ncu details text file and extract key metrics."""

    metrics = {
        'kernel': os.path.basename(filepath).replace('ncu_', '').replace('_details.txt', ''),
        'file': filepath
    }

    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}", file=sys.stderr)
        return metrics

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Patterns to extract metrics
        patterns = {
            # Occupancy metrics
            'theoretical_occupancy': r'Theoretical Occupancy\s+%\s+([\d.]+)',
            'achieved_occupancy': r'Achieved Occupancy\s+%\s+([\d.]+)',
            'achieved_warps_per_sm': r'Achieved Active Warps Per SM\s+warp\s+([\d.]+)',
            'theoretical_warps_per_sm': r'Theoretical Active Warps per SM\s+warp\s+([\d.]+)',

            # Block limits
            'block_limit_registers': r'Block Limit Registers\s+block\s+([\d.]+)',
            'block_limit_shared_mem': r'Block Limit Shared Mem\s+block\s+([\d.]+)',
            'block_limit_warps': r'Block Limit Warps\s+block\s+([\d.]+)',

            # Memory cycles
            'avg_dram_active_cycles': r'Average DRAM Active Cycles\s+cycle\s+([\d,.]+)',
            'total_dram_elapsed_cycles': r'Total DRAM Elapsed Cycles\s+cycle\s+([\d,.]+)',
            'avg_l1_active_cycles': r'Average L1 Active Cycles\s+cycle\s+([\d,.]+)',
            'total_l1_elapsed_cycles': r'Total L1 Elapsed Cycles\s+cycle\s+([\d,.]+)',
            'avg_l2_active_cycles': r'Average L2 Active Cycles\s+cycle\s+([\d,.]+)',
            'total_l2_elapsed_cycles': r'Total L2 Elapsed Cycles\s+cycle\s+([\d,.]+)',
            'avg_sm_active_cycles': r'Average SM Active Cycles\s+cycle\s+([\d,.]+)',
            'total_sm_elapsed_cycles': r'Total SM Elapsed Cycles\s+cycle\s+([\d,.]+)',

            # Branch metrics
            'branch_instructions_ratio': r'Branch Instructions Ratio\s+%\s+([\d.]+)',
            'branch_instructions': r'Branch Instructions\s+inst\s+([\d,.]+)',
            'branch_efficiency': r'Branch Efficiency\s+%\s+([\d.]+)',
            'avg_divergent_branches': r'Avg\. Divergent Branches\s+branches\s+([\d,.]+)',

            # Throughput metrics (if present)
            'sm_throughput_pct': r'sm__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)',
            'dram_throughput_pct': r'dram__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)',
            'l1_throughput_pct': r'l1tex__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)',
            'l2_throughput_pct': r'lts__throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)',

            # Duration
            'duration_ns': r'gpu__time_duration\.sum\s+nsecond\s+([\d,.]+)',
            'duration_us': r'gpu__time_duration\.sum\s+usecond\s+([\d,.]+)',

            # Compute/memory throughput
            'compute_memory_throughput_pct': r'gpu__compute_memory_throughput\.avg\.pct_of_peak_sustained_elapsed\s+%\s+([\d.]+)',

            # Warp stalls (if present)
            'warp_stall_short_scoreboard': r'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active\.pct\s+%\s+([\d.]+)',
            'warp_stall_long_scoreboard': r'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active\.pct\s+%\s+([\d.]+)',
            'warp_stall_drain': r'smsp__average_warps_issue_stalled_drain_per_issue_active\.pct\s+%\s+([\d.]+)',

            # Memory operations
            'global_load_bytes': r'l1tex__t_bytes_pipe_lsu_mem_global_op_ld\.sum\.per_second\s+byte/second\s+([\d,.]+)',
            'global_store_bytes': r'l1tex__t_bytes_pipe_lsu_mem_global_op_st\.sum\.per_second\s+byte/second\s+([\d,.]+)',
        }

        # Extract metrics
        for metric_name, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                value_str = match.group(1).replace(',', '')
                try:
                    metrics[metric_name] = float(value_str)
                except ValueError:
                    metrics[metric_name] = value_str

        # Calculate derived metrics
        if 'achieved_occupancy' in metrics and 'theoretical_occupancy' in metrics:
            theo = metrics['theoretical_occupancy']
            ach = metrics['achieved_occupancy']
            if theo > 0:
                metrics['occupancy_ratio'] = ach / theo

        # Calculate DRAM utilization
        if 'avg_dram_active_cycles' in metrics and 'total_dram_elapsed_cycles' in metrics:
            active = metrics['avg_dram_active_cycles']
            elapsed = metrics['total_dram_elapsed_cycles']
            if elapsed > 0:
                metrics['dram_utilization_pct'] = (active / elapsed) * 100

        # Calculate SM utilization
        if 'avg_sm_active_cycles' in metrics and 'total_sm_elapsed_cycles' in metrics:
            active = metrics['avg_sm_active_cycles']
            elapsed = metrics['total_sm_elapsed_cycles']
            if elapsed > 0:
                metrics['sm_utilization_pct'] = (active / elapsed) * 100

    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)

    return metrics

def format_metric_value(value):
    """Format metric value for CSV output."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <ncu_details.txt> [output.csv]")
        print(f"       {sys.argv[0]} data/profiling/*.txt")
        print()
        print("Parse ncu text output and extract key performance metrics")
        sys.exit(1)

    # Collect all input files
    input_files = []
    output_file = None

    for arg in sys.argv[1:]:
        if arg.endswith('.csv'):
            output_file = arg
        elif os.path.isfile(arg) and arg.endswith('.txt'):
            input_files.append(arg)
        elif '*' in arg:
            # Handle glob patterns
            import glob
            input_files.extend(glob.glob(arg))

    if not input_files:
        print("Error: No input files found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {len(input_files)} ncu text file(s)...")

    # Parse all files
    all_metrics = []
    for filepath in sorted(input_files):
        print(f"  Parsing: {filepath}")
        metrics = parse_ncu_text_file(filepath)
        all_metrics.append(metrics)

    # Collect all unique metric keys
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())

    # Sort keys, put kernel and file first
    ordered_keys = ['kernel', 'file']
    other_keys = sorted([k for k in all_keys if k not in ordered_keys])
    ordered_keys.extend(other_keys)

    # Output results
    if output_file:
        # Write to CSV
        import csv
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
            writer.writeheader()
            for metrics in all_metrics:
                # Format float values
                formatted = {k: format_metric_value(v) for k, v in metrics.items()}
                writer.writerow(formatted)
        print(f"\nResults written to: {output_file}")
    else:
        # Print to stdout in a readable format
        print("\n" + "="*80)
        print("NCU PROFILING RESULTS SUMMARY")
        print("="*80)

        for metrics in all_metrics:
            print(f"\n{metrics.get('kernel', 'unknown')}:")
            print("-" * 40)

            # Key metrics to highlight
            key_metrics = [
                ('achieved_occupancy', 'Achieved Occupancy (%)', 'pct'),
                ('theoretical_occupancy', 'Theoretical Occupancy (%)', 'pct'),
                ('sm_utilization_pct', 'SM Utilization (%)', 'pct'),
                ('dram_utilization_pct', 'DRAM Utilization (%)', 'pct'),
                ('branch_efficiency', 'Branch Efficiency (%)', 'pct'),
                ('avg_divergent_branches', 'Avg Divergent Branches', 'num'),
                ('duration_us', 'Duration (μs)', 'num'),
            ]

            for key, label, fmt in key_metrics:
                if key in metrics:
                    value = metrics[key]
                    if fmt == 'pct':
                        print(f"  {label:30s}: {value:>10.2f}")
                    else:
                        print(f"  {label:30s}: {value:>10.4f}")

        # Save to default CSV in same directory as first input
        default_output = os.path.join(
            os.path.dirname(input_files[0]) or '.',
            'ncu_parsed_summary.csv'
        )

        import csv
        with open(default_output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys, extrasaction='ignore')
            writer.writeheader()
            for metrics in all_metrics:
                formatted = {k: format_metric_value(v) for k, v in metrics.items()}
                writer.writerow(formatted)

        print(f"\n\nFull results also saved to: {default_output}")

if __name__ == '__main__':
    main()
