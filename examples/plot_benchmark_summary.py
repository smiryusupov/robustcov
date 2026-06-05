"""Plot estimator ranking / win-rate summary CSVs.

Run:
    python benchmarks/benchmark_summary.py --input results/small_sample.csv --csv results/summary.csv
    python examples/plot_benchmark_summary.py --input results/summary.csv --metric mean_rank --output results/summary_rank.png
"""
from __future__ import annotations

import argparse
import robustcov as rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--metric', default='mean_rank', choices=['mean_rank', 'median_error', 'win_rate', 'median_seconds'])
    parser.add_argument('--output', default='results/benchmark_summary.png')
    args = parser.parse_args()
    rc.plot_benchmark_bars(
        args.input,
        category_col='method',
        value_col=args.metric,
        title=f'Benchmark summary: {args.metric}',
        output_path=args.output,
        show=False,
    )
    print('saved plot to', args.output)
