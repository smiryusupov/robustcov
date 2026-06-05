"""Plot speed benchmark CSV as a bar chart.

Run:
    python examples/plot_speed_comparison.py --input results/speed.csv --output results/speed.png
"""
from __future__ import annotations

import argparse
import robustcov as rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV from benchmarks/speed_estimators.py')
    parser.add_argument('--output', default='results/speed_comparison.png')
    args = parser.parse_args()

    rc.plot_benchmark_bars(
        args.input,
        category_col='method',
        value_col='median_seconds',
        title='Estimator median fit time',
        output_path=args.output,
        show=False,
    )
    print('saved plot to', args.output)
