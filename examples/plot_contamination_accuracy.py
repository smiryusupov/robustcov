"""Plot contamination benchmark CSV as a line chart.

Run:
    python examples/plot_contamination_accuracy.py --input results/accuracy.csv --output results/accuracy.png
"""
from __future__ import annotations

import argparse
import robustcov as rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input CSV from benchmarks/accuracy_vs_contamination.py')
    parser.add_argument('--output', default='results/accuracy_contamination.png')
    args = parser.parse_args()

    rc.plot_benchmark_curve(
        args.input,
        x_col='contamination',
        y_col='rel_fro_error',
        group_col='method',
        title='Relative covariance error vs contamination',
        output_path=args.output,
        show=False,
    )
    print('saved plot to', args.output)
