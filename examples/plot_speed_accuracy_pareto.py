"""Plot speed-accuracy Pareto chart from a tradeoff benchmark CSV.

Run:
    python benchmarks/fastmcd_quality_speed_tradeoff.py --csv results/tradeoff.csv
    python examples/plot_speed_accuracy_pareto.py --input results/tradeoff.csv --output results/pareto.png
"""
from __future__ import annotations

import argparse
import robustcov as rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default='results/speed_accuracy_pareto.png')
    args = parser.parse_args()
    rc.plot_speed_accuracy_pareto(args.input, output_path=args.output, show=False)
    print('saved plot to', args.output)
