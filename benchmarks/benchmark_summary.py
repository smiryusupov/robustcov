
"""Aggregate benchmark CSVs into ranking/win-rate tables.

This script expects a CSV with a method column, an error column, and optional timing /
scenario columns. It computes within-scenario error ranks, win rates, median error, and
median seconds. It can also write presentation-friendly HTML and Markdown tables.

Examples:
    python benchmarks/benchmark_summary.py --input results/small_sample.csv --csv results/small_sample_summary.csv
    python benchmarks/benchmark_summary.py --input results/small_sample.csv --html results/small_sample_summary.html
"""
from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _to_float(value):
    try:
        x = float(value)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return np.nan


def _scenario_key(row, cols):
    return tuple(row.get(c, '') for c in cols)


def summarize(rows, method_col, error_col, time_col, scenario_cols):
    groups = defaultdict(list)
    for row in rows:
        groups[_scenario_key(row, scenario_cols)].append(row)

    method_stats = defaultdict(lambda: {
        'errors': [], 'times': [], 'ranks': [], 'wins': 0, 'appearances': 0, 'failures': 0,
    })

    for _, items in groups.items():
        scored = []
        for row in items:
            method = row.get(method_col, '')
            err = _to_float(row.get(error_col, ''))
            sec = _to_float(row.get(time_col, '')) if time_col else np.nan
            st = method_stats[method]
            st['appearances'] += 1
            if np.isfinite(sec):
                st['times'].append(sec)
            if np.isfinite(err):
                st['errors'].append(err)
                scored.append((err, method))
            else:
                st['failures'] += 1
        scored.sort(key=lambda x: x[0])
        if scored:
            best_err = scored[0][0]
            for rank, (err, method) in enumerate(scored, start=1):
                method_stats[method]['ranks'].append(rank)
                if err == best_err:
                    method_stats[method]['wins'] += 1

    out = []
    for method, st in method_stats.items():
        errors = np.asarray(st['errors'], dtype=float)
        times = np.asarray(st['times'], dtype=float)
        ranks = np.asarray(st['ranks'], dtype=float)
        appearances = max(1, st['appearances'])
        out.append({
            'method': method,
            'appearances': st['appearances'],
            'failures': st['failures'],
            'win_rate': f"{st['wins'] / appearances:.4f}",
            'mean_rank': f"{float(np.mean(ranks)):.4f}" if ranks.size else 'nan',
            'median_error': f"{float(np.median(errors)):.4f}" if errors.size else 'nan',
            'mean_error': f"{float(np.mean(errors)):.4f}" if errors.size else 'nan',
            'median_seconds': f"{float(np.median(times)):.6f}" if times.size else '',
        })
    out.sort(key=lambda r: (float(r['mean_rank']) if r['mean_rank'] != 'nan' else 1e9, float(r['median_error']) if r['median_error'] != 'nan' else 1e9))
    return out


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append('| ' + ' | '.join(fieldnames) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(fieldnames)) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(str(row.get(c, '')) for c in fieldnames) + ' |')
    path.write_text('\n'.join(lines) + '\n')


def write_html(path: Path, rows, fieldnames, title='Benchmark summary'):
    path.parent.mkdir(parents=True, exist_ok=True)
    body_rows = []
    for i, row in enumerate(rows):
        cells = ''.join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in fieldnames)
        cls = ' class="top"' if i == 0 else ''
        body_rows.append(f"<tr{cls}>{cells}</tr>")
    head = ''.join(f"<th>{html.escape(c.replace('_', ' ').title())}</th>" for c in fieldnames)
    path.write_text(f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>{html.escape(title)}</title>
<style>
:root {{ --bg:#f7f8fb; --card:#ffffff; --ink:#17202a; --muted:#5d6d7e; --line:#e5e8ef; --accent:#2d6cdf; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
main {{ max-width:1100px; margin:40px auto; padding:0 20px; }}
h1 {{ font-size:34px; margin:0 0 8px; }}
p {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:18px; box-shadow:0 10px 30px rgba(23,32,42,.06); overflow:hidden; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ background:#eef3ff; color:#1f3a68; text-align:left; font-size:13px; letter-spacing:.02em; }}
th,td {{ padding:12px 14px; border-bottom:1px solid var(--line); }}
tr.top td {{ font-weight:700; background:#f6fbf8; }}
tr:hover td {{ background:#fafcff; }}
.badge {{ display:inline-block; background:#eaf1ff; color:#1f55ad; border-radius:999px; padding:5px 10px; font-size:13px; }}
</style>
</head>
<body><main>
<span class=\"badge\">robustcov benchmark</span>
<h1>{html.escape(title)}</h1>
<p>Lower mean rank and lower median error are better. The first row is highlighted as the best aggregate rank.</p>
<section class=\"card\"><table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></section>
</main></body></html>""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--csv', default='')
    parser.add_argument('--html', default='')
    parser.add_argument('--markdown', default='')
    parser.add_argument('--method-col', default='method')
    parser.add_argument('--error-col', default='rel_fro_error')
    parser.add_argument('--time-col', default='median_seconds')
    parser.add_argument('--scenario-cols', nargs='+', default=['n', 'p', 'df'])
    args = parser.parse_args()

    with open(args.input, newline='') as f:
        rows = list(csv.DictReader(f))
    summary = summarize(rows, args.method_col, args.error_col, args.time_col, args.scenario_cols)
    fieldnames = ['method', 'appearances', 'failures', 'win_rate', 'mean_rank', 'median_error', 'mean_error', 'median_seconds']
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(summary)

    if args.csv:
        write_csv(Path(args.csv), summary, fieldnames)
    if args.markdown:
        write_markdown(Path(args.markdown), summary, fieldnames)
    if args.html:
        write_html(Path(args.html), summary, fieldnames)
