from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = 'robustcov'
author = 'robustcov contributors'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'benchmarks.rst', 'benchmark_report.rst', 'use_cases.rst']
html_theme = 'furo'
html_static_path = ['_static']
autodoc_typehints = 'description'

html_css_files = ['custom.css']
