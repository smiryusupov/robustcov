# Contributing

Thank you for improving `robustcov`.

## Development setup

```bash
git clone https://github.com/smiryusupov/robustcov.git
cd robustcov
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -U pip
python -m pip install -e ".[dev,docs,examples]"
```

## Checks before opening a pull request

```bash
python -m compileall -q robustcov tests examples benchmarks docs
pytest -ra
sphinx-build -b html -W docs docs/_build/html
```

The package contains a C++/pybind11 extension built with `scikit-build-core`, so changes to `src/`, `CMakeLists.txt`, or build metadata should be tested through the wheel workflow before release.
