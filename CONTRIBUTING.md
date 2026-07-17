# Contributing

Thank you for improving `robustcov`.

## Development setup

```bash
git clone https://github.com/smiryusupov/robustcov.git
cd robustcov
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e ".[dev,docs,examples]"
```

## Tests

```bash
python -m compileall -q robustcov tests examples benchmarks docs
python -m pytest -q
```

## Documentation

Read [`docs/WRITING_STYLE.md`](docs/WRITING_STYLE.md) before adding a guide or
gallery page. Build the documentation with warnings treated as errors:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```
