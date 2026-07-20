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


## Minimum dependency testing

The normal development environment uses current compatible dependencies. Before
changing numerical or packaging code, also test the oldest supported dependency
set for the active Python interpreter:

```bash
python -m pip install -U pip scikit-build-core pybind11 pytest
python -m pip install --only-binary=:all: -r requirements/minimum.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
```

## API compatibility

Public API changes must follow [`docs/api_stability.rst`](docs/api_stability.rst).
Add deprecations and removals to `CHANGELOG.md`, including the replacement and
planned removal release when applicable.
