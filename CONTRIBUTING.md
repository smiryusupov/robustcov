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
python -m pytest -q -m "unit or integration or native"
python -m pytest -q -m statistical --durations=20
python -m pytest -q -m benchmark --durations=20
python -m pytest -q -m packaging --durations=20
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
python -m pytest -q -m "unit or integration or native"
```

## API compatibility

Public API changes must follow [`docs/api_stability.rst`](docs/api_stability.rst).
At the package root and in `robustcov.experimental`, only names listed in the
namespace's `__all__` are supported exports. Import implementation helpers under
underscore-prefixed aliases so they do not become accidental public bindings.
Add deprecations and removals to `CHANGELOG.md`, including the replacement and
planned removal release when applicable.

## Adding or changing a public method

RobustCov distinguishes literature implementations, package-specific composites,
and experimental research interfaces. A public method change must keep that
provenance visible without adding defensive language to user-facing guides.

Before opening a pull request for a new estimator or algorithm:

1. Add its canonical entry to `robustcov.provenance.METHOD_PROVENANCE`.
2. Add primary references to `REFERENCE_CATALOG` and `docs/references.bib`.
3. State whether the implementation reproduces a published algorithm, adapts it,
   or composes established components.
4. Document assumptions, deviations from the paper, and unavailable guarantees.
5. Add focused tests and benchmark, validation, performance-gate, or workflow
   ownership.
6. Classify every new top-level symbol in `robustcov/_public_api.json`.
7. Keep benchmark claims scoped to the documented data-generating process,
   contamination model, and comparison set.

Experimental methods belong in `robustcov.experimental` unless compatibility or
a deliberate transition requires a temporary top-level export. Experimental
status must be visible in the API documentation and changelog.

## Release-facing language

Use direct, factual wording. Cite the original method and describe RobustCov's
implementation choices. Avoid universal state-of-the-art claims unless a
reviewed benchmark and its scope support that exact statement.
