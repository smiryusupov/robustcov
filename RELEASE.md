# Release checklist

The project builds CPython wheels for Python 3.12, 3.13, and 3.14 on Linux,
Windows, and macOS with GitHub Actions and `cibuildwheel`.

## One-time setup

1. Configure the PyPI Trusted Publisher for:
   - owner/repository: `smiryusupov/robustcov`
   - workflow: `.github/workflows/wheels.yml`
   - environment: `pypi`
2. Create a protected GitHub environment named `pypi`.
3. Import the repository on Read the Docs. It uses `.readthedocs.yaml`.

## Local pre-release checks

Start from a clean checkout and an isolated environment:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,docs,examples]"
python scripts/release_check.py
python -m compileall -q robustcov tests examples benchmarks docs scripts
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -m pytest -q
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Test the oldest supported dependencies in a separate environment:

```bash
python -m pip install -U pip scikit-build-core pybind11 pytest
python -m pip install --only-binary=:all: -r requirements/minimum.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
```

Build and inspect fresh distributions:

```bash
rm -rf build dist native-free-wheel
python -m build
python -m twine check dist/*
python scripts/release_check.py dist/*
python scripts/package_smoke_test.py --expect-native yes dist/*

python -m build --wheel \
  -Ccmake.define.ROBUSTCOV_BUILD_NATIVE=OFF \
  --outdir native-free-wheel
python scripts/release_check.py native-free-wheel/*.whl
python scripts/package_smoke_test.py \
  --expect-native no native-free-wheel/*.whl
```

## Release sign-off

- Update `CHANGELOG.md` and remove resolved entries from `Unreleased`.
- Make `pyproject.toml`, `robustcov.__version__`, and `CITATION.cff` agree.
- Confirm that CI is green for Python 3.12–3.14, all supported operating
  systems, minimum dependencies, documentation, sdist, and wheels.
- Save artifact SHA-256 checksums.
- Test installation outside the source tree.
- Review the public API changes against `docs/api_stability.rst`.

## Tag and publish

Replace `X.Y.Z` with the release version:

```bash
git tag -s vX.Y.Z -m "robustcov X.Y.Z"
git push origin vX.Y.Z
```

The `wheels` workflow builds, validates, and publishes only for `v*` tags using
PyPI Trusted Publishing.

## Conda-forge

After the PyPI release is live:

```bash
conda create -n grayskull -c conda-forge grayskull conda-build
conda activate grayskull
grayskull pypi robustcov
```

Compare the generated recipe with `conda/recipe/meta.yaml`, replace the source
SHA-256, and submit it to `conda-forge/staged-recipes`.
