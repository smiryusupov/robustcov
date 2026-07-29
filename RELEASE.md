# Release checklist

RobustCov publishes CPython wheels for Python 3.12, 3.13, and 3.14 on Linux,
Windows, and macOS. Release distributions are built once, checked, and then
published through short-lived OpenID Connect credentials.

## One-time publishing setup

Configure two Trusted Publishers for `.github/workflows/wheels.yml`:

1. **TestPyPI**
   - owner/repository: `smiryusupov/robustcov`
   - workflow: `wheels.yml`
   - environment: `testpypi`
2. **PyPI**
   - owner/repository: `smiryusupov/robustcov`
   - workflow: `wheels.yml`
   - environment: `pypi`

Create matching GitHub environments. Protect `pypi` with a required reviewer
and restrict it to release tags. A manual approval is optional for `testpypi`.
Do not store PyPI API tokens in repository secrets.

Import the repository on Read the Docs; it uses `.readthedocs.yaml`.

## Prepare the release commit

The alpha release version is declared in four places and checked automatically:

```bash
python scripts/check_release_version.py --expected 0.1.0
```

Before generating public benchmark snapshots, commit the implementation so the
snapshot publisher can record a clean source commit.

Regenerate, review, and publish the case studies one at a time. Publishing a
snapshot changes tracked documentation files, so commit FD002 before publishing
FD004; the publisher intentionally refuses to create reviewed evidence from a
dirty working tree.

```bash
python examples_external/cmapss_dro_pca_monitoring.py \
  --download --subset FD002 --outdir results/external/cmapss_fd002
python scripts/publish_external_snapshot.py publish cmapss_fd002 \
  --results results/external/cmapss_fd002 \
  --command "python examples_external/cmapss_dro_pca_monitoring.py --download --subset FD002 --outdir results/external/cmapss_fd002" \
  --replace
git add docs/_static/external_results docs/_generated docs/external_results
git commit -m "Publish reviewed C-MAPSS FD002 snapshot"

python examples_external/cmapss_dro_pca_monitoring.py \
  --subset FD004 --outdir results/external/cmapss_fd004
python scripts/publish_external_snapshot.py publish cmapss_fd004 \
  --results results/external/cmapss_fd004 \
  --command "python examples_external/cmapss_dro_pca_monitoring.py --subset FD004 --outdir results/external/cmapss_fd004" \
  --replace
git add docs/_static/external_results docs/_generated docs/external_results
git commit -m "Publish reviewed C-MAPSS FD004 snapshot"
```

## Local release-candidate checks

Start from a clean checkout and isolated environment:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev,docs,examples]"
python scripts/check_release_version.py --require-prerelease
python scripts/release_check.py --release-candidate
python -m compileall -q robustcov tests examples benchmarks docs scripts
```

Run the grouped test suites independently:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -m pytest -q -m "unit or integration or native" --durations=20
python -m pytest -q -m statistical --durations=20
python -m pytest -q -m benchmark --durations=20
python -m pytest -q -m packaging --durations=20
```

Build the documentation strictly:

```bash
rm -rf docs/_build
python -m sphinx -j 4 -E -a -W --keep-going \
  -b html docs docs/_build/html
```

Test the oldest supported dependencies in a separate environment:

```bash
python -m pip install -U pip scikit-build-core pybind11 pytest
python -m pip install --only-binary=:all: -r requirements/minimum.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
```

## Build and inspect local artifacts

```bash
rm -rf build dist native-free-wheel release-metadata
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

python scripts/write_artifact_checksums.py \
  dist/* native-free-wheel/*.whl \
  --output release-metadata/SHA256SUMS
```

## TestPyPI rehearsal

From GitHub Actions, run **release distributions** manually and select
`publish_target=testpypi`.

The workflow:

1. builds the same sdist and platform wheels used by production;
2. runs the release-candidate evidence and archive checks;
3. publishes through the `testpypi` Trusted Publisher;
4. waits for index visibility;
5. installs the package outside the checkout; and
6. runs covariance, PCA, PCP, conformal-calibration, and native smoke tests.

Do not reuse the same version for repeated TestPyPI uploads. Increment the alpha
serial, for example from `0.1.0a2` to `0.1.0a3`, when a rehearsal must be
repeated after publication.

## Production tag and publish

After the TestPyPI rehearsal succeeds and all required CI jobs are green:

```bash
python scripts/check_release_version.py --tag v0.1.0
git tag -a v0.1.0 -m "robustcov 0.1.0"
git push origin v0.1.0
```

The tag-triggered workflow verifies that the tag matches the package version,
rebuilds and checks the distributions, and publishes through the protected
`pypi` environment. Trusted Publishing generates package attestations by
default.

Download and retain the `release-metadata` workflow artifact containing
`SHA256SUMS`.

## Conda-forge

After the PyPI release is live:

```bash
conda create -n grayskull -c conda-forge grayskull conda-build
conda activate grayskull
grayskull pypi robustcov
```

Compare the generated recipe with `conda/recipe/meta.yaml`, replace the source
SHA-256, and submit it to `conda-forge/staged-recipes`.
