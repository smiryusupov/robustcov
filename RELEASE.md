# Release checklist

This project publishes CPython wheels for Python 3.12, 3.13, and 3.14 on Ubuntu, Windows, and macOS with GitHub Actions and `cibuildwheel`.

## One-time setup

1. Create the GitHub repository and push the project.
2. Create a PyPI project named `robustcov` from the first release upload.
3. In PyPI, configure a Trusted Publisher for this repository:
   - owner/repository: `smiryusupov/robustcov`
   - workflow: `.github/workflows/wheels.yml`
   - environment: `pypi`
4. In GitHub, create an environment named `pypi`.
5. Import the repository on Read the Docs. Read the Docs will discover the root `.readthedocs.yaml`.

## Release steps

```bash
python -m pip install -U build twine
python -m compileall -q robustcov tests examples benchmarks docs
python -m build
python -m twine check dist/*
```

Then tag and push:

```bash
git tag v0.0.1
git push origin v0.0.1
```

The `wheels` workflow builds the sdist and wheels, checks metadata, and publishes to PyPI only for `v*` tags.

## Conda-forge

After the PyPI release is live, use the PyPI metadata to generate a conda recipe:

```bash
conda create -n grayskull -c conda-forge grayskull conda-build
conda activate grayskull
grayskull pypi robustcov
```

Compare the generated recipe with `conda/recipe/meta.yaml`, replace `REPLACE_WITH_PYPI_SDIST_SHA256`, and submit it to `conda-forge/staged-recipes`.
