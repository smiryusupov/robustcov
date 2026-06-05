# Conda packaging

`robustcov` is first published to PyPI as wheels and an sdist. Conda users can install those wheels inside a conda environment immediately:

```bash
conda create -n robustcov python=3.12 pip
conda activate robustcov
python -m pip install robustcov
```

For a native `conda install -c conda-forge robustcov` package, submit a recipe to `conda-forge/staged-recipes` after the PyPI release exists.

Recommended flow:

```bash
conda create -n grayskull -c conda-forge grayskull conda-build
conda activate grayskull
grayskull pypi robustcov
```

Then compare the generated recipe with `conda/recipe/meta.yaml`, fill in the PyPI sdist SHA256, and submit it to conda-forge. The package metadata in `pyproject.toml` is deliberately complete enough for Grayskull to generate most of the recipe automatically.
