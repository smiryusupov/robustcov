# Dependency test sets

`minimum.txt` pins the oldest NumPy, SciPy, and scikit-learn versions tested for
each supported CPython version. Runtime metadata in `pyproject.toml` must not
claim lower versions than this file.

To test the minimum dependency set for the active Python interpreter:

```bash
python -m pip install -U pip scikit-build-core pybind11 pytest
python -m pip install -r requirements/minimum.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m pytest -q
```

Use binary wheels in CI so the job checks package compatibility rather than
spending time compiling the scientific Python stack:

```bash
python -m pip install --only-binary=:all: -r requirements/minimum.txt
```
