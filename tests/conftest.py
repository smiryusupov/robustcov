"""Pytest configuration."""

try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None
else:
    # Use a non-interactive backend in CI/headless environments.
    # Plotting tests use pytest.importorskip when Matplotlib is absent.
    matplotlib.use("Agg", force=True)
