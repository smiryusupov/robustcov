"""Pytest configuration."""

import matplotlib

# Use a non-interactive backend in CI/headless environments.
# This avoids Tk/Tcl backend failures on Windows runners.
matplotlib.use("Agg", force=True)
