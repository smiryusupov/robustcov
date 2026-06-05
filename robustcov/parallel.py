"""Thread-control helpers for the optional OpenMP C++ backend."""

from __future__ import annotations

from contextlib import contextmanager

from . import _robustcov_cpp as _cpp


def has_openmp() -> bool:
    """Return True when the C++ extension was compiled with OpenMP support."""
    return bool(_cpp.has_openmp())


def get_num_threads() -> int:
    """Return the current maximum number of OpenMP threads used by C++ kernels."""
    return int(_cpp.get_num_threads())


def set_num_threads(n_threads: int) -> None:
    """Set the maximum number of OpenMP threads used by C++ kernels.

    The setting is process-global, matching OpenMP's native behavior. Builds without
    OpenMP accept this call but continue to run serially.
    """
    n_threads = int(n_threads)
    if n_threads < 1:
        raise ValueError("n_threads must be >= 1")
    _cpp.set_num_threads(n_threads)


@contextmanager
def thread_limit(n_threads: int | None):
    """Temporarily set the C++ OpenMP thread limit inside a ``with`` block."""
    if n_threads is None:
        yield
        return
    old = get_num_threads()
    set_num_threads(int(n_threads))
    try:
        yield
    finally:
        set_num_threads(old)
