# Copyright 2026 Shohruh Miryusupov
# SPDX-License-Identifier: Apache-2.0

"""Thread-control helpers for the optional OpenMP C++ backend."""

from __future__ import annotations

from contextlib import contextmanager

from ._native import native_available, require_native


def has_openmp() -> bool:
    """Return whether the compiled extension has OpenMP support.

    A native-free installation returns ``False`` rather than failing at import time.
    """
    if not native_available():
        return False
    return bool(require_native("has_openmp").has_openmp())


def get_num_threads() -> int:
    """Return the active native thread limit.

    Native-free installations report one thread because NumPy fallbacks remain
    serial from robustcov's point of view.
    """
    if not native_available():
        return 1
    return int(require_native("get_num_threads").get_num_threads())


def set_num_threads(n_threads: int) -> None:
    """Set the maximum number of OpenMP threads used by native kernels.

    The setting is process-global, matching OpenMP's native behavior. In a
    native-free installation, requesting one thread is a harmless no-op and
    requesting more than one thread raises an actionable error.
    """
    n_threads = int(n_threads)
    if n_threads < 1:
        raise ValueError("n_threads must be >= 1")
    if not native_available():
        if n_threads == 1:
            return
        raise RuntimeError(
            "Cannot set more than one native thread because the robustcov "
            "native extension is unavailable"
        )
    require_native("set_num_threads").set_num_threads(n_threads)


@contextmanager
def thread_limit(n_threads: int | None):
    """Temporarily set the native OpenMP thread limit inside a ``with`` block."""
    if n_threads is None:
        yield
        return
    old = get_num_threads()
    set_num_threads(int(n_threads))
    try:
        yield
    finally:
        set_num_threads(old)
