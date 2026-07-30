# Temporary regression tests

Tests in this directory protect narrowly scoped fixes while the affected API is
being stabilized. They are intentionally small and readable.

Remove `test_fast_mcd_diagnostics_regression.py` after the FastMCD diagnostic
contract has survived one release cycle and equivalent assertions have been
folded into the permanent estimator contract tests.

Remove `test_fast_mcd_c_step_reference.py` after the native C-step has been
refactored into independently testable production components or the same
reference comparison has been folded into the permanent native-kernel tests.
