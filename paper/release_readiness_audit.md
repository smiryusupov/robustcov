# Release-readiness audit

This file records checks before a stable release and future JOSS submission.

## Local checks

- [ ] `python -m compileall -q robustcov tests examples benchmarks docs`
- [ ] `pytest -ra`
- [ ] `python -m sphinx -b html -W --keep-going docs docs/_build/html`
- [x] `python -m build`
- [x] `python -m twine check dist/*`

## Repository metadata

- [ ] `LICENSE` present
- [ ] `README.md` explains value proposition
- [ ] `CITATION.cff` present
- [ ] `CONTRIBUTING.md` present
- [ ] `CHANGELOG.md` present
- [ ] `paper/paper.md` present
- [ ] `paper/paper.bib` present
- [ ] `paper/benchmark_claims.md` present

## Before JOSS submission

- [ ] Fill paper author affiliation
- [ ] Fill paper ORCID or remove ORCID field
- [ ] Fill paper date
- [ ] Replace acknowledgement TODO
- [ ] Confirm package version
- [ ] Create tagged release
- [ ] Archive release if needed
