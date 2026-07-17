# Documentation writing style

The documentation should read like technical notes written by a maintainer, not
like product copy or a repeated report template.

## Start with the task

Open a page by stating what the data are and what the reader is trying to do.
Avoid generic introductions such as “Why this matters” or “This example
demonstrates a valuable capability.”

Prefer:

> Each row is a daily yield-curve change. The example asks whether three stable
> factors can still be recovered after a few maturity quotes are corrupted.

Avoid:

> This important example demonstrates why robust methods add value in finance.

## Use page-specific sections

Do not give every example the same headings. Choose headings that match the
analysis, such as “Factor loadings,” “Inspecting the final batch,” or “When
filtering hurts.” Common utility headings such as “Run the example” and “Source”
are fine.

## Report results without selling them

State the metric, comparison, and relevant tradeoff. Do not call a result
“valuable,” “honest,” “compelling,” or a “win.” Let the evidence carry the
claim.

Prefer:

> LocalOutlierFactor has the highest F1 at 0.90. AutoRobustScatter reaches 0.80
> and has a higher runtime.

Avoid:

> This is a valuable honest example that proves robustcov is competitive.

## Put caveats where they matter

A caveat is clearer next to the choice it affects. Explain threshold limitations
in the threshold section, time-series limitations in the data section, and
clinical limitations near the medical interpretation. A final limitations
section is useful only when several constraints remain.

## Use concrete nouns and verbs

Name the estimator, dataset, metric, and failure mode. Replace phrases such as
“the workflow,” “the solution,” or “this capability” when a more precise noun is
available.

## Keep claims proportional

Synthetic examples show behavior under a known data-generating process. They do
not establish production accuracy. External benchmark pages should include weak
results as well as strong ones and should distinguish statistical quality from
runtime.

## Prefer short paragraphs

Most paragraphs should make one point in two to four sentences. Lists are useful
for parameters, signals, or alternatives, but should not repeat the surrounding
prose.

## Before submitting documentation

Run:

```bash
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Then read the rendered page rather than reviewing only the RST source. Check
that nearby gallery pages do not repeat the same section names and opening
sentences.
