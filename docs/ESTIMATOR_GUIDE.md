# Which estimator should I use?

This project now has two main estimator families.

## Classical contamination / separable outliers

Use `FastMCD` when `n` is comfortably larger than `p` and you expect a clean central subset plus separable outliers.

```python
rc.FastMCD(quality="balanced", random_state=0)
```

This is the replacement for adding an MVE implementation right now: MVE is historically important, but it is expensive and less central to the package's speed-focused niche. FastMCD plus diagnostics is more useful for the current roadmap.

## Small sample, high dimension, heavy tails

Use `RegularizedCauchy` or `StudentTScatter` when samples are few, tails are heavy, or `p` is close to or larger than `n`.

```python
rc.RegularizedCauchy(alpha=0.10)
rc.StudentTScatter(df=3, alpha=0.05)
```

These are often better than empirical covariance, Ledoit-Wolf/OAS, and MCD when the data are very heavy tailed or rank deficient.

## Shape-only heavy-tail estimation

Use `RegularizedTyler` when you want robust shape rather than covariance scale.

```python
rc.RegularizedTyler(alpha=0.10, scale_correction="radial_median")
```

`KLRegularizedTyler` and `WieselTyler` are currently documented aliases around the same regularized Tyler prototype. Keep this explicit until a separate exact objective/update is implemented.

## Auto selection

Use `AutoRobustScatter(selection="diagnostic")` for a quick unsupervised choice among robust scatter estimators. Use `selection="stability"` when you can afford split-sample refitting.

```python
auto = rc.AutoRobustScatter(selection="diagnostic").fit(X)
```

Auto selection is a diagnostic heuristic, not an oracle. Benchmark it when ground truth is available.

## Experimental estimators

`HellingerRegularizedTyler` is experimental. It is useful for exploratory comparisons but should not be described as a finalized published Hellinger optimizer yet.
