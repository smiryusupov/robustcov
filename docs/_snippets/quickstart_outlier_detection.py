import numpy as np
import robustcov as rc

rng = np.random.default_rng(0)

# Heavy-tailed data with injected outliers.
X = rng.standard_t(df=3, size=(400, 5))
X[:30] += 8.0

est = rc.FastMCD(quality="balanced", random_state=42).fit(X)
print(est.location_)
print(est.covariance_)
print(est.radial_kurtosis_)

det = rc.RobustOutlierDetector(
    estimator=rc.FastMCD(quality="balanced", random_state=42),
    contamination=0.075,
).fit(X)
print(det.labels_)
