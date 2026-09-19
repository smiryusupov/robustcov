import numpy as np
import robustcov as rc

rng = np.random.default_rng(0)

# Heavy-tailed data with injected row outliers.
X = rng.standard_t(df=3, size=(400, 5))
X[:30] += 8.0

det = rc.RobustOutlierDetector(
    estimator=rc.FastMCD(quality="balanced", random_state=42),
    contamination=0.075,
).fit(X)

print(det.estimator_.location_)
print(det.estimator_.covariance_)
print(det.labels_)
