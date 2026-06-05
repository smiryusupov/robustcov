import numpy as np
import robustcov as rc

rng = np.random.default_rng(0)
X = rng.standard_t(df=3, size=(400, 5))
X[:30] += 8.0  # contamination

mcd = rc.FastMCD(quality="fast", random_state=42).fit(X)
print("FastMCD location:", np.round(mcd.location_, 3))
print("FastMCD support size:", int(mcd.support_.sum()))
print("FastMCD radial kurtosis:", round(mcd.radial_kurtosis_, 3))

tyler = rc.RegularizedTyler(alpha=0.05, scale_correction="radial_median").fit(X)
print("Reg Tyler scale:", round(tyler.scale_, 3))
print("Reg Tyler radial kurtosis:", round(tyler.radial_kurtosis_, 3))

detector = rc.RobustOutlierDetector(
    estimator=rc.FastMCD(quality="fast", random_state=42),
    threshold="empirical",
    alpha=0.95,
).fit(X)
print("Detected outliers:", int((detector.labels_ == -1).sum()))
