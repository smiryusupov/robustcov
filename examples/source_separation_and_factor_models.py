"""Robust ICA, SOBI, and static factor-model examples."""

import numpy as np

import robustcov as rc

rng = np.random.default_rng(7)
mixing = np.array([[1.0, 0.4, -0.2], [0.2, 1.2, 0.5], [-0.4, 0.3, 0.9]])

n_samples = 2500
independent = np.column_stack(
    [
        rng.laplace(size=n_samples) / np.sqrt(2.0),
        rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=n_samples),
        rng.normal(size=n_samples),
    ]
)
observed = independent @ mixing.T
ica = rc.TwoScatterICA(radial_clip_quantile=0.90).fit(observed)
print("Two-scatter ICA MDI:", rc.minimum_distance_index(ica.unmixing_, mixing))

coefficients = np.array([0.85, -0.55, 0.20])
temporal = np.zeros((n_samples, 3))
innovations = rng.normal(size=(n_samples, 3))
for index in range(1, n_samples):
    temporal[index] = coefficients * temporal[index - 1] + innovations[index]
observed_temporal = temporal @ mixing.T
impulses = rng.choice(n_samples, 40, replace=False)
observed_temporal[impulses] += rng.normal(scale=25.0, size=(impulses.size, 3))
sobi = rc.RobustSOBI(lags=12).fit(observed_temporal)
print("Robust SOBI MDI:", rc.minimum_distance_index(sobi.unmixing_, mixing))

n_features, n_factors = 15, 3
loadings, _ = np.linalg.qr(rng.normal(size=(n_features, n_factors)))
factors = rng.standard_t(4, size=(500, n_factors))
factor_data = factors @ loadings.T + 0.25 * rng.normal(size=(500, n_features))
factor_model = rc.RobustFactorModel(
    n_factors="auto", method="kendall", max_factors=6
).fit(factor_data)
print("Selected factor count:", factor_model.n_factors_)
