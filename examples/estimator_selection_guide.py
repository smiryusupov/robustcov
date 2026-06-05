"""Print a practical estimator-selection guide."""

if __name__ == "__main__":
    print("""
robustcov estimator selection guide

FastMCD
  Use when: n is comfortably larger than p and you expect separable outliers.
  Strength: fast robust covariance and support/outlier diagnostics.

RegularizedTyler / KLRegularizedTyler / WieselTyler
  Use when: data are heavy-tailed, p is close to n, or p > n.
  Strength: stable shape estimation with shrinkage.

StudentTScatter
  Use when: you want a smooth heavy-tailed covariance-like M-estimator with fixed df.
  Strength: less hard rejection than MCD; useful for diffuse heavy tails.

RegularizedCauchy
  Use when: very heavy tails and small samples are expected.
  Strength: strong radial downweighting plus shrinkage.

HellingerRegularizedTyler
  Experimental. Uses Tyler weights with square-root-space shrinkage.
  Use only for exploratory comparisons until the exact objective/update is finalized.
""")
