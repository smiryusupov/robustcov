# API stability

This is still an MVP/prototype package.

## Relatively stable

- `FastMCD`
- `TylerShape`
- `RegularizedTyler`
- `StudentTScatter`
- `RegularizedCauchy`
- robust distance plotting helpers
- benchmark CSV scripts

## Experimental

- `AutoRobustScatter` scoring details
- `HellingerRegularizedTyler`
- exact naming of KL/Wiesel Tyler aliases
- automatic threshold/tail calibration

## Not prioritized now

- `MinVolumeEllipsoid` / MVE

MVE is not removed forever, but it is not the strongest differentiator for this project. The current evidence points to a better niche: efficient FastMCD for classical contamination plus regularized Cauchy/Student-t scatter for small-sample heavy-tail problems.
