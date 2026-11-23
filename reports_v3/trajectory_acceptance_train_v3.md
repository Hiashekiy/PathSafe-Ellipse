# PathSafe-Ellipse V3 — trajectory dataset (train_v3)

- git commit: `197b161`
- label_version: `v3` (query point = every resampled A* station)
- trajectory dir: `D:\ProjectDirectory\PathSafe-Ellipse\data\processed_v3\trajectories`
- wall clock: 0.0 min (0.00 s / trajectory, 24 shard(s))

## 1. Totals

| metric | value |
|---|---|
| trajectories (NPZ) | 50648 |
| scenarios | 17038 |
| map groups | 30 |
| query points | 1487728 |
| points / trajectory (mean, p05, p95) | 29.4 / 10 / 66 |
| label_valid rate (all points) | 0.9793 |
| safety_verified rate | 0.9980 |
| continuity_verified rate | 0.9793 |
| inserted continuity points | 84261 (0.0566 of all points) |
| total size | 0.42 GB |

## 2. Structure / geometry checks (spec 6.1, 6.2)

| check | result |
|---|---|
| structurally checked files | 50648 |
| files with structural problems | 401 |
| deeply checked files (evenly sampled) | 400 |
| deeply checked files with problems | 16 |
| A* query points in blocked cells | 446 |
| centre-line points in blocked cells | 18 |
| safety_verified ellipses re-verified exactly | 1.0000 |
| continuity pass rate over adjacent points | 0.9912 |
| LiDAR max abs error after recast | 0.00e+00 m |

## 3. Distributions (sampled files, spec 6.4 — no clearance metrics)

| metric | p05 / p50 / p95 |
|---|---|
| ellipse area [m²] | 0.23 / 24.84 / 113.10 |
| semi-major a [m] | 0.41 / 4.12 / 6.00 |
| semi-minor b [m] | 0.19 / 1.99 / 6.00 |
| centre offset [m] | 0.00 / 1.06 / 2.30 |
| LiDAR range [m] | 0.22 / 9.00 / 9.00 |
| solver status | {"optimal": 89699, "failed": 181, "fallback_circle": 19, "verification_failed_no_constrain": 19, "solver_failed:no_feasible_start": 11, "solver_failed:eigen_window": 2} |

## 4. Notes

- one NPZ == one complete trajectory; every per-point array shares one point order
- occupancy is `scene.blocked` (raw obstacles + robot safety inflation), 0 = traversable
- no clearance / patch / dist_m field is stored
- 50648 files, listing: see `index.json`
