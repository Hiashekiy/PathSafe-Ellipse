# PathSafe-Ellipse V3 — trajectory dataset acceptance (phase1)

- git commit: `197b161`
- label_version: `v3`  (query point = every resampled A* station)
- config: `configs\data_generation_v3.yaml`
- trajectory dir: `D:\ProjectDirectory\PathSafe-Ellipse\data\processed_v3\trajectories`
- wall clock: 1.4 s (1.44 s / trajectory)

## 1. Structure (spec 6.1)

| check | result |
|---|---|
| files with structural problems | 0 / 1 |
| map shape | (256, 256) uint8, 0 = traversable |
| trajectories (files) | 1 |
| query points | 36 |
| points / trajectory (mean, min, max) | 36.0, 36, 36 |
| center_offset == centerline - path | PASS |

## 2. Geometry (spec 6.2)

| check | result |
|---|---|
| A* query points in blocked cells | 0 |
| centre-line points in blocked cells | 0 |
| safety_verified ellipses re-verified exactly | 1.0000 |
| continuity pass rate over adjacent query points | 1.0000 |
| LiDAR max abs error after recast | 0.00e+00 m |

## 3. Label statistics (spec 6.4, no clearance metrics)

| metric | value |
|---|---|
| scenarios (scenes) | 1 |
| map groups | 1 |
| label_valid rate | 1.0000 |
| safety_verified rate | 1.0000 |
| continuity_verified rate | 1.0000 |
| ellipse area p05 / p50 / p95 [m²] | 8.25 / 12.58 / 65.90 |
| semi-major a p50 [m] | 2.74 |
| semi-minor b p50 [m] | 1.41 |
| centre offset p50 / p95 [m] | 1.342 / 2.194 |
| LiDAR range p05 / p50 / p95 [m] | 0.13 / 9.00 / 9.00 |
| solver status counts | {"optimal": 36} |

## 4. Per-file rows

| file | scene | path | N | safe | pairs ok | problems |
|---|---|---|---|---|---|---|
| Sydney_0_1024_part_1_0_c77_216_f48_k0_path_000.npz | Sydney_0_1024_part_1_0_c77_216_f48_k0 | 0 | 36 | 36/36 | 35/35 | — |

## 5. Files

- `Sydney_0_1024_part_1_0_c77_216_f48_k0_path_000.npz`
