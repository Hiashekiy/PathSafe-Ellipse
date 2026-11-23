# PathSafe-Ellipse V3 — trajectory dataset acceptance (pilot10)

- git commit: `197b161`
- label_version: `v3`  (query point = every resampled A* station)
- config: `configs\data_generation_v3.yaml`
- trajectory dir: `D:\ProjectDirectory\PathSafe-Ellipse\data\processed_v3\trajectories`
- wall clock: 34.2 s (3.42 s / trajectory)

## 1. Structure (spec 6.1)

| check | result |
|---|---|
| files with structural problems | 0 / 10 |
| map shape | (256, 256) uint8, 0 = traversable |
| trajectories (files) | 10 |
| query points | 350 |
| points / trajectory (mean, min, max) | 35.0, 10, 70 |
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
| scenarios (scenes) | 6 |
| map groups | 6 |
| label_valid rate | 1.0000 |
| safety_verified rate | 1.0000 |
| continuity_verified rate | 1.0000 |
| ellipse area p05 / p50 / p95 [m²] | 6.28 / 22.00 / 113.10 |
| semi-major a p50 [m] | 3.90 |
| semi-minor b p50 [m] | 1.89 |
| centre offset p50 / p95 [m] | 1.270 / 2.240 |
| LiDAR range p05 / p50 / p95 [m] | 0.23 / 9.00 / 9.00 |
| solver status counts | {"optimal": 350} |

## 4. Per-file rows

| file | scene | path | N | safe | pairs ok | problems |
|---|---|---|---|---|---|---|
| Sydney_0_1024_part_1_0_c77_216_f48_k0_path_000.npz | Sydney_0_1024_part_1_0_c77_216_f48_k0 | 0 | 36 | 36/36 | 35/35 | — |
| Sydney_0_1024_part_1_0_c77_216_f48_k0_path_001.npz | Sydney_0_1024_part_1_0_c77_216_f48_k0 | 1 | 10 | 10/10 | 9/9 | — |
| Moscow_0_1024_part_0_0_c114_254_f64_k0_path_000.npz | Moscow_0_1024_part_0_0_c114_254_f64_k0 | 0 | 60 | 60/60 | 59/59 | — |
| Moscow_0_1024_part_0_0_c114_254_f64_k0_path_001.npz | Moscow_0_1024_part_0_0_c114_254_f64_k0 | 1 | 34 | 34/34 | 33/33 | — |
| London_0_512_part_0_0_c163_34_f80_k0_path_000.npz | London_0_512_part_0_0_c163_34_f80_k0 | 0 | 43 | 43/43 | 42/42 | — |
| London_0_512_part_0_0_c163_34_f80_k0_path_001.npz | London_0_512_part_0_0_c163_34_f80_k0 | 1 | 17 | 17/17 | 16/16 | — |
| Berlin_0_1024_part_1_1_c202_2_f48_k0_path_000.npz | Berlin_0_1024_part_1_1_c202_2_f48_k0 | 0 | 24 | 24/24 | 23/23 | — |
| Berlin_0_1024_part_1_1_c202_2_f48_k0_path_001.npz | Berlin_0_1024_part_1_1_c202_2_f48_k0 | 1 | 21 | 21/21 | 20/20 | — |
| Berlin_2_256_part_0_0_c144_99_f64_k0_path_000.npz | Berlin_2_256_part_0_0_c144_99_f64_k0 | 0 | 70 | 70/70 | 69/69 | — |
| Moscow_2_1024_part_1_1_c53_153_f80_k0_path_000.npz | Moscow_2_1024_part_1_1_c53_153_f80_k0 | 0 | 35 | 35/35 | 34/34 | — |

## 5. Files

- `Sydney_0_1024_part_1_0_c77_216_f48_k0_path_000.npz`
- `Sydney_0_1024_part_1_0_c77_216_f48_k0_path_001.npz`
- `Moscow_0_1024_part_0_0_c114_254_f64_k0_path_000.npz`
- `Moscow_0_1024_part_0_0_c114_254_f64_k0_path_001.npz`
- `London_0_512_part_0_0_c163_34_f80_k0_path_000.npz`
- `London_0_512_part_0_0_c163_34_f80_k0_path_001.npz`
- `Berlin_0_1024_part_1_1_c202_2_f48_k0_path_000.npz`
- `Berlin_0_1024_part_1_1_c202_2_f48_k0_path_001.npz`
- `Berlin_2_256_part_0_0_c144_99_f64_k0_path_000.npz`
- `Moscow_2_1024_part_1_1_c53_153_f80_k0_path_000.npz`
