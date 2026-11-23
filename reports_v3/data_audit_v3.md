# V3 数据审计报告（面向新网络的数据契约）

- 数据目录：`D:\ProjectDirectory\PathSafe-Ellipse\data\processed_v3\trajectories\train_v3`
- 审计文件数：50648；场景 17038；`map_group` 30
- 查询点总数：1487728；每轨迹 N：{"mean": 29.37387458537356, "p05": 10.0, "p50": 24.0, "p95": 66.0, "min": 6.0, "max": 382.0}
- 唯一地图数：17038（SDF 只需按地图生成，平均每地图 2.97 条轨迹）
- 已含 `signed_distance_m` 的文件：0 → Task 2 生成独立数据版本

## 1 字段清单（plan 1.1 / 1.3-1）

- 50648 个文件：meta_json, center_offset, centerline_xy, continuity_verified, ellipse_S, ellipse_ab, ellipse_angle, ellipse_area, goal_xy, label_valid, lidar_ranges, occupancy, path_xy, safety_verified, solver_status, start_xy
- 结构问题：{}

## 2 地图与配置（plan 1.3-2）

- `occupancy` 形状：[(256, 256)]
- `crop_px` [192, 256, 320] → 与 256 相同，无未被覆盖的像素（`valid_mask` 全 1，见 Task 2 说明）
- `r_scene` [0.1875, 0.25, 0.3125] m/px；`physical_field_m` [48.0, 64.0, 80.0]
- LiDAR：[16] 射线，量程 [9.0] m

## 3 标签一致性（仅诊断用，训练不使用中心/尺寸/朝向标签）

- `label_valid` 0.9793；`safety_verified` 0.9980；`continuity_verified` 0.9793
- a≥b 最小比例 1.000000；面积 vs πab 最大相对误差 8.72e-13；S 最小特征值 1.000e-06
- 退化椭圆（a<1 mm）平均占比 0.0000

## 4 世界坐标 ↔ 像素坐标（plan 1.3-5）

- 约定：cell (row=i, col=j) 中心 = (x, y) = (j·r, i·r)；`pixel = xy / r_scene`，`[row, col]` 索引
- 路径点全部在地图内：最小比例 1.000000
- 路径点落在 blocked 格：均值 0.00017（有效点均值 0.00006）
- 重采样间距 [m]：{"mean": 1.422252738628479, "p05": 0.7912629388921038, "p50": 1.425933122198769, "p95": 1.9397972621701631, "min": 0.2757202460586533, "max": 2.0}；min 0.000 / max 2.000
- 弧长 `path_s` 单调：True

## 5 LiDAR 射线（plan 1.3-6）

- 存储约定：`angle_k = 2πk/16`，方向 `(cos, sin)` 与 `path_xy` 同一米制坐标系
- 真实命中比例：均值 0.3380（p05 0.0000），其余为量程截断 → 必须用 `ray_valid` 区分
- 命中射线终点落在 blocked 格：最低 0.0000，均值 0.0977
- 命中射线中点落在 free 格：最低 0.7460
- 独立重投影校验（同一 DDA 重算 16 射线）：{"pairs": 2304, "max_abs_err_m": 4.731730669504941e-07, "mean_abs_err_m": 1.3884526650359301e-08, "frac_gt_1mm": 0.0}

## 6 距离场（Task 2）

- 256×256 米制有符号距离场：范围 [-45.375, 39.219] m
- 各图 p01 分布：{"mean": -5.717908708946934, "p05": -12.605046224594116, "p50": -4.625, "p95": -2.1364415287971505, "min": -41.625, "max": 0.15625}
- 各图 p99 分布：{"mean": 12.048487462940344, "p05": 5.34375, "p50": 10.790192127227783, "p95": 22.34375, "min": 1.40625, "max": 35.46875}
- 符号与 blocked 掩码不一致比例 ≤ 0.0000
- A* 路径点处 SDF：{"mean": 3.130890387554306, "p05": 0.125, "p50": 1.2320752143859863, "p95": 12.125, "min": -0.15625, "max": 35.46875}；其中 <0（路径点被判进障碍）比例 0.00027（2500 个抽样文件）

## 7 样本可视化（plan 1.3-3）

- `audit_00_Berlin_0_1024_part_1_0_c77_234_f48_k1_path_001.png` — Berlin_0_1024_part_1_0_c77_234_f48_k1 (N=30)
- `audit_01_Berlin_1_1024_part_0_1_c154_22_f80_k1_path_000.png` — Berlin_1_1024_part_0_1_c154_22_f80_k1 (N=28)
- `audit_02_Berlin_2_1024_part_0_1_c174_203_f48_k2_path_001.png` — Berlin_2_1024_part_0_1_c174_203_f48_k2 (N=8)
- `audit_03_Boston_2_1024_part_0_1_c112_70_f48_k2_path_000.png` — Boston_2_1024_part_0_1_c112_70_f48_k2 (N=12)
- `audit_04_Denver_2_1024_part_0_0_c10_35_f80_k2_path_001.png` — Denver_2_1024_part_0_0_c10_35_f80_k2 (N=47)
- `audit_05_London_0_1024_part_0_0_c58_176_f80_k2_path_002.png` — London_0_1024_part_0_0_c58_176_f80_k2 (N=24)
- `audit_06_Moscow_0_1024_part_0_0_c160_49_f80_k2_path_000.png` — Moscow_0_1024_part_0_0_c160_49_f80_k2 (N=33)
- `audit_07_Moscow_0_1024_part_0_1_c27_79_f80_k1_path_000.png` — Moscow_0_1024_part_0_1_c27_79_f80_k1 (N=16)
- `audit_08_Moscow_1_1024_part_0_1_c176_109_f48_k2_path_000.png` — Moscow_1_1024_part_0_1_c176_109_f48_k2 (N=15)
- `audit_09_Moscow_1_256_part_0_0_c198_66_f64_k0_path_002.png` — Moscow_1_256_part_0_0_c198_66_f64_k0 (N=18)
- `audit_10_NewYork_0_1024_part_0_0_c51_42_f64_k1_path_002.png` — NewYork_0_1024_part_0_0_c51_42_f64_k1 (N=28)
- `audit_11_NewYork_0_512_part_0_0_c167_122_f64_k2_path_001.png` — NewYork_0_512_part_0_0_c167_122_f64_k2 (N=44)
- `audit_12_NewYork_1_1024_part_0_0_c128_167_f80_k2_path_000.png` — NewYork_1_1024_part_0_0_c128_167_f80_k2 (N=76)
- `audit_13_NewYork_1_1024_part_1_0_c18_44_f48_k1_path_000.png` — NewYork_1_1024_part_1_0_c18_44_f48_k1 (N=67)
- `audit_14_Paris_0_256_part_0_0_c36_252_f64_k0_path_001.png` — Paris_0_256_part_0_0_c36_252_f64_k0 (N=37)
- `audit_15_Shanghai_0_1024_part_0_1_c60_166_f80_k1_path_001.png` — Shanghai_0_1024_part_0_1_c60_166_f80_k1 (N=110)
- `audit_16_Shanghai_1_1024_part_1_0_c25_245_f48_k1_path_002.png` — Shanghai_1_1024_part_1_0_c25_245_f48_k1 (N=18)
- `audit_17_Sydney_0_1024_part_1_0_c111_158_f64_k2_path_002.png` — Sydney_0_1024_part_1_0_c111_158_f64_k2 (N=39)
- `audit_18_Sydney_1_1024_part_0_0_c109_161_f80_k2_path_000.png` — Sydney_1_1024_part_0_0_c109_161_f80_k2 (N=15)
- `audit_19_Sydney_2_1024_part_0_0_c249_209_f64_k1_path_002.png` — Sydney_2_1024_part_0_0_c249_209_f64_k1 (N=23)
