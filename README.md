# PathSafe-Ellipse

面向搜索路径的学习式安全椭圆生成 —— 数据生成 + 网络训练/测试。

本项目是**独立工程**：不依赖、不修改 `Neural-IRIS` / `TrajSafe-Diffuser` 的源码，
只把 `../Neural-IRIS`（同级目录）当作**只读参考**（地图解析、局部裁剪、膨胀、
边界提取、IRIS 求解思路、NPZ 组织）。

---

## 1. 环境

已存在的 Conda 环境（未新建、未改动）：

```powershell
E:\CondaEnvData\envs\GGMPC\python.exe
```

实测版本（`scripts/inspect_maps.py` 会打印并写入 `reports/map_inventory.json`）：

| 组件 | 版本 |
|---|---|
| Python | 3.10.19 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| CVXPY | 1.7.5（可用求解器 CLARABEL / SCS / OSQP / SCIPY） |
| CLARABEL | 0.11.1 |
| SCS | 3.2.11 |
| PyTorch | 2.9.1+cu126（RTX 4070，CUDA 可用） |
| Matplotlib | 3.10.7 |

所有依赖都已存在，**不需要安装任何新包**（`requirements.txt` 仅作记录）。

---

## 2. 目录

```text
PathSafe-Ellipse/
├── configs/data_generation.yaml     # 全部参数（无硬编码）
├── data/
│   ├── street-map/{train,val}/      # 用户提供的原始 .map（只读，不改）
│   ├── processed/
│   │   ├── map_metadata/            #（见 reports/map_inventory.csv）
│   │   └── inflated_maps/           # 膨胀地图缓存（含指纹，避免重复膨胀）
│   ├── intermediate/paths/          # A* 路径（每图一个 npz）
│   ├── intermediate/samples/        # 路径点级中间样本（每图一个 npz）
│   ├── labels/map_index.json        # map_id ↔ 地图文件名
│   └── splits/{train,val,test}.npz  # 最终数据集
├── src/pathsafe_ellipse/
│   ├── map_io.py                    # .map 解析（严格校验）+ 地图组 ID
│   ├── map_preprocess.py            # 3×3 膨胀、距离变换、空洞缓存
│   ├── path_search.py               # 8 邻域 A*（禁止穿角）+ 起终点采样
│   ├── path_sampling.py             # 弧长稀疏采样 + 转弯点 + 切向
│   ├── local_features.py            # 128×128 patch、supercover 射线、局部净空
│   ├── candidate_centers.py         # 16 射线 × 3 偏置比例候选中心
│   ├── mvie.py                      # 固定中心最大体积椭圆（barrier + cvxpy/CLARABEL）
│   ├── ellipse_solver.py            # 固定中心 IRIS 迭代（累积切面）
│   ├── ellipse_validation.py        # ρ≤η 包含性 + 连续几何安全判定（精确）
│   ├── dataset_builder.py           # 主流程（paths / samples 两个 worker）
│   ├── dataset_filter.py            # 质量过滤 + 原因计数
│   ├── dataset_split.py             # 地图组级划分
│   ├── visualization.py             # 地图/样本/路径可视化
│   ├── model.py                     # 椭圆回归网络
│   ├── torch_dataset.py             # 位压缩 NPZ → 内存数据集
│   └── evaluation.py                # 回归 + 几何指标（IoU / 包含 / 安全）
├── scripts/
│   ├── inspect_maps.py              # S0/S1：环境、地图清单、膨胀
│   ├── test_geometry.py             # 几何内核自检（25 项）
│   ├── generate_dataset.py          # S2-S5：paths → samples → filter → split → NPZ
│   ├── validate_dataset.py          # 独立复验 + 质量报告
│   ├── visualize_samples.py          # S6：地图/样本/路径可视化
│   ├── make_report.py               # 统计报告（分子/分母）
│   ├── train_ellipse_net.py         # 训练
│   ├── evaluate_ellipse_net.py      # 测试 + 预测可视化
│   └── bench_micro.py / profile_samples.py / bench_solver.py  # 开发用基准
├── outputs/{previews,path_visualizations,ellipse_visualizations,checkpoints}/
├── reports/                          # 所有报告（json + md + png）
└── logs/
```

---

## 3. 运行顺序

```powershell
$py = "E:\CondaEnvData\envs\GGMPC\python.exe"

# S0/S1 环境 + 地图清单 + 膨胀缓存
& $py scripts\inspect_maps.py

# 几何内核自检（必须先通过）
& $py scripts\test_geometry.py

# S2-S5 数据生成（小规模先验证链路）
& $py scripts\generate_dataset.py --max-maps 3 --paths-per-map 6 --workers 1 --tag _dev

# 全量：10 个分片并行（文件系统分片，无 IPC）
powershell -File scripts\launch_shards.ps1 -Shards 10 -PathsPerMap 30
& $py scripts\generate_dataset.py --stage assemble     # 过滤 + 划分 + NPZ

# 独立复验 + 报告 + 可视化
& $py scripts\validate_dataset.py
& $py scripts\make_report.py
& $py scripts\visualize_samples.py

# 训练 + 测试
& $py scripts\train_ellipse_net.py --epochs 40 --batch-size 256
& $py scripts\evaluate_ellipse_net.py --split test
```

> 分片采用**文件系统级并行**（每个地图写自己的 NPZ），因为本机 Windows 沙箱
> 禁止命名管道，`ProcessPoolExecutor` / `multiprocessing.Pool` 不可用。

---

## 4. 关键约定

* **坐标**：全程 `(x, y) = (col, row)`；栅格 `(row=i, col=j)` 的中心位于连续坐标
  `(x=j, y=i)`；patch 中路径点恒为连续点 `(64.0, 64.0)`。
* **单位**：cell 与 m 显式区分；`ellipse_ab`、`ellipse_center_offset`、`ellipse_area`
  以 **m / m²** 存储，`ray_distances` 归一化到 `[0,1]`，`clearance` 为 cell。
* **膨胀**：3×3 方形结构元素膨胀一次，`occupancy_inflated` 是 A*、patch、射线、
  椭圆优化、最终安全判定**唯一**使用的地图。
* **椭圆表示**：`E(c, L) = {c + Lz : ‖z‖ ≤ 1}`，`L` 对称正定，特征值为半轴，
  特征向量为主轴方向；半轴按 `a ≥ b` 排列，长轴方向向量按 `cos φ ≥ 0` 规范化，
  消除 π 周期歧义；`a/b < 1.02` 时 `angle_valid = 0`。
* **数据划分**：地图组级（同一母地图的所有 `_part_` 分块同组），文件
  `data/splits/map_group_split.json`，复现时复用同一划分。

---

## 5. 相对实施文档的取舍（均有记录与理由）

| 项 | 文档 | 本项目 | 理由 |
|---|---|---|---|
| 每图路径数 | 100 | **30** | 180 图 ×100 路径 ≈ 72 万样本、离线椭圆求解算力需求过大；30 路径/图 ≈ 34 万样本，一夜可完成且足够训练。参数在 YAML 中可改回 100 |
| 数据集划分 | 优先地图组 | **地图组**（30 组 → 24/3/3） | 用户给的 `train/val` 目录存在**同组跨集**（如 `Berlin_1_1024_part_0_0` 在 val 而 `part_0_1` 在 train），直接使用会造成泄漏，故重新按组划分 |
| MVIE 求解器 | CLARABEL 优先，SCS 备用 | **barrier 内点（生产）+ CLARABEL/SCS（校验与回退）** | cvxpy 每次改参数都要重新编译（~80 ms/解），算力上不可行；自研 3 变量 log-barrier 与 CLARABEL 解的 `log det`/半轴一致到 1e-3（`reports/` 记录复验），CLARABEL 仍在 barrier 失败时自动接管 |
| 障碍边界点云 | 边界点 + jitter | **障碍掩膜 1 格膨胀后取角点** | 与 jitter 同样保守（避免切面"切过栅格内部"），但点数少 ~8 倍，稠密地图上速度提升约 4× |
| 椭圆尺寸上限 | 未指定 | 受 128×128 patch 窗口约束（≤ 63.5 cells ≈ 15.9 m） | 标签定义为"局部可视窗口内的最大安全内接椭圆"；窗口外无观测，不能声明安全。报告中明确说明该饱和效应 |
| 安全验收 | 允许超采样近似 | **连续精确判定**：对 bbox 内每个障碍栅格方形解析求二次型最小值 | 文档允许两种，本项目实现严格版本，无近似 |

---

## 6. 数据集字段

`data/splits/{train,val,test}.npz`

| 字段 | 类型 | 形状 | 含义 |
|---|---|---|---|
| `map_patch` | uint8 | (N,128,128) | 膨胀局部占据图（网络输入） |
| `map_patch_packed` | uint8 | (N,128,16) | 同上按位打包（训练时省内存） |
| `ray_distances` | float32 | (N,16) | 归一化测距 ∈[0,1]，未命中=1 |
| `ray_distances_raw` | float32 | (N,16) | 未归一化测距（cell） |
| `ray_hits` | uint8 | (N,16) | 是否命中障碍 |
| `path_tangent` | float32 | (N,2) | 路径切向 |
| `offset_direction` | int64 | (N,) | 0–15 射线编号；16 = 不偏置 |
| `offset_ratio` | float32 | (N,) | 偏置比例 λ |
| `ellipse_ab` | float32 | (N,2) | 长/短半轴 (m)，a ≥ b |
| `ellipse_angle` | float32 | (N,2) | (sin φ, cos φ) |
| `ellipse_center_offset` | float32 | (N,2) | 椭圆中心相对路径点偏移 (m) |
| `ellipse_area` | float32 | (N,) | 面积 (m²) |
| `point_inside_metric` | float32 | (N,) | ρ = ‖L⁻¹(p−c)‖ ≤ 0.9 |
| `angle_valid` | uint8 | (N,) | 近圆时方向角不可靠 → 0 |
| `label_valid` | uint8 | (N,) | 通过全部验收 |
| `point_xy` | int32 | (N,2) | 路径点全局栅格坐标 |
| `map_id` / `path_id` / `point_id` | int32 | (N,) | 溯源编号（`map_id` → `data/labels/map_index.json`） |
| `clearance` | float32 | (N,) | 路径点局部净空（cell） |
| `n_candidates` / `n_candidates_valid` | int32 | (N,) | 该点候选数与通过验收的候选数 |
| `ellipse_iters` / `ellipse_converged` | int32/uint8 | (N,) | IRIS 迭代次数/是否收敛 |
| `clearance_class` | uint8 | (N,) | 0 紧凑(1–3)、1 中等(3–8)、2 开阔(>8) cell |

---

## 7. 结果

**数据**：180 张地图 → 5400 条 A* 路径 → **326 406** 个有效路径点样本
（train 261 349 / val 31 567 / test 33 490，按 24/3/3 个地图组划分）。
独立复验（每 split 抽 4000 例，从 NPZ 重建几何）：椭圆与膨胀障碍**严格无交集** 4000/4000、
路径点 ρ ≤ 0.9 4000/4000、射线特征重投偏差 0.0、无地图组/路径跨 split。

**网络**（`EllipseNet`，0.95 M 参数，40 epoch，RTX 4070，22 分钟）：test 上椭圆 IoU **0.768**、
中心误差 1.21 m、半轴误差 1.09/0.62 m、方向误差 16.0°、路径点落入预测椭圆 0.967、
预测椭圆无碰撞 0.720、**安全且包含路径点 0.692**。

完整交付说明见 `docs/实验交付说明.md`。详细产物：

* `map_inventory.csv/json` — 地图清单与元信息
* `dataset_statistics.md` — 数据生成统计（含分子/分母）
* `dataset_quality_report.md` — 独立复验（几何、射线、划分泄漏检查）
* `training_report.md` / `test_metrics.json` — 训练与测试指标
* `training_curves.png` / `test_predictions.png` — 曲线与预测可视化
* `outputs/previews/`（29）、`outputs/ellipse_visualizations/`（21）、`outputs/path_visualizations/`（8）— 图
* `outputs/checkpoints/ellipse_net_best.pt` — 最优模型

---

## 8. 方案 V3：网络结构重实现（`psen_v3/`）

按 `PathSafe-EllipseNet_网络结构与训练实施方案.md` 重新实现的一套独立管道，
与上面的 V2/V2.1 代码并存、互不影响：

* 数据：在原始 V3 轨迹数据（`data/processed_v3/`，未改动）之上新建距离场数据版本
  `data/processed_v3sdf/`（17 038 张唯一地图的米制有符号距离场 + 轨迹索引），
  由 `scripts/v3_build_sdf.py` 生成；`scripts/v3_audit_dataset.py` 做全量数据审计。
* 网络 `psen_v3/`：两个独立 ResUNet（256×256 全分辨率）+ 9 Token 占据查询（Flatten 前加相对位置）
  + 每射线 3 点 Distance 采样 + 点内 16 射线 Self-Attention + `MLP([c_occ, Flatten(R̂)])`
  + Path Self-Attention + Direction(16)/Ratio(1)/Shape(a,b,θ) 三头 + 16 个射线约束候选椭圆；
  **不使用任何人工射线编号 / 中心偏移 / 偏移比例标签**，只做几何损失的概率期望训练。
* 训练/评估/可视化：`psen_v3/train.py` / `evaluate.py` / `visualize.py`，
  消融 `scripts/v3_ablate.py`。

**测试集（`map_group` 完全 held-out，5 054 条轨迹 / 143 698 点）**：
精确碰撞率 **12.95%**（安全率 0.8705）、SDF 采样安全率 0.9808、中心落在障碍格 2.1e-4、
相邻中点连续性 0.967、整条走廊成功率 0.400。

完整说明（数据审计、距离场约定、逐条结构对照、权重标定、阶段 A/B/C 结果、消融、局限）见
**`docs/V3_网络与训练实施报告.md`**。


