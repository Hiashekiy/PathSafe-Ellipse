# PathSafe-EllipseNet（方案 V3 结构实现）

用**零人工标签**的方式，从一张 256×256 全局占据图 + 一条 A\* 轨迹 + 每个路径点的 16 条 LiDAR
测距射线，直接预测一组**连续的安全椭圆走廊**（沿轨迹逐点一个椭圆，中心必须落在某条 LiDAR
射线上）。

- 训练损失只有几何项：中心净空、椭圆安全、面积、相邻椭圆连接、相邻偏移连贯 ——
  **不使用** GT 中心 / 半轴 / 朝向 / 射线编号标签。
- 当前模型：`checkpoints/v3_best.pt`（5.04 M 参数）。
- 实施依据：`PathSafe-EllipseNet_网络结构与训练实施方案.md`；完整交付报告见
  **[`docs/report.md`](docs/report.md)**。

## 当前结果（`map_group` 完全 held-out 的测试集：5 054 条轨迹 / 143 698 点）

| 指标 | 数值 |
|---|---|
| 精确碰撞率（blocked 栅格解析判定） | **7.43%**（安全率 0.9257） |
| SDF 采样安全率（可微代理，仅作对照） | 0.9939 |
| 相邻中点连续性 | **0.9940**（标注椭圆自身为 1.000） |
| 整条路径走廊成功率 | **0.545** |
| 越出地图边界比例 | 0.85% |
| 单条轨迹推理耗时（batch 1，RTX 4070） | **10.9 ms**（模型 8.4 ms；同场景复用地图特征 2.8 ms） |

定性与定量对比：`results/figures/predictions/`（6 条预测/GT 对照 + 16 候选放大）、
`results/figures/grid/`（24 条轨迹网格，含纯白底版）、`results/figures/curves/`（损失曲线与指标表）。

## 目录

```text
configs/v3.yaml            全部超参（含标定后的 loss.scales、消融开关）
ellipsenet/                当前实现（唯一代码来源）
├── models/                resunet.py（两个独立 ResUNet）、psenet.py（主网络）
├── geometry/              sampling.py（坐标查询/可微 SDF/椭圆采样）、losses.py、
│                          exact.py（精确椭圆-栅格判定，自带，无外部依赖）
├── data/                  sdf.py（距离场定义）、dataset.py（轨迹级数据集/划分/翻转增强）
├── evaluation/            metrics.py（精确+张量指标）、runner.py
└── train.py · evaluate.py · visualize.py · visualize_grid.py · config.py
tests/                     阶段 A 测试：test_sdf / test_alignment / test_candidates / test_losses
scripts/                   8 个独立脚本（见下）
data/
├── maps/                  原始 MovingAI 地图（用户输入，只读）
├── trajectories/          轨迹级数据集（50 648 条轨迹，训练/验收的输入）
└── distance_fields/       距离场数据版本（17 038 张唯一地图 + index.jsonl + manifest.json）
checkpoints/v3_best.pt     交付模型（另含 v3_config.json / v3_train_log.csv / v3_run_meta.json）
results/
├── metrics/               验收指标、审计、消融、GT 基准、数据契约、推理时延（json/md）
├── figures/               预测图（predictions/）、24 条网格（grid/）、审计样本（audit/）、
│                          损失曲线与指标表（curves/）
└── logs/                  三个保留下来的训练控制台日志
docs/report.md             实施报告（结构对照、标定过程、全部数字与复现命令）
```

`scripts/` 一览：`audit_dataset.py`（数据审计）、`build_distance_fields.py`（距离场数据版本）、
`check_data_contract.py`（字段/单位契约）、`run_ablation.py`（8 项消融）、
`make_report_figures.py`（损失曲线与指标表）、`measure_gt_continuity.py`（标注椭圆连续性基准）、
`bench_inference.py`（单条推理时延）、`bench_resunet.py`（ResUNet 阶段级剖析）。

## 环境

`E:/CondaEnvData/envs/GGMPC/python.exe`（Python 3.10.19，torch 2.9.1+cu126、numpy 2.2.6、
scipy 1.15.3、matplotlib 3.10.7、pyyaml），RTX 4070 12 GB。见 `requirements.txt`。

## 常用命令

```bash
PY=E:/CondaEnvData/envs/GGMPC/python.exe

# 阶段 A 测试（几何算子）
$PY tests/test_sdf.py
$PY tests/test_alignment.py
$PY tests/test_candidates.py
$PY tests/test_losses.py

# 验收（held-out）
$PY ellipsenet/evaluate.py --checkpoint checkpoints/v3_best.pt --split test

# 可视化
$PY ellipsenet/visualize.py --checkpoint checkpoints/v3_best.pt --split test --n 6 \
    --seed 7 --candidate-figs 2 --out-dir results/figures/predictions
$PY ellipsenet/visualize_grid.py --checkpoint checkpoints/v3_best.pt --split test \
    --n 24 --seed 11 --cols 6 --plain --no-titles --big 3 --out-dir results/figures/grid

# 推理时延基准（batch 1）
$PY scripts/bench_inference.py --checkpoint checkpoints/v3_best.pt --iters 50

# 从头重建：距离场数据版本 → 训练 → 连续性微调（详见报告 13 节）
$PY scripts/build_distance_fields.py --workers 12 --verify 40
$PY ellipsenet/train.py --tag train10k_e100 --sample-train 10000 --epochs 100 --batch-size 8 \
    --patience 25 --snapshot-every 10 --set train__eval_every=2
$PY ellipsenet/train.py --tag connctl --resume checkpoints/train10k_e100/best.pt --no-epoch-carry \
    --sample-train 10000 --epochs 4 --batch-size 8 --lr 4e-5 --set loss__w_span=0.0
$PY ellipsenet/train.py --tag conft3 --resume checkpoints/connctl/best.pt --no-epoch-carry \
    --sample-train 10000 --epochs 3 --batch-size 8 --lr 1e-5 --set loss__w_smooth=0.5
```

## 设计要点（与实施方案一一对应）

| 环节 | 实现 |
|---|---|
| 地图编码 | **两个独立 ResUNet**（占据 / 距离场各一，256×256 全分辨率，C=128） |
| 路径点查询 | 3×3 九个 Token，**Flatten 之前**加相对位置编码，再 MLP 聚合 |
| 射线特征 | 每条射线沿程 3 点查询 Distance 特征 + SDF 标量 + λ + 测距 + 方向 + 有效位 |
| 点-射线 | 只做 MLP 融合（无 Cross-Attention），占据特征广播到 16 条射线 |
| 点内交互 | 同一点 16 条射线 Self-Attention（2 层 4 头 Pre-LN） |
| 点特征 | `MLP([c_occ, Flatten(R̂)])` → 弧长位置编码 → **Path Self-Attention** |
| 输出头 | Direction(16 logits) / Ratio(1) / Shape(a,b,θ)，`π=softmax(l/τ)`，无 CE 无 Gumbel |
| 候选 | `q_ij = p_i + α·max(d_ij − m,0)·u_ij`，中心严格位于各自射线上；推理 `argmax` 取真实候选 |
| 损失 | `L_choice = Σπ L_local`（净空+安全+面积）`+ λ_c L_conn`（边界覆盖）`+ λ_s L_smooth`（相邻偏移连贯） |

两次关键取舍（详见报告 §9.6）：弧间隔中点覆盖项 `w_span` **实测与安全项硬冲突**
（安全率 0.92→0.50），默认关闭；改为罚相邻偏移向量不一致 `w_smooth` 后，
中点连续性 0.963→0.994、走廊成功率 0.429→0.545，代价是碰撞率 5.5%→7.4%。

## 凸区域生成（椭圆 → 多面体）

`ellipsenet/geometry/polytopes.py` 把每个预测椭圆吹成凸多面体（半空间交）。方法借鉴
Neural-IRIS 的局部 patch 贪心 + 分离超平面，做了三处改动：方格"近端面"平面、闭式精确
重叠判定、8 方向种子 + 单一精确循环（循环终止条件即验收）。

```bash
$PY tests/test_polytopes.py                          # 10 组不变量（含速度与并行/串行等价安全）
$PY scripts/build_convex_regions.py --n 8 --seed 5   # 预测→区域→验收+图
```

实测（RTX 4070）：端到端 **18.0 ms/轨迹（0.71 ms/椭圆）**；同一地图 256 个椭圆一次批量
**41 µs/椭圆**；CPU 回退 0.35 ms/椭圆；「可行且椭圆自身安全」的区域 **0 违规**（302 与 202
两个样本）。原始违规区域 6–9 个，全部属于自身碰撞或越出地图的椭圆（几何上无解）。
`window_margin_m` 是区域大小/速度的旋钮。

## 已知局限

1. 目标函数没有“贴路径”约束，学到的解是“把中心推到开阔处、椭圆撑得更大”：
   面积中位约为标注椭圆的 1.1–1.4 倍，中心误差中位 2.4–2.5 m。
2. 碰撞集中在窄通道 / 路口拐角（中心碰撞率仅 2e-4，问题在尺寸与朝向）。
3. “边界互相包含覆盖率”实测封顶 ≈0.43（标注椭圆自身 0.4295），不宜作为优化目标。
4. 单次运行结果，未做多种子方差；0.9940 这类数字带有 ±1–2 点不确定性。

## 版本

- 数据版本：`data/trajectories`（label_version v3）+ `data/distance_fields`（sdf_version v3sdf-1）
- 模型：`checkpoints/v3_best.pt`（从 `train10k_e100` → `connctl` → `conft3` 逐级微调而来）
- git 历史保留全部开发提交；工作区只保留本版代码、模型与必要数据。
