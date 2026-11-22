"""Print exactly what a generated scene NPZ contains (fields, shapes, value ranges)."""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from pathsafe_ellipse.v2.pilot import load_scene_npz  # noqa: E402

DESC = {
    "occupancy": "原始占据栅格（0=可通行, 1=障碍），256x256 单元",
    "blocked": "机器人膨胀后占据（D<=0.29 m 或无效区）——A*/中心线用的就是它",
    "valid_mask": "该单元是否落在源地图内（1=有效）",
    "dist_m": "到障碍面的距离场 D(x)，单位米（膨胀/净空都用它）",
    "meta_json": "本场景的全部生成元数据（版本/提交/参数/种子/来源）",
    "point_id": "标签在场景内的唯一编号（第几条标签）",
    "anchor_id": "沿路径的锚点序号",
    "path_id": "属于哪条路径（0,1,2,3）",
    "p_xy": "A* 站点 p_i 的坐标 (m)",
    "q_xy": "椭圆中心 q_i 的坐标 (m)",
    "center_offset": "q_i - p_i，即网络要预测的中心偏移 (m)",
    "path_tangent": "A* 站点处的单位切向",
    "lidar_ranges": "从 p_i 发出的 16 条射线距离 (m)",
    "ellipse_S": "椭圆形状矩阵 S 的 4 个分量 (2x2, m^2)",
    "ellipse_ab": "椭圆半长轴/半短轴 (m)",
    "ellipse_angle": "椭圆主轴方向 (rad)",
    "ellipse_area": "椭圆面积 (m^2)",
    "clearance": "中心 q_i 处的净空 D(q_i) (m)",
    "solver_status": "椭圆求解状态（optimal / initial_circle_kept / ...）",
    "safety_verified": "精确几何验证是否通过（1=安全）",
    "continuity_verified": "与相邻锚点的连续性是否通过（1/0）",
    "label_valid": "safety AND continuity —— 只有 1 才可用于训练",
    "generation_time_ms": "该椭圆求解耗时 (ms)",
    "rho_left": "与左邻锚点在中点的 Rho 值",
    "rho_right": "与右邻锚点在中点的 Rho 值",
    "path_p_xy": "每条路径的 A* 折线点列（按 offsets 切分）",
    "path_q_xy": "每条路径的优化中心线折线点列",
    "path_p_offsets": "path_p_xy 的分段边界（按 path_id 取值用）",
    "path_q_offsets": "path_q_xy 的分段边界",
}


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "dataset_v21"
    files = sorted(glob.glob(os.path.join(ROOT, "data", "processed_v2_1",
                                         f"pilot_{tag}", "v2_1", "*.npz")))
    print(f"dataset '{tag}': {len(files)} scene files")
    f = files[0]
    d = load_scene_npz(f)
    print(f"\n== 任意一个场景文件的完整内容: {os.path.basename(f)} ==")
    print(f"{'field':22s} {'shape':16s} {'dtype':10s} {'示例值':34s} 说明")
    for k in sorted(d):
        if k == "meta":
            continue
        v = d[k]
        if v.ndim == 0:
            ex = str(v)[:32]
        elif v.size and v.dtype.kind in "fiu":
            flat = v.ravel()
            ex = f"min {flat.min():.3g} / max {flat.max():.3g}"
        elif v.size:
            ex = str(v.ravel()[:2])
        else:
            ex = "-"
        print(f"{k:22s} {str(v.shape):16s} {str(v.dtype):10s} {ex:34s} {DESC.get(k, '')}")

    m = d["meta"]
    print(f"\n== meta（{len(m)} 个字段）==")
    for k in sorted(m):
        print(f"  {k:26s} {json.dumps(m[k], ensure_ascii=False)[:90]}")

    n = int(d["path_id"].shape[0])
    print(f"\n== 这个场景的规模 ==")
    print(f"  路径数 {len(d['path_path_id'])}   标签数 {n}   "
          f"其中 label_valid=1 的 {int(d['label_valid'].sum())}")
    print(f"  路径 0 的 A* 折线点 {int(d['path_p_offsets'][1])} 个，"
          f"中心线折线点 {int(d['path_q_offsets'][1])} 个")
    print(f"  每条标签自带: 16 维 LiDAR + 2 维切向 + 4 维椭圆矩阵 + 3 维(a,b,θ) + 面积/净空/状态/标记")

    # aggregate over the whole dataset
    tot = {"labels": 0, "valid": 0, "paths": 0, "scenes": 0, "bytes": 0}
    areas = []
    for p in files:
        dd = load_scene_npz(p)
        tot["scenes"] += 1
        tot["labels"] += int(dd["path_id"].shape[0])
        tot["valid"] += int(dd["label_valid"].sum())
        tot["paths"] += len(dd["path_path_id"])
        tot["bytes"] += os.path.getsize(p)
        areas.append(dd["ellipse_area"])
    areas = np.concatenate(areas) if areas else np.zeros(0)
    print(f"\n== 整个数据集 ==")
    print(f"  场景 {tot['scenes']} | 路径 {tot['paths']} | 标签 {tot['labels']} "
          f"（可用 {tot['valid']}）| 体积 {tot['bytes'] / 1e6:.1f} MB")
    if areas.size:
        print(f"  椭圆面积: p50 {np.median(areas):.1f} m2, min {areas.min():.3f}, "
              f"max {areas.max():.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
