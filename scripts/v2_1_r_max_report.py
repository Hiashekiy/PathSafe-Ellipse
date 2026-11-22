"""Aggregate the R_max sweep (stage 3) into one table + figure."""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports_v2_1")
VIZ = os.path.join(ROOT, "outputs_v2_1", "scenarios")

RMAX = [1.5, 2.0, 3.0, 4.0]
data = {}
for r in RMAX:
    p = os.path.join(REP, f"scenarios_r{r:.1f}.json")
    with open(p, encoding="utf-8") as fh:
        data[r] = json.load(fh)

scen = list(next(iter(data.values()))["scenarios"].keys())
print("R_max sweep over the five canonical scenarios (V2.1)")
print(f"{'scenario':16s} {'R_max':>6} {'gain[m]':>8} {'atR':>5} {'off_max':>8} "
      f"{'off_p95':>8} {'area_p50':>9} {'area_min':>9} {'<5m2':>6} {'homo':>6} {'ms':>7}")
rows = []
for name in scen:
    for r in RMAX:
        v = data[r]["scenarios"][name]["v21"]
        rows.append({"scenario": name, "r_max": r, "gain": v["gain"],
                     "at_R_limit": v.get("at_R_limit", 0),
                     "offset_max": v["offset_max"], "offset_p95": v["offset_p95"],
                     "area_p50": v["area_p50"], "area_min": v["area_min"],
                     "n_lt5": v["n_lt5"], "n_anchors": v["n_anchors"],
                     "homotopy": v["homotopy"], "ms": v["centreline_ms"],
                     "status": v["status"]})
        print(f"{name:16s} {r:6.1f} {v['gain']:+8.2f} {v.get('at_R_limit', 0):5d} "
              f"{v['offset_max']:8.2f} {v['offset_p95']:8.2f} {v['area_p50']:9.2f} "
              f"{v['area_min']:9.3f} {v['n_lt5']:3d}/{v['n_anchors']:<3d} "
              f"{str(v['homotopy']):>6} {v['centreline_ms']:7.0f}")

# aggregate per R_max
print("\naggregate")
print(f"{'R_max':>6} {'mean gain':>10} {'mean area p50':>14} {'atR share':>10} "
      f"{'mean ms':>9} {'max off':>8}")
agg = {}
for r in RMAX:
    gains, areas, atr, mss, offs = [], [], [], [], []
    for name in scen:
        v = data[r]["scenarios"][name]["v21"]
        gains.append(v["gain"])
        areas.append(v["area_p50"])
        mss.append(v["centreline_ms"])
        offs.append(v["offset_max"])
        tot = max(v["n_anchors"], 1)
        atr.append(v.get("at_R_limit", 0) / tot)
    agg[r] = {"mean_gain": float(np.mean(gains)), "mean_area_p50": float(np.mean(areas)),
              "at_R_share": float(np.mean(atr)), "mean_ms": float(np.mean(mss)),
              "max_offset": float(np.max(offs))}
    print(f"{r:6.1f} {agg[r]['mean_gain']:10.2f} {agg[r]['mean_area_p50']:14.2f} "
          f"{agg[r]['at_R_share']:10.2f} {agg[r]['mean_ms']:9.0f} {agg[r]['max_offset']:8.2f}")

fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))
for name in scen:
    sub = [r for r in rows if r["scenario"] == name]
    xs = [s["r_max"] for s in sub]
    axes[0].plot(xs, [s["gain"] for s in sub], "-o", ms=3, label=name)
    axes[1].plot(xs, [s["area_p50"] for s in sub], "-o", ms=3, label=name)
    axes[2].plot(xs, [s["at_R_limit"] / max(s["n_anchors"], 1) for s in sub], "-o", ms=3,
                 label=name)
axes[0].set_ylabel("clearance gain [m]"); axes[0].set_title("gain vs R_max")
axes[1].set_ylabel("area p50 [m2]"); axes[1].set_title("label area vs R_max")
axes[2].set_ylabel("share of stations at R_i limit"); axes[2].set_title("constraint activity")
for ax in axes[:3]:
    ax.set_xlabel("R_max [m]"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
xs = RMAX
axes[3].bar([str(r) for r in xs], [agg[r]["mean_ms"] for r in xs])
axes[3].set_ylabel("mean centreline solve [ms]"); axes[3].set_title("runtime")
fig.suptitle("V2.1 stage 3 - R_max sensitivity across five canonical scenarios")
fig.tight_layout()
dst_fig = os.path.join(VIZ, "_r_max_sweep.png")
fig.savefig(dst_fig, dpi=115)
plt.close(fig)

dst = os.path.join(REP, "r_max_sweep.json")
with open(dst, "w", encoding="utf-8") as fh:
    json.dump({"rows": rows, "aggregate": {str(k): v for k, v in agg.items()}}, fh, indent=2)
print(f"\n-> {dst}\n-> {dst_fig}")
