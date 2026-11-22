"""Stage-1 acceptance summary: no accepted path with failures, no false optimal_safe."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "reports_v2_1", "scenarios_r3.0.json"), encoding="utf-8"))

print(f"{'scenario':16s} {'method':6s} {'label_status':26s} {'accepted':9s} hard_failures")
bad = 0
for name, row in d["scenarios"].items():
    for m in ("v2", "v21"):
        v = row[m]
        fails = ",".join(v["chk"].get("hard_failures", [])) or "-"
        print(f"{name:16s} {m:6s} {v['label_status']:26s} {str(v['accepted']):9s} {fails}")
        if v["label_status"] == "optimal_safe" and (not v["accepted"] or not v["solver_success"]):
            bad += 1
        if v["accepted"] and v["chk"].get("hard_failures"):
            bad += 1
        if (not v["accepted"]) and v["label_status"] not in ("failed", "infeasible"):
            bad += 1
print()
print("mislabelled rows:", bad)
print("d_safe:", d["scenarios"]["wall_one_side"]["v21"]["chk"].get("d_safe"),
      "| segment step m:", d["scenarios"]["wall_one_side"]["v21"]["chk"].get("segment_step_m"),
      "| homotopy lam step m:", d["scenarios"]["wall_one_side"]["v21"]["chk"].get("homotopy_lam_step_m"),
      "| segment samples:", d["scenarios"]["wall_one_side"]["v21"]["chk"].get("n_segment_samples"))
