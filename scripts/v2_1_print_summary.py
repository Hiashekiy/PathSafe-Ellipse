"""Print the V2.1 pilot summary in a compact table."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "reports_v2_1", "v2_pilot_summary.json"), encoding="utf-8"))
print("scenes:", d["n_scenes"], "| anchors:", d["n_anchors"],
      "| verified:", d["n_verified_safe"], "| total_s:", d["total_s"])
print()
print("scene                                               FOV   occ    anchors ins  statuses")
for s in d["scenes"]:
    print(f"{s['scene_id'][:48]:50s} {s['physical_field_m']:4.0f} {s['density_inflated']:6.3f} "
          f"{s['n_anchors']:7d} {s['n_inserted']:4d}  {s['statuses']}")
print()
print("per-path centreline")
for s in d["scenes"]:
    for r in s.get("paths", []):
        print(f"  {s['scene_id'][:30]:32s} p{r['path_id']} {r['centerline_method'][:26]:27s} "
              f"{r['status'][:24]:26s} gain {r['clearance_gain_mean']:+.2f} m  "
              f"len x{r['length_ratio']:.3f}  homotopy {str(r['homotopy_verified']):5s}  "
              f"offset max {r['max_offset_m']:.2f} m  centreline {r['centerline_ms']:.0f} ms")
