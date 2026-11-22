"""Print the degraded/similar scenes from the stage-4 statistics."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, "reports_v2_1", "stage4_stats_stage4_20.json"),
                   encoding="utf-8"))

print("scenes DEGRADED (ratio < 0.98):")
for r in d["scenes"]:
    if "v2" not in r or not r["v2"].get("areas_p50") or not r["v21"].get("areas_p50"):
        continue
    ratio = r["v21"]["areas_p50"] / r["v2"]["areas_p50"]
    if ratio < 0.98:
        print("  {:44s} {:7.2f} -> {:7.2f} m2  ratio {:.2f}  density {:.3f} field {:.0f} m"
              .format(r["scene"][:44], r["v2"]["areas_p50"], r["v21"]["areas_p50"],
                      ratio, r["density"], r["field_m"]))
print("\nscenes SIMILAR (0.98 .. 1.02):")
for r in d["scenes"]:
    if "v2" not in r or not r["v2"].get("areas_p50") or not r["v21"].get("areas_p50"):
        continue
    ratio = r["v21"]["areas_p50"] / r["v2"]["areas_p50"]
    if 0.98 <= ratio <= 1.02:
        print("  {:44s} {:7.2f} -> {:7.2f} m2  ratio {:.2f}"
              .format(r["scene"][:44], r["v2"]["areas_p50"], r["v21"]["areas_p50"], ratio))
print("\nlargest improvements:")
pairs = [(r["v21"]["areas_p50"] / r["v2"]["areas_p50"], r) for r in d["scenes"]
         if "v2" in r and r["v2"].get("areas_p50") and r["v21"].get("areas_p50")]
for ratio, r in sorted(pairs, reverse=True)[:5]:
    print("  {:44s} {:7.2f} -> {:7.2f} m2  ratio {:.2f}"
          .format(r["scene"][:44], r["v2"]["areas_p50"], r["v21"]["areas_p50"], ratio))
print("\nsmall-area share {:.3f} -> {:.3f} | min area {:.3f} -> {:.3f} m2".format(
    d["B_ellipse"]["v2"]["small_share_mean"], d["B_ellipse"]["v21"]["small_share_mean"],
    d["B_ellipse"]["v2"]["area_min_min"], d["B_ellipse"]["v21"]["area_min_min"]))
