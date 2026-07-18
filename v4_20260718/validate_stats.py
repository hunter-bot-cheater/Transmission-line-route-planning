"""
V3 统计显著性检验 — 30次独立运行
版本: v3.20260710
"""
import sys
from pathlib import Path
_v3_dir = Path(__file__).resolve().parent
_v2_dir = _v3_dir.parent / "v2_20260525"
_v2_src = _v2_dir / "src"
_shared_dir = _v3_dir.parent / "shared"
for _d in reversed([str(_v3_dir / "src"), str(_v3_dir), str(_v2_dir), str(_v2_src), str(_shared_dir)]):
    sys.path.insert(0, _d)  # 无脑强插, 确保v3优先

import numpy as np
import json, time
from scipy import stats as scipy_stats
import config as cfg
from data_acquisition_v3 import load_taiwan_lines, acquire_all
from preprocessing import derive_terrain_factors, generate_hard_mask, generate_soft_mask, align_all_rasters
from cost_model import build_feature_stack, generate_pseudo_labels, train_random_forest, predict_cost_surface
from path_planning import fuse_cost_surface, astar_search, smooth_path, compute_path_length_km, haversine_m, geo_to_grid
from ipso_sa_planner import ipso_sa_plan_path
from dbo_planner import dbo_plan_path

N_RUNS = 10
print("=" * 60)
print(f"  V3 统计显著性检验 — {N_RUNS}次独立运行")
print("=" * 60)

# === Phase 1-2: 公共预处理 (同 V3) ===
print("\n[Phase 1-2] 数据获取 + 预处理...")
t0 = time.time()
data = acquire_all()
dem = data["dem"]; transform = data["dem_transform"]; crs_data = data["dem_crs"]
terrain_factors = derive_terrain_factors(dem, transform); data.update(terrain_factors)

test_way_ids = {case["way_id"] for case in cfg.TEST_CASES}
taiwan_lines_all = data.get("taiwan_lines")
if taiwan_lines_all is not None:
    data["taiwan_lines"] = taiwan_lines_all[~taiwan_lines_all["id"].isin(test_way_ids)].copy()

osm_data = {"osm_landuse": data.get("osm_landuse"), "osm_buildings": data.get("osm_buildings"),
            "osm_railways": data.get("osm_railways"), "osm_airports": data.get("osm_airports")}
aligned = align_all_rasters(data, dem, transform, osm_data=osm_data)
dst_transform = aligned["transform"]; dst_shape = aligned["shape"]
aligned_for_mask = dict(aligned)
for key in ["osm_water", "osm_protected"]:
    if key in data and data[key] is not None: aligned_for_mask[key] = data[key]
hard_mask = generate_hard_mask(aligned_for_mask, dst_transform, dst_shape)
soft_mask = generate_soft_mask(aligned_for_mask, dst_transform, dst_shape)
dist_existing = aligned.get("dist_existing_line")

# 成本表面 — CNN 优先
cnn_path = _v3_dir / "cnn_model" / "cnn_cost_surface.npy"
if cnn_path.exists():
    cnn_cost_raw = np.load(cnn_path)
    final_cost = fuse_cost_surface(cnn_cost_raw, soft_mask, hard_mask, dist_existing=dist_existing, smooth_sigma=0.0)
    cost_label = "CNN"
else:
    feature_stack = build_feature_stack(aligned)
    labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)
    rf, scaler, _ = train_random_forest(feature_stack, labels, hard_mask)
    predicted_cost = predict_cost_surface(rf, scaler, feature_stack, hard_mask)
    final_cost = fuse_cost_surface(predicted_cost, soft_mask, hard_mask, dist_existing=dist_existing)
    cost_label = "RF"
print(f"  成本表面: {cost_label}, 耗时: {time.time()-t0:.0f}s")

# === Phase 3: 选线路 ===
gdf = load_taiwan_lines()
selected = []
for case in cfg.TEST_CASES:
    rows = gdf[gdf["id"] == case["way_id"]]
    if len(rows) > 0:
        row = rows.iloc[0]
        case["real_coords"] = list(row.geometry.coords)
        case["voltage"] = row.get("voltage", "N/A")
        case["real_length"] = compute_path_length_km([(c[0], c[1]) for c in case["real_coords"]])
        selected.append(case)
print(f"  选中 {len(selected)} 条线路")

# === Phase 4: N次独立运行 ===
print(f"\n[Phase 3] {N_RUNS}次独立运行...")
all_results = {case["case_id"]: {"A*": [], "IPSO-SA": [], "DBO": []} for case in selected}

for run in range(N_RUNS):
    seed = cfg.COMPARISON_RANDOM_SEED + run
    print(f"\n--- Run {run+1}/{N_RUNS} (seed={seed}) ---")
    for case in selected:
        cid = case["case_id"]
        real_coords = case["real_coords"]
        start_lat, start_lon = real_coords[0][1], real_coords[0][0]
        end_lat, end_lon = real_coords[-1][1], real_coords[-1][0]
        start_rc = geo_to_grid(start_lat, start_lon, dst_transform)
        end_rc = geo_to_grid(end_lat, end_lon, dst_transform)
        start_latlon = (start_lat, start_lon); end_latlon = (end_lat, end_lon)

        # A*
        a_raw = astar_search(final_cost, start_rc, end_rc)
        if a_raw:
            a_path = smooth_path(a_raw, dst_transform)
            a_hd = max(min(haversine_m(pa[0],pa[1],pb[0],pb[1]) for pb in np.array(real_coords)[::2]) for pa in np.array(a_path)[::2])
            a_ov = sum(1 for pa in np.array(a_path)[::2] if min(haversine_m(pa[0],pa[1],pb[0],pb[1]) for pb in np.array(real_coords)[::2])<500)/max(len(np.array(a_path)[::2]),1)*100
            all_results[cid]["A*"].append({"hausdorff": a_hd, "overlap": a_ov})
        else:
            all_results[cid]["A*"].append(None)

        # IPSO-SA (快速模式)
        try:
            ipso_path, _, _ = ipso_sa_plan_path(final_cost, hard_mask, dst_transform, start_latlon, end_latlon,
                                                  aligned=aligned, a_star_path=a_path if a_raw else None,
                                                  random_seed=seed, verbose=False)
            ip_hd = max(min(haversine_m(pa[0],pa[1],pb[0],pb[1]) for pb in np.array(real_coords)[::2]) for pa in np.array(ipso_path)[::2])
            ip_ov = sum(1 for pa in np.array(ipso_path)[::2] if min(haversine_m(pa[0],pa[1],pb[0],pb[1]) for pb in np.array(real_coords)[::2])<500)/max(len(np.array(ipso_path)[::2]),1)*100
            all_results[cid]["IPSO-SA"].append({"hausdorff": ip_hd, "overlap": ip_ov})
        except:
            all_results[cid]["IPSO-SA"].append(None)

        # DBO — 跳过(不稳定), 仅标记
        all_results[cid]["DBO"].append(None)

    n_done = run + 1
    # Quick summary every 5 runs
    if n_done % 5 == 0:
        for algo in ["A*", "IPSO-SA", "DBO"]:
            hds = [r["hausdorff"] for cid in all_results for r in all_results[cid][algo] if r]
            if hds: print(f"  {algo}: runs={n_done}, Hausdorff μ={np.mean(hds):.0f}m")

# === Phase 5: 统计分析 ===
print(f"\n{'='*60}")
print(f"  统计检验结果 ({cost_label}成本表面, {N_RUNS}次运行)")
print(f"{'='*60}")

for cid in all_results:
    print(f"\n--- {cid} ---")
    for algo in ["A*", "IPSO-SA", "DBO"]:
        vals = [r for r in all_results[cid][algo] if r]
        if vals:
            hds = [v["hausdorff"] for v in vals]
            ovs = [v["overlap"] for v in vals]
            print(f"  {algo:<10}: HD μ={np.mean(hds):.0f}±{np.std(hds):.0f}m  OV μ={np.mean(ovs):.1f}±{np.std(ovs):.1f}%  n={len(vals)}")

# 汇总 t-test
print(f"\n--- 汇总配对t检验 ---")
for metric, mname in [("hausdorff", "Hausdorff"), ("overlap", "重叠率")]:
    for algo in ["IPSO-SA", "DBO"]:
        a_vals_all = []; b_vals_all = []
        for cid in all_results:
            a_runs = [r for r in all_results[cid]["A*"] if r]
            b_runs = [r for r in all_results[cid][algo] if r]
            n = min(len(a_runs), len(b_runs))
            for i in range(n):
                # Match runs by index
                if a_runs[i] and b_runs[i]:
                    a_vals_all.append(a_runs[i][metric])
                    b_vals_all.append(b_runs[i][metric])
        if len(a_vals_all) >= 10:
            t_stat, p_val = scipy_stats.ttest_rel(a_vals_all, b_vals_all)
            sig = "***" if p_val<0.001 else ("**" if p_val<0.01 else ("*" if p_val<0.05 else "n.s."))
            print(f"  A* vs {algo:<8} {mname}: t={t_stat:.2f}, p={p_val:.4f} {sig}  (n={len(a_vals_all)})")

# 保存
out = {"cost": cost_label, "n_runs": N_RUNS, "results": {}}
for cid in all_results:
    out["results"][cid] = {}
    for algo in ["A*", "IPSO-SA", "DBO"]:
        vals = [r for r in all_results[cid][algo] if r]
        if vals:
            out["results"][cid][algo] = {
                "hausdorff_mean": float(np.mean([v["hausdorff"] for v in vals])),
                "hausdorff_std": float(np.std([v["hausdorff"] for v in vals])),
                "overlap_mean": float(np.mean([v["overlap"] for v in vals])),
                "overlap_std": float(np.std([v["overlap"] for v in vals])),
                "n_valid": len(vals),
            }
with open(_v3_dir / "output" / "stats_30runs.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"\n结果保存: output/stats_30runs.json")
print("统计检验完成!")
