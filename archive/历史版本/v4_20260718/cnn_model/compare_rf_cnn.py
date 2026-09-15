"""
RF vs CNN 成本预测对比 — A* 路径质量
版本: v3.20260710
"""
import sys
from pathlib import Path

_v3_dir = Path(__file__).resolve().parent.parent
_v2_dir = _v3_dir.parent / "v2_20260525"
_v2_src = _v2_dir / "src"
_shared_dir = _v3_dir.parent / "shared"
for _d in reversed([str(_v3_dir), str(_v2_dir), str(_v2_src), str(_shared_dir)]):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import json
import time
import config as cfg
from data_acquisition import acquire_all, load_taiwan_lines
from preprocessing import derive_terrain_factors, generate_hard_mask, generate_soft_mask, align_all_rasters
from cost_model import build_feature_stack, generate_pseudo_labels, train_random_forest, predict_cost_surface
from path_planning import fuse_cost_surface, astar_search, smooth_path, compute_path_length_km, haversine_m, _strict_quality_gate, geo_to_grid

print("=" * 60)
print("  RF vs CNN 成本预测 — A* 路径质量对比")
print("=" * 60)

# === 1. 公共预处理 (复用 V2) ===
print("\n[1/4] 数据获取 + 预处理...")
t0 = time.time()
data = acquire_all()
dem = data["dem"]
transform = data["dem_transform"]
crs_data = data["dem_crs"]
terrain_factors = derive_terrain_factors(dem, transform)
data.update(terrain_factors)

test_way_ids = {case["way_id"] for case in cfg.TEST_CASES}
taiwan_lines_all = data.get("taiwan_lines")
if taiwan_lines_all is not None:
    data["taiwan_lines"] = taiwan_lines_all[~taiwan_lines_all["id"].isin(test_way_ids)].copy()

osm_data = {
    "osm_landuse": data.get("osm_landuse"), "osm_buildings": data.get("osm_buildings"),
    "osm_railways": data.get("osm_railways"), "osm_airports": data.get("osm_airports"),
}
aligned = align_all_rasters(data, dem, transform, osm_data=osm_data)
dst_transform = aligned["transform"]
dst_shape = aligned["shape"]

aligned_for_mask = dict(aligned)
for key in ["osm_water", "osm_protected"]:
    if key in data and data[key] is not None:
        aligned_for_mask[key] = data[key]
hard_mask = generate_hard_mask(aligned_for_mask, dst_transform, dst_shape)
soft_mask = generate_soft_mask(aligned_for_mask, dst_transform, dst_shape)
dist_existing = aligned.get("dist_existing_line")
print(f"  预处理完成, 耗时: {time.time() - t0:.0f}s")

# === 2. RF 成本表面 ===
print("\n[2/4] 生成 RF 成本表面...")
feature_stack = build_feature_stack(aligned)
labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)
rf, scaler, _ = train_random_forest(feature_stack, labels, hard_mask)
rf_predicted = predict_cost_surface(rf, scaler, feature_stack, hard_mask)
rf_cost = fuse_cost_surface(rf_predicted, soft_mask, hard_mask, dist_existing=dist_existing)

# === 3. CNN 成本表面 ===
print("\n[3/4] 加载 CNN 成本表面...")
cnn_dir = Path(__file__).resolve().parent
cnn_cost_npy = np.load(cnn_dir / "cnn_cost_surface.npy")  # (H, W)
# CNN 成本已经包含硬约束 (inf), 这里只做软约束+走廊偏好
cnn_cost = fuse_cost_surface(cnn_cost_npy, soft_mask, hard_mask, dist_existing=dist_existing)

# === 4. A* 对比 ===
print("\n[4/4] A* 路径对比...")
gdf = load_taiwan_lines()
test_cases = cfg.TEST_CASES

print(f"\n{'案例':<10} {'真实长度':<10} {'RF长度':<10} {'CNN长度':<10} {'RF Hausdorff':<14} {'CNN Hausdorff':<14} {'RF重叠':<10} {'CNN重叠':<10}")
print("-" * 96)

results = []
for case in test_cases:
    rows = gdf[gdf["id"] == case["way_id"]]
    if len(rows) == 0:
        continue
    row = rows.iloc[0]
    real_coords = list(row.geometry.coords)
    real_length = compute_path_length_km([(c[0], c[1]) for c in real_coords])
    start_lat, start_lon = real_coords[0][1], real_coords[0][0]
    end_lat, end_lon = real_coords[-1][1], real_coords[-1][0]
    start_rc = geo_to_grid(start_lat, start_lon, dst_transform)
    end_rc = geo_to_grid(end_lat, end_lon, dst_transform)

    # RF A*
    rf_astar_raw = astar_search(rf_cost, start_rc, end_rc)
    rf_path = smooth_path(rf_astar_raw, dst_transform) if rf_astar_raw else None
    rf_len = compute_path_length_km(rf_path) if rf_path else 0

    # CNN A*
    cnn_astar_raw = astar_search(cnn_cost, start_rc, end_rc)
    cnn_path = smooth_path(cnn_astar_raw, dst_transform) if cnn_astar_raw else None
    cnn_len = compute_path_length_km(cnn_path) if cnn_path else 0

    # 验证指标 (内联,避免导入循环)
    def _hd(a, b):
        if a is None: return 0
        a_arr = np.array(a)
        b_arr = np.array(b)
        return max(
            max(min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b_arr) for pa in a_arr[::2]),
            max(min(haversine_m(pb[0], pb[1], pa[0], pa[1]) for pa in a_arr) for pb in b_arr[::2])
        )
    def _ov(a, b):
        if a is None: return 0
        a_arr = np.array(a)
        b_arr = np.array(b)
        within = sum(1 for pa in a_arr[::2]
                     if min(haversine_m(pa[0], pa[1], pb[0], pb[1]) for pb in b_arr[::2]) < 500)
        return within / max(len(a_arr[::2]), 1) * 100

    rf_hd = _hd(rf_path, real_coords)
    cnn_hd = _hd(cnn_path, real_coords)
    rf_ov = _ov(rf_path, real_coords)
    cnn_ov = _ov(cnn_path, real_coords)

    print(f"{case['case_id']:<10} {real_length:<10.1f} {rf_len:<10.1f} {cnn_len:<10.1f} {rf_hd:<14.0f} {cnn_hd:<14.0f} {rf_ov:<10.1f} {cnn_ov:<10.1f}")
    results.append({
        "case_id": case["case_id"],
        "rf_hausdorff": rf_hd, "cnn_hausdorff": cnn_hd,
        "rf_overlap": rf_ov, "cnn_overlap": cnn_ov,
        "rf_length": rf_len, "cnn_length": cnn_len,
    })

# 汇总
if results:
    rf_hds = [r["rf_hausdorff"] for r in results]
    cnn_hds = [r["cnn_hausdorff"] for r in results]
    rf_ovs = [r["rf_overlap"] for r in results]
    cnn_ovs = [r["cnn_overlap"] for r in results]
    print(f"\n汇总 (n={len(results)}):")
    print(f"  RF  Hausdorff: {np.mean(rf_hds):.0f}m  重叠率: {np.mean(rf_ovs):.1f}%")
    print(f"  CNN Hausdorff: {np.mean(cnn_hds):.0f}m  重叠率: {np.mean(cnn_ovs):.1f}%")

    with open(cnn_dir / "rf_cnn_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "summary": {
            "rf_hausdorff_mean": float(np.mean(rf_hds)),
            "cnn_hausdorff_mean": float(np.mean(cnn_hds)),
            "rf_overlap_mean": float(np.mean(rf_ovs)),
            "cnn_overlap_mean": float(np.mean(cnn_ovs)),
        }}, f, ensure_ascii=False, indent=2)

print("\n对比完成!")
