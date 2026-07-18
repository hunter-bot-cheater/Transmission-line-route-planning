"""A* 搜索效率对比: RF vs CNN 成本表面 (Octile启发式)"""
import sys; from pathlib import Path
_v3_dir = Path(__file__).resolve().parent.parent
_v2_dir = _v3_dir.parent / "v2_20260525"; _v2_src = _v2_dir / "src"
_shared_dir = _v3_dir.parent / "shared"
for _d in reversed([str(_v3_dir / "src"), str(_v3_dir), str(_v2_dir), str(_v2_src), str(_shared_dir)]):
    sys.path.insert(0, _d)

import numpy as np, math, heapq, time
import config as cfg
from data_acquisition import acquire_all, load_taiwan_lines
from preprocessing import derive_terrain_factors, generate_hard_mask, generate_soft_mask, align_all_rasters
from cost_model import build_feature_stack, generate_pseudo_labels, train_random_forest, predict_cost_surface
from path_planning import fuse_cost_surface, geo_to_grid

print("加载数据...")
t0 = time.time()
data = acquire_all()
dem = data["dem"]; transform = data["dem_transform"]
terrain_factors = derive_terrain_factors(dem, transform); data.update(terrain_factors)

test_way_ids = {case["way_id"] for case in cfg.TEST_CASES}
tl = data.get("taiwan_lines")
if tl is not None: data["taiwan_lines"] = tl[~tl["id"].isin(test_way_ids)].copy()
aligned = align_all_rasters(data, dem, transform, {})
dst_transform = aligned["transform"]
hard_mask = generate_hard_mask(dict(aligned), dst_transform, aligned["shape"])
soft_mask = generate_soft_mask(dict(aligned), dst_transform, aligned["shape"])
feature_stack = build_feature_stack(aligned)
labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)
rf, scaler, _ = train_random_forest(feature_stack, labels, hard_mask)
rf_predicted = predict_cost_surface(rf, scaler, feature_stack, hard_mask)
dist_ex = aligned.get("dist_existing_line")
rf_cost = fuse_cost_surface(rf_predicted, soft_mask, hard_mask, dist_existing=dist_ex)

cnn_path = _v3_dir / "cnn_model" / "cnn_cost_surface.npy"
cnn_raw = np.load(cnn_path)
# CNN输出已含inf+已完成平滑, 不经过fuse_cost_surface
assert cnn_raw.shape == aligned["shape"], f"Shape mismatch: CNN {cnn_raw.shape} vs aligned {aligned['shape']}"
cnn_cost = cnn_raw.astype(np.float32)
valid_cnn = cnn_cost[cnn_cost < np.inf]
print(f"  CNN表面: shape={cnn_cost.shape}, 有效={len(valid_cnn):,}, 范围=[{valid_cnn.min():.4f}, {valid_cnn.max():.4f}]")
print(f"  预处理完成 ({time.time()-t0:.0f}s), grid={aligned['shape']}")

def astar_count_nodes(cost_raster, start_rc, end_rc):
    H, W = cost_raster.shape; sr, sc = start_rc; er, ec = end_rc
    valid = cost_raster[cost_raster < np.inf]
    if len(valid) == 0: return None, 0
    c_min = max(np.percentile(valid, 1), 1e-6)
    neighbors = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
    nd = [1.0, math.sqrt(2), 1.0, math.sqrt(2), 1.0, math.sqrt(2), 1.0, math.sqrt(2)]
    def h(r,c): dr,dc=abs(r-er),abs(c-ec); return c_min*(max(dr,dc)+(math.sqrt(2)-1)*min(dr,dc))
    g_score = {start_rc: 0.0}; open_set = [(h(sr,sc), 0, start_rc)]
    came_from = {}; closed = set(); nodes = 0
    while open_set:
        f, _, cur = heapq.heappop(open_set)
        if cur in closed: continue; nodes += 1
        if cur == (er, ec):
            path = [cur]
            while cur in came_from: cur = came_from[cur]; path.append(cur)
            return path[::-1], nodes
        closed.add(cur); cr, cc = cur
        for ni, (dr, dc) in enumerate(neighbors):
            nr, nc = cr+dr, cc+dc
            if not (0 <= nr < H and 0 <= nc < W): continue
            if np.isinf(cost_raster[nr, nc]): continue
            if dr and dc:
                if np.isinf(cost_raster[cr+dr, cc]) or np.isinf(cost_raster[cr, cc+dc]): continue
            tg = g_score[cur] + cost_raster[nr, nc] * nd[ni]
            if (nr, nc) not in g_score or tg < g_score[(nr, nc)]:
                g_score[(nr, nc)] = tg
                heapq.heappush(open_set, (tg+h(nr, nc), nodes, (nr, nc)))
                came_from[(nr, nc)] = cur
    return None, nodes

gdf = load_taiwan_lines()
print(f"\n=== A* 搜索效率对比 ===")
print(f"{'案例':<10} {'RF节点':<10} {'CNN节点':<10} {'CNN节省%':<10}")
rf_all, cnn_all = [], []
for case in cfg.TEST_CASES:
    rows = gdf[gdf["id"] == case["way_id"]]
    if len(rows) == 0: continue
    rc = list(rows.iloc[0].geometry.coords)
    s = geo_to_grid(rc[0][1], rc[0][0], dst_transform)
    e = geo_to_grid(rc[-1][1], rc[-1][0], dst_transform)
    _, rf_n = astar_count_nodes(rf_cost, s, e)
    _, cnn_n = astar_count_nodes(cnn_cost, s, e)
    save = (rf_n-cnn_n)/rf_n*100 if rf_n>0 else 0
    rf_all.append(rf_n); cnn_all.append(cnn_n)
    print(f"{case['case_id']:<10} {rf_n:<10} {cnn_n:<10} {save:+.0f}%")

print(f"\nRF  平均: {np.mean(rf_all):.0f} 节点")
print(f"CNN 平均: {np.mean(cnn_all):.0f} 节点 ({(np.mean(cnn_all)-np.mean(rf_all))/np.mean(rf_all)*100:+.0f}%)")
