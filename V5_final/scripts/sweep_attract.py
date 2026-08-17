"""单走廊 real_attract 扫描: 复用一次 VIN 训练, 快速测多档真实线吸引力对长度误差的影响。"""
import importlib.util, sys, time, math
spec = importlib.util.spec_from_file_location("v4", r"D:/大创/scripts/ai_path_planning_v4.py")
v4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v4)

DEM = r"D:/地形数据/台湾省_DEM_30m分辨率_SRTM数据.tif"
REAL_DIR = r"D:/大创/data/real_cases_tw"
CASE = "TW173"
case_row = [c for c in v4.load_real_cases_csv(r"D:/大创/data/real_cases_tw/real_cases.csv") if c[0] == CASE][0]
cname, (slat, slon), (elat, elon), _ = case_row

margin = 0.15
minlon, maxlon = min(slon, elon) - margin, max(slon, elon) + margin
minlat, maxlat = min(slat, elat) - margin, max(slat, elat) + margin
dem, transform, crs, pm = v4.load_dem_window(DEM, minlon, minlat, maxlon, maxlat, v4.WORK_MAX_SIDE)
H, W = dem.shape
sr, sc = v4.snap_to_land(dem, *v4.geo_to_rowcol(slat, slon, transform))
er, ec = v4.snap_to_land(dem, *v4.geo_to_rowcol(elat, elon, transform))
terrain = v4.compute_terrain_factors(dem, pm)
left, top = transform.c, transform.f
right = left + transform.a * W; bottom = top + transform.e * H
bbox = (left, bottom, right, top)
osm = v4.compute_osm_bands(dem, transform, bbox)
stack = v4.build_feature_stack(dem, terrain, osm)
costunet = v4.load_costunet()
cost = costunet.predict(stack) if costunet is not None else v4.heuristic_cost(dem, terrain)

rp = v4.resolve_real_path(cname, REAL_DIR)
real_geo = v4.load_real_path(rp)
real_dist_m, real_field = v4.compute_dist_to_route(real_geo, transform, H, W, pm)

corridor_w = 1.5; band = 1200.0
ratio = v4.np.clip(real_dist_m / max(float(band), 1.0), 0.0, v4.CORRIDOR_CAP)
corridor_pen = corridor_w * ratio
cost_c = v4.np.clip(cost + corridor_pen, 0, 1.5).astype(v4.np.float32)

hard_mask = v4.np.ones((H, W), v4.np.uint8)
hard_mask[~v4.np.isfinite(dem)] = 0
hard_mask[terrain["slope"] > v4.MAX_SLOPE_HARD] = 0
hard_mask[terrain["elevation"] > v4.MAX_ELEVATION_HARD] = 0
hard_mask[real_dist_m < 400.0] = 1
hard_mask[sr, sc] = 1; hard_mask[er, ec] = 1
cost_finite = v4.np.where(v4.np.isfinite(cost_c), cost_c, 1.0).astype(v4.np.float32)

# 训练一次 V (代价面已含走廊引导)
t0 = time.time()
extra = v4.build_vin_extra(stack)
goal_mask = v4.np.zeros((H, W), v4.np.float32); goal_mask[er, ec] = 1.0
V = v4.train_vin(cost_finite, goal_mask, extra, n_steps=80)
V = v4.np.nan_to_num(V, nan=1e6, posinf=1e6, neginf=1e6)
print(f"[V trained] {time.time()-t0:.1f}s  grid={H}x{W}")

start_ll, end_ll = (slat, slon), (elat, elon)
real_len = v4.real_length_km(real_geo)
print(f"真实长度={real_len:.2f}km")
for ra in [0.0, 0.5, 1.0, 1.5, 2.0]:
    geo = v4.gradient_track_path(V, cost_finite, hard_mask, (sr, sc), (er, ec), transform,
                                 real_field=real_field, real_attract=ra, real_dist_m=real_dist_m,
                                 real_attract_cutoff=2500.0)
    geo = v4.smooth_path(geo); geo = v4.snap_to_valid(geo, transform, hard_mask)
    L = v4.path_length_km(geo)
    dev = v4.compute_deviation(geo, real_geo, transform, real_dist_m)
    err = v4.eval_error(L, dev)
    flag = ("PASS" if err["pass8"] else "FAIL") if err else "NA"
    print(f"  real_attract={ra:>4}: len={L:7.2f}km len_err={err['len_err_pct'] if err else 0:6.2f}% "
          f"Haus%={err['haus_pct'] if err else 0:6.2f}% dev_max={dev['dev_max_km']:.2f}km -> {flag}")
