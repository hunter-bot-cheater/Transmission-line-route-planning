# transmission_path_planner/io_export.py
"""文件命名与路径导出（SHP/GeoJSON）。"""

from .config import *

def _safe_name(cname):
    # 与 extract_real_cases.py 的 safe() 保持一致: 非(字母/数字/_/-)统一转 '_'
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in cname)


def resolve_real_path(cname, real_dir):
    """按走廊名解析真实线路 SHP: 优先显式映射(支持子目录, 递归搜索), 再按 '<净化名>_real_path.shp' 在 real_dir 查找。"""
    if cname in REAL_PATH_MAP:
        fname = REAL_PATH_MAP[cname]
        # 显式映射: 先在同层, 再递归子目录(如 comparisons/case_01/case_01_real_path.shp)
        cand0 = REAL_PATH_DIR / fname
        if cand0.exists():
            return cand0
        hits = list(REAL_PATH_DIR.rglob(fname))
        if hits:
            return hits[0]
    cand = Path(real_dir) / f"{_safe_name(cname)}_real_path.shp"
    if cand.exists():
        return cand
    hits = list(Path(real_dir).rglob(f"{_safe_name(cname)}_real_path.shp"))
    return hits[0] if hits else None


def geojson_coordinates(geo):
    """Serialize a route stored as (lon, lat) pairs in GeoJSON coordinate order."""
    return [[float(lon), float(lat)] for lon, lat in geo]


def export_route(geo, crs, shp_path, geojson_path):
    if not geo:
        return None
    # 支持单折线或折线列表(线网)
    if geo and isinstance(geo[0][0], (list, tuple)):
        lines = [LineString([(lon, lat) for lon, lat in p]) for p in geo if len(p) >= 2]
        names = [f"seg{i}" for i in range(len(lines))]
        seg_polys = geo
    else:
        lines = [LineString([(lon, lat) for lon, lat in geo])]
        names = ["route"]
        seg_polys = [geo]
    if HAS_GPD and lines:
        gdf = gpd.GeoDataFrame({"name": names}, geometry=lines, crs=crs or "EPSG:4326")
        try:
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            return shp_path
        except Exception:
            pass
    feats = [{"type": "Feature", "properties": {"name": names[i]},
              "geometry": {"type": "LineString",
                           "coordinates": geojson_coordinates(seg_polys[i])}}
             for i in range(len(lines))]
    feat = {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs or "EPSG:4326"}},
            "features": feats}
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(feat, f, ensure_ascii=False, indent=2)
    return geojson_path
