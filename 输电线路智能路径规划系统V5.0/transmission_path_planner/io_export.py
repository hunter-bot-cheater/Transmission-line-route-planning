# transmission_path_planner/io_export.py
"""文件命名与路径导出（SHP/GeoJSON）。"""

import math

from .config import *

def safe_reference_name(cname):
    # 与 extract_real_cases.py 的 safe() 保持一致: 非(字母/数字/_/-)统一转 '_'
    return "".join(ch if (ch.isalnum() or ch in "_-") else "_" for ch in cname)


def _safe_name(cname):
    return safe_reference_name(cname)


class AmbiguousReferenceError(ValueError):
    pass


def _one_reference_match(hits, cname):
    hits = sorted(set(hits))
    if len(hits) > 1:
        raise AmbiguousReferenceError(f"{cname} 匹配到多个文件: {', '.join(map(str, hits))}")
    return hits[0] if hits else None


def resolve_real_path(cname, real_dir, include_global_mapping=True):
    """优先使用用例目录中的同名线路，再查显式映射和递归目录。"""
    candidate_name = f"{_safe_name(cname)}_real_path.shp"
    candidate = Path(real_dir) / candidate_name
    if candidate.exists():
        return candidate
    if include_global_mapping and cname in REAL_PATH_MAP:
        fname = REAL_PATH_MAP[cname]
        # 显式映射: 先在同层, 再递归子目录(如 comparisons/case_01/case_01_real_path.shp)
        cand0 = REAL_PATH_DIR / fname
        if cand0.exists():
            return cand0
        hits = list(REAL_PATH_DIR.rglob(fname))
        mapped = _one_reference_match(hits, cname)
        if mapped:
            return mapped
    hits = list(Path(real_dir).rglob(candidate_name))
    return _one_reference_match(hits, cname)


def geojson_coordinates(geo):
    """Serialize a route stored as (lon, lat) pairs in GeoJSON coordinate order."""
    return [[float(lon), float(lat)] for lon, lat in geo]


def write_geojson_verified(path, document, indent=None):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(document, f, ensure_ascii=False, indent=indent)
    with open(path, "r", encoding="utf-8") as f:
        readback = json.load(f)
    expected = json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = json.dumps(
        readback, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if actual != expected:
        raise RuntimeError(f"GeoJSON route export readback verification failed: {path}")


def export_route(geo, crs, shp_path, geojson_path):
    if not geo:
        return None
    crs_name = str(crs) if crs else "EPSG:4326"
    # 支持单折线或折线列表(线网)
    if isinstance(geo[0], (list, tuple)) and (not geo[0] or isinstance(geo[0][0], (list, tuple))):
        segments = [segment for segment in geo if len(segment) >= 2]
        names = [f"seg{i}" for i in range(len(segments))]
    else:
        segments = [geo] if len(geo) >= 2 else []
        names = ["route"] if segments else []
    if not segments:
        return None
    normalized_segments = []
    try:
        for segment in segments:
            normalized_segment = []
            for point in segment:
                if isinstance(point, (str, bytes, bytearray, memoryview, dict)) or not hasattr(point, "__getitem__"):
                    return None
                if len(point) != 2:
                    return None
                if any(isinstance(value, (str, bytes, bytearray, memoryview)) for value in point):
                    return None
                lon, lat = (float(value) for value in point)
                if not math.isfinite(lon) or not math.isfinite(lat):
                    return None
                normalized_segment.append((lon, lat))
            normalized_segments.append(normalized_segment)
    except (TypeError, ValueError, OverflowError, IndexError):
        return None
    segments = normalized_segments
    lines = [LineString([(lon, lat) for lon, lat in segment]) for segment in segments]
    shp_exported = False
    if HAS_GPD and lines:
        gdf = gpd.GeoDataFrame({"name": names}, geometry=lines, crs=crs_name)
        try:
            gdf.to_file(shp_path, driver="ESRI Shapefile")
            shp_exported = True
        except Exception:
            pass
    feats = [{"type": "Feature", "properties": {"name": names[i]},
              "geometry": {"type": "LineString",
                           "coordinates": geojson_coordinates(segments[i])}}
             for i in range(len(lines))]
    feat = {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": crs_name}},
            "features": feats}
    write_geojson_verified(geojson_path, feat, indent=2)
    return shp_path if shp_exported else geojson_path
