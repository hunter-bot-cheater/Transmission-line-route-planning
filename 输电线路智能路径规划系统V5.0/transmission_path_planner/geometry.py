# transmission_path_planner/geometry.py
"""几何与坐标工具（经纬度<->栅格、Haversine 与 WGS84 距离）。"""

from .config import *
from pyproj import Geod

_WGS84_GEOD = Geod(ellps="WGS84")

def geo_to_rowcol(lat, lon, transform):
    col, row = ~transform * (lon, lat)
    return int(math.floor(row)), int(math.floor(col))


def rowcol_to_geo(row, col, transform):
    lon = transform.c + (col + 0.5) * transform.a
    lat = transform.f + (row + 0.5) * transform.e
    return lon, lat  # 约定: geo 为 (lon, lat)，与 sample_dem/compute_metrics/path_length_km/绘图/导出一致


def pixel_spacing_meters(transform, lat_center, crs=None):
    """Return (row, column) pixel spacing in ground meters near the raster center."""
    geographic = crs is None
    unit_factor = 1.0
    if crs is not None and HAS_RASTERIO:
        try:
            raster_crs = rasterio.crs.CRS.from_user_input(crs)
            geographic = raster_crs.is_geographic
            if not geographic:
                factor = raster_crs.linear_units_factor
                unit_factor = float(factor[1]) if isinstance(factor, tuple) else float(factor)
        except Exception:
            geographic = any(token in str(crs).upper() for token in ("4326", "GEOGCS", "GEOGCRS", "LONG LAT"))
    elif crs is not None:
        geographic = any(token in str(crs).upper() for token in ("4326", "GEOGCS", "GEOGCRS", "LONG LAT"))

    if geographic:
        center_lon = transform.c + 0.5 * (transform.a + transform.b)
        col_m = haversine_km(center_lon, lat_center,
                             center_lon + transform.a, lat_center + transform.d) * 1000.0
        row_m = haversine_km(center_lon, lat_center,
                             center_lon + transform.b, lat_center + transform.e) * 1000.0
        return max(row_m, 1e-6), max(col_m, 1e-6)

    col_m = math.hypot(transform.a, transform.d) * unit_factor
    row_m = math.hypot(transform.b, transform.e) * unit_factor
    return max(row_m, 1e-6), max(col_m, 1e-6)


def pixel_meters(transform, lat_center, crs=None):
    """Compatibility scalar: geometric mean of local row/column pixel spacing."""
    row_m, col_m = pixel_spacing_meters(transform, lat_center, crs)
    return math.sqrt(row_m * col_m)


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def path_length_km(coords):
    return sum(haversine_km(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1])
               for i in range(len(coords) - 1))


def geodesic_distance_km(lon1, lat1, lon2, lat2):
    return float(_WGS84_GEOD.inv(lon1, lat1, lon2, lat2)[2]) / 1000.0


def geodesic_path_length_km(coords):
    return sum(
        _WGS84_GEOD.inv(start[0], start[1], end[0], end[1])[2]
        for start, end in zip(coords, coords[1:])
    ) / 1000.0


def point_to_polyline_km(lon, lat, polylines):
    """点到折线集合(多条折线)的最小 haversine 距离 km。polylines: [[(lon,lat),...],...]"""
    best = 1e9
    for poly in polylines:
        for i in range(len(poly) - 1):
            lon0, lat0 = poly[i]
            lon1, lat1 = poly[i + 1]
            d_m = math.hypot((lon1 - lon0) * 111000 * math.cos(math.radians((lat0 + lat1) / 2)),
                             (lat1 - lat0) * 111000)
            n = max(int(d_m / 30.0), 1)
            for k in range(n + 1):
                t = k / n
                plon = lon0 + (lon1 - lon0) * t
                plat = lat0 + (lat1 - lat0) * t
                d = haversine_km(lon, lat, plon, plat)
                if d < best:
                    best = d
    return best


def real_length_km(real_geo):
    """参考线(单折线或折线列表/线网)的 WGS84 椭球总长度 km。"""
    if not real_geo:
        return 0.0
    if isinstance(real_geo[0][0], (list, tuple)):
        return sum(geodesic_path_length_km(p) for p in real_geo)
    return geodesic_path_length_km(real_geo)
