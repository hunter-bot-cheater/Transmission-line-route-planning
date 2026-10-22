# transmission_path_planner/data.py
"""DEM 读取、地形因子计算、山体阴影。"""

from .config import *
from .geometry import *

def find_dem():
    for p in DEM_CANDIDATES:
        if p.exists():
            return p
    return None


def load_dem_window(dem_path, minlon, minlat, maxlon, maxlat, max_side=WORK_MAX_SIDE):
    """读取 DEM 在给定 bbox 内的窗口，并重采样到最长边 <= max_side。
    返回 (dem, transform, crs, pixel_m)。"""
    if HAS_RASTERIO:
        with rasterio.open(dem_path) as src:
            if src.crs is None or src.crs != rasterio.crs.CRS.from_epsg(4326):
                raise ValueError("当前命令行规划仅支持带 EPSG:4326 坐标系的 DEM，请先重投影")
            crs = str(src.crs)
            # 限制到 DEM 实际范围
            dem_left, dem_bottom, dem_right, dem_top = src.bounds
            minlon = max(minlon, dem_left)
            maxlon = min(maxlon, dem_right)
            minlat = max(minlat, dem_bottom)
            maxlat = min(maxlat, dem_top)
            window = from_bounds(minlon, minlat, maxlon, maxlat, src.transform)
            window = window.round_offsets().round_lengths()
            Hn, Wn = src.height, src.width
            window = type(window)(
                col_off=max(0, window.col_off),
                row_off=max(0, window.row_off),
                width=min(window.width, Wn - window.col_off),
                height=min(window.height, Hn - window.row_off),
            )
            arr = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
            nodata = src.nodata
    elif tifffile is not None:
        with tifffile.TiffFile(str(dem_path)) as tf:
            gm = tf.geotiff_metadata
            sx = gm["ModelPixelScale"][0]
            tie = gm["ModelTiepoint"]
            a = sx
            e = -sx
            c = tie[3] - tie[0] * sx
            f = tie[4] - tie[1] * (-sx)
            transform = Affine(a, 0, c, 0, e, f)
            crs = "EPSG:4326"
            full = tf.pages[0].asarray().astype(np.float32)
            nodata = -32768.0
            # bbox -> slice
            c0 = max(0, int((minlon - c) / a))
            c1 = min(full.shape[1], int((maxlon - c) / a) + 1)
            r1 = max(0, int((maxlat - f) / e))   # e<0
            r0 = min(full.shape[0], int((minlat - f) / e) + 1)
            if r0 > r1:
                r0, r1 = r1, r0
            arr = full[r0:r1, c0:c1].copy()
            transform = Affine(a, 0, c + c0 * a, 0, e, f + r0 * e)
        print("  [warn] 使用 tifffile 全量读取兜底路径")
    else:
        raise RuntimeError("需要 rasterio 或 tifffile 读取 DEM")

    # nodata 处理
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    arr = np.where((arr < -100) | (arr > 9000), np.nan, arr)

    # 尽量保留输入 DEM 的原生网格；只有窗口超出 max_side 时才降采样。
    Hn, Wn = arr.shape
    source_lat = transform.f + transform.e * Hn / 2.0
    source_row_m, source_col_m = pixel_spacing_meters(transform, source_lat, crs)
    source_pm = math.sqrt(source_row_m * source_col_m)
    scale = max(Hn, Wn) / max_side
    if scale > 1.01:
        Hr = max(1, int(round(Hn / scale)))
        Wr = max(1, int(round(Wn / scale)))
        if HAS_SCIPY:
            from skimage.transform import resize
            arr = resize(arr, (Hr, Wr), order=1, preserve_range=True).astype(np.float32)
        left = transform.c
        top = transform.f
        right = left + transform.a * Wn
        bottom = top + transform.e * Hn
        a_new = (right - left) / Wr
        e_new = (bottom - top) / Hr
        transform = Affine(a_new, 0, left, 0, e_new, top)

    lat0 = transform.f + transform.e * (arr.shape[0] / 2.0)
    pm = pixel_meters(transform, lat0, crs)
    resample_factor = max(1.0, scale)
    print(
        f"  DEM 输入窗口 {Wn}x{Hn} / 规划网格 {arr.shape[1]}x{arr.shape[0]}  "
        f"原始像元≈{source_pm:.1f}m / 实际像元≈{pm:.1f}m  "
        f"降采样倍率={resample_factor:.2f}x  CRS={crs}"
    )
    if resample_factor > 1.25:
        print("  [warn] DEM 为适配工作网格已明显降采样；增大 --max-side 可保留更多地形细节")
    return arr, transform, crs, pm


def _fill(arr, fill=None):
    if fill is None:
        fill = np.nanmedian(arr) if np.isfinite(arr).any() else 0.0
    return np.where(np.isfinite(arr), arr, fill)


def _terrain_gradient(z, row_m, col_m):
    dy = np.gradient(z, row_m, axis=0) if z.shape[0] > 1 else np.zeros_like(z)
    dx = np.gradient(z, col_m, axis=1) if z.shape[1] > 1 else np.zeros_like(z)
    return dy, dx


def compute_terrain_factors(dem, pixel_m, transform=None, crs=None):
    if transform is not None:
        row_m, col_m = pixel_spacing_meters(
            transform, (transform.f + transform.e * dem.shape[0] / 2), crs)
    elif np.isscalar(pixel_m):
        row_m = col_m = float(pixel_m)
    else:
        row_m, col_m = map(float, pixel_m)
    neighborhood_m = math.sqrt(row_m * col_m)
    z = _fill(dem).astype(np.float32)
    dy, dx = _terrain_gradient(z, row_m, col_m)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
    aspect = np.arctan2(dy, -dx)
    aspect_cos = np.cos(aspect)
    aspect_sin = np.sin(aspect)

    def local_std(size):
        values = z.astype(np.float64, copy=False)
        mean = ndi.uniform_filter(values, size=size, mode="reflect")
        mean_sq = ndi.uniform_filter(values * values, size=size, mode="reflect")
        np.subtract(mean_sq, mean * mean, out=mean_sq)
        np.maximum(mean_sq, 0.0, out=mean_sq)
        np.sqrt(mean_sq, out=mean_sq)
        return mean_sq.astype(np.float32)

    tri = local_std(3) if HAS_SCIPY else np.abs(z - z)

    def px_of_m(m):
        return max(1, int(round(m / neighborhood_m)))

    def tpi_at(radius_px):
        if HAS_SCIPY:
            size = 2 * radius_px + 1
            mean = ndi.uniform_filter(z, size=size, mode="reflect")
            return z - mean
        return z - z

    def rough_at(radius_px):
        if HAS_SCIPY:
            return local_std(2 * radius_px + 1)
        return np.abs(z - z)

    tpi_100 = tpi_at(px_of_m(100))
    tpi_300 = tpi_at(px_of_m(300))
    tpi_900 = tpi_at(px_of_m(900))
    rough_3 = rough_at(px_of_m(100))
    rough_9 = rough_at(px_of_m(300))
    rough_27 = rough_at(px_of_m(900))

    if HAS_SCIPY:
        curv = ndi.laplace(z) / (neighborhood_m ** 2)
    else:
        curv = np.zeros_like(z)
    curv = np.nan_to_num(curv, nan=0.0)

    return {
        "elevation": z,
        "slope": slope.astype(np.float32),
        "aspect_cos": aspect_cos.astype(np.float32),
        "aspect_sin": aspect_sin.astype(np.float32),
        "tri": np.nan_to_num(tri, nan=0.0).astype(np.float32),
        "tpi_100": tpi_100.astype(np.float32),
        "tpi_300": tpi_300.astype(np.float32),
        "tpi_900": tpi_900.astype(np.float32),
        "profile_curvature": curv.astype(np.float32),
        "plan_curvature": curv.astype(np.float32),
        "roughness_3": np.nan_to_num(rough_3, nan=0.0).astype(np.float32),
        "roughness_9": np.nan_to_num(rough_9, nan=0.0).astype(np.float32),
        "roughness_27": np.nan_to_num(rough_27, nan=0.0).astype(np.float32),
    }


def _landuse_tag_to_code(tags):
    if not tags:
        return 0
    lu = (tags.get("landuse") or "").lower()
    nat = (tags.get("natural") or "").lower()
    mapping = {
        "forest": 1, "wood": 1, "scrub": 1, "heath": 1,
        "farmland": 2, "farmyard": 2, "orchard": 2, "vineyard": 2,
        "meadow": 2, "grass": 2, "grassland": 2, "greenfield": 2,
        "bare_rock": 3, "bare_ground": 3, "scree": 3, "sand": 3, "beach": 3,
        "residential": 4, "retail": 4, "commercial": 4, "urban": 4,
        "industrial": 5, "quarry": 5, "brownfield": 5, "construction": 5,
        "reservoir": 6, "basin": 6, "water": 6, "salt_pond": 6, "aquaculture": 6,
        "wetland": 7, "marsh": 7, "swamp": 7, "mud": 7,
        "cemetery": 8, "recreation_ground": 8, "village_green": 8,
        "military": 8, "allotments": 8, "plant_nursery": 8,
    }
    for key, code in mapping.items():
        if key in lu or key in nat:
            return code
    return 0


_PKL_CACHE = {}


def _hillshade(dem):
    z = _fill(dem)
    if min(z.shape) < 2:
        return np.full(z.shape, 0.5, dtype=np.float32)
    ls = LightSource(azdeg=315, altdeg=45)
    return np.clip(ls.hillshade(z, vert_exag=max(30.0, 4000.0 / max(np.nanmax(z) - np.nanmin(z), 1.0))), 0, 1)
