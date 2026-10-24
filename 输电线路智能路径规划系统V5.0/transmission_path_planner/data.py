# transmission_path_planner/data.py
"""DEM 读取、地形因子计算、山体阴影。"""

from .config import *
from .geometry import *

def find_dem():
    for p in DEM_CANDIDATES:
        if p.exists():
            return p
    return None


POPULATION_DENSITY_NORMALIZATION_PEOPLE_PER_KM2 = 10000.0


def _matching_population_stac_metadata(population_path):
    source_name = Path(population_path).name
    for metadata_path in Path(population_path).parent.glob("stac_item*.json"):
        try:
            item = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        href = item.get("assets", {}).get("data", {}).get("href", "")
        asset_name = href.split("?", 1)[0].rsplit("/", 1)[-1]
        if asset_name == source_name:
            return item, metadata_path
    return {}, None


def load_population_density_window(population_path, shape, transform, crs,
                                   required_mask=None):
    """Warp people-per-cell counts to the DEM grid as people/km².

    Population pixels are converted using WGS84 ellipsoidal pixel areas before
    area-weighted alignment. Missing population data remains NaN and is
    reported in metadata and remains NaN, including wherever ``required_mask`` is true.
    """
    if not HAS_RASTERIO:
        raise RuntimeError("人口栅格需要 rasterio；请安装 rasterio 后重试")

    from pyproj import Geod
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import Window, from_bounds

    required = (np.ones(shape, dtype=bool) if required_mask is None
                else np.asarray(required_mask, dtype=bool))
    if required.shape != tuple(shape):
        raise ValueError("required_mask shape must match population destination grid")

    with rasterio.open(population_path) as src:
        if src.count != 1:
            raise ValueError("人口栅格必须是单波段")
        if src.crs is None or src.crs != rasterio.crs.CRS.from_epsg(4326):
            raise ValueError("人口栅格必须使用 EPSG:4326")
        if (src.transform.b != 0 or src.transform.d != 0
                or src.transform.a <= 0 or src.transform.e >= 0):
            raise ValueError("人口栅格必须是北向上的 EPSG:4326 网格")
        source_tags = src.tags()
        stac_item, stac_path = _matching_population_stac_metadata(population_path)
        stac_properties = stac_item.get("properties", {})
        source_description = (
            source_tags.get("TIFFTAG_IMAGEDESCRIPTION")
            or source_tags.get("TIFFTAG_DOCUMENTNAME")
            or stac_properties.get("title")
        )
        source_label = " ".join(filter(None, (
            source_description, source_tags.get("TIFFTAG_DOCUMENTNAME"),
            Path(population_path).name,
        )))
        provider = "WorldPop" if "worldpop" in source_label.casefold() else None
        copyright_tag = source_tags.get("TIFFTAG_COPYRIGHT")
        license_id = (
            "CC-BY-4.0" if copyright_tag and "cc-by-4.0" in copyright_tag.casefold()
            else copyright_tag or stac_item.get("license")
        )
        license_url = (
            "https://creativecommons.org/licenses/by/4.0/"
            if license_id == "CC-BY-4.0" else None
        )
        doi = stac_properties.get("Digital Object Identifier (DOI)")
        if doi is None:
            doi = next((link.get("href") for link in stac_item.get("links", [])
                        if link.get("rel") == "doi"), None)

        dst_height, dst_width = map(int, shape)
        left = transform.c
        top = transform.f
        right = left + transform.a * dst_width
        bottom = top + transform.e * dst_height
        try:
            window = from_bounds(
                min(left, right), min(bottom, top), max(left, right), max(bottom, top),
                src.transform,
            ).round_offsets().round_lengths()
            window = window.intersection(Window(0, 0, src.width, src.height))
        except Exception as exc:
            raise ValueError("人口栅格与 DEM 规划范围没有重叠") from exc
        if window.width <= 0 or window.height <= 0:
            raise ValueError("人口栅格与 DEM 规划范围没有重叠")

        counts = src.read(1, window=window).astype(np.float64)
        valid = src.read_masks(1, window=window) > 0
        if src.nodata is not None:
            valid &= counts != src.nodata
        valid &= np.isfinite(counts)
        if np.any(counts[valid] < 0):
            raise ValueError("人口栅格包含负的人口数")

        source_transform = src.window_transform(window)
        geod = Geod(ellps="WGS84")
        row_areas_m2 = np.empty(counts.shape[0], dtype=np.float64)
        x0 = source_transform.c
        x1 = x0 + source_transform.a
        for row in range(counts.shape[0]):
            y0 = source_transform.f + row * source_transform.e
            y1 = y0 + source_transform.e
            area = geod.polygon_area_perimeter(
                [x0, x1, x1, x0], [y0, y0, y1, y1]
            )[0]
            row_areas_m2[row] = abs(area)
        if np.any(row_areas_m2 <= 0) or not np.isfinite(row_areas_m2).all():
            raise ValueError("无法计算人口栅格源像元面积")

        density = counts / row_areas_m2[:, None] * 1_000_000.0
        density[~valid] = np.nan
        aligned = np.full(shape, np.nan, dtype=np.float32)
        source_coverage = valid.astype(np.float32)
        aligned_coverage = np.zeros(shape, dtype=np.float32)
        reproject(
            source=source_coverage,
            destination=aligned_coverage,
            src_transform=source_transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=0.0,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )
        reproject(
            source=density.astype(np.float32),
            destination=aligned,
            src_transform=source_transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
            init_dest_nodata=True,
        )
        aligned[aligned_coverage < 1.0 - 1e-6] = np.nan
        missing = required & (~np.isfinite(aligned) | (aligned_coverage < 1.0 - 1e-6))
        required_cells = int(required.sum())
        valid_required_cells = required_cells - int(missing.sum())
        metadata = {
            "status": ("complete" if not missing.any() else
                       "partial" if valid_required_cells else "no_coverage"),
            "source_path": str(Path(population_path).resolve()),
            "source_crs": src.crs.to_string(),
            "source_pixel_size_degrees": [abs(src.transform.a), abs(src.transform.e)],
            "source_nodata": src.nodata,
            "source_value_unit": "people_per_cell",
            "aligned_value_unit": "people_per_km2",
            "source_metadata": {
                "provider": provider,
                "asset_provider": stac_properties.get("provider"),
                "dataset_title": source_description,
                "document_name": source_tags.get("TIFFTAG_DOCUMENTNAME"),
                "copyright_tag": copyright_tag,
                "license_id": license_id,
                "license_url": license_url,
                "release": stac_properties.get("Release"),
                "version": stac_properties.get("Version"),
                "year": stac_properties.get("year"),
                "attribution": (
                    f"{provider}; dataset title: {source_description or Path(population_path).name}"
                    if provider else None
                ),
                "doi": doi,
                "metadata_file": str(stac_path.resolve()) if stac_path else None,
                "metadata_basis": ("embedded GeoTIFF tags and matching adjacent STAC item"
                                   if stac_path else
                                   "embedded GeoTIFF tags and source filename"),
            },
            "resampling": "area_weighted_average_on_density",
            "required_dem_cell_count": required_cells,
            "valid_required_cell_count": valid_required_cells,
            "missing_required_cell_count": int(missing.sum()),
            "coverage_ratio": (valid_required_cells / required_cells
                               if required_cells else 1.0),
            "nodata_policy": "preserve_as_nan_for_explicit_unknown_policy",
            "normalization_scale_people_per_km2": (
                POPULATION_DENSITY_NORMALIZATION_PEOPLE_PER_KM2),
        }
        return aligned, metadata


def population_density_cost(density_people_per_km2):
    """Convert people/km² to a fixed-scale [0, 1] soft-cost proxy."""
    density = np.asarray(density_people_per_km2, dtype=np.float32)
    valid = np.isfinite(density)
    if np.any(density[valid] < 0):
        raise ValueError("人口密度不能为负数")
    scale = POPULATION_DENSITY_NORMALIZATION_PEOPLE_PER_KM2
    cost = np.full(density.shape, np.nan, dtype=np.float32)
    cost[valid] = np.clip(
        np.log1p(density[valid]) / math.log1p(scale), 0.0, 1.0)
    return cost


def population_cost_for_routing(density_people_per_km2):
    """Penalize unknown cells conservatively, retaining their uncertainty mask.

    The fixed value is a search penalty, not an estimate of population density.
    """
    proxy = population_density_cost(density_people_per_km2)
    unknown = ~np.isfinite(proxy)
    return np.where(unknown, 1.0, proxy).astype(np.float32), unknown


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
