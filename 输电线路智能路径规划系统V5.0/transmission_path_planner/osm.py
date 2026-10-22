# transmission_path_planner/osm.py
"""OpenStreetMap 矢量数据的栅格化与距离/密度/用地波段。"""

from .config import *
from .geometry import *
from .data import _fill

_PKG_CACHE = {}
_PKL_CACHE = _PKG_CACHE

class OSMBands(dict):
    """栅格波段字典；状态元数据放在属性中，不影响 ``items()`` 的数组接口。"""

    def __init__(self, *args, metadata=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata or {}


def load_osm_geoms_with_status(pkl_path, geom_type, bbox):
    """读取 Overpass 缓存并返回 (bbox 内几何, 图层状态元数据)。

    ``uncovered_bbox`` 表示请求框完全位于缓存中观察到的要素范围之外；
    ``empty_in_bbox`` 表示范围框相交但没有要素命中。OSM 覆盖完整性仍需另行核验。
    """
    pkl_path = Path(pkl_path)
    metadata = {
        "path": str(pkl_path), "status": None, "source_element_count": 0,
        "geometry_count_in_bbox": 0, "observed_feature_bounds": None,
        "coverage_basis": "observed_feature_extent",
    }
    if not HAS_SHAPELY:
        metadata.update(status="unavailable_dependency", reason="shapely_missing")
        return [], metadata
    if not pkl_path.exists():
        metadata.update(status="missing_file", reason="cache_file_not_found")
        return [], metadata
    if pkl_path in _PKL_CACHE:
        data = _PKL_CACHE[pkl_path]
    else:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            _PKL_CACHE[pkl_path] = data
        except Exception as exc:
            metadata.update(status="read_error", reason=type(exc).__name__)
            return [], metadata
    if not isinstance(data, dict):
        metadata.update(status="invalid_cache", reason="expected_overpass_object")
        return [], metadata
    elems = data.get("elements", [])
    if not isinstance(elems, list):
        metadata.update(status="invalid_cache", reason="elements_not_a_list")
        return [], metadata
    metadata["source_element_count"] = len(elems)
    if not elems:
        metadata.update(status="empty_source", coverage_status="unknown")
        return [], metadata
    box_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    out = []
    extent = [float("inf"), float("inf"), float("-inf"), float("-inf")]
    usable_geometry_count = 0
    for el in elems:
        if not isinstance(el, dict):
            continue
        etype = el.get("type")
        if etype not in ("way", "relation"):
            continue
        tags = el.get("tags", {}) or {}
        # 收集该元素的所有几何坐标列表:
        #  - way: 直接用 top-level geometry;
        #  - relation(multipolygon, OSM landuse 多为 relation): 优先 geometry,
        #    否则遍历 members 中 type=="way" 的成员几何(member 通常无 geometry 顶层字段)。
        geom_lists = []
        if etype == "way":
            g = el.get("geometry")
            if g:
                geom_lists.append(g)
        else:  # relation
            g = el.get("geometry")
            if g:
                geom_lists.append(g)
            else:
                for m in el.get("members", []) or []:
                    if m.get("type") == "way":
                        mg = m.get("geometry")
                        if mg:
                            geom_lists.append(mg)
        for g in geom_lists:
            if not g:
                continue
            try:
                coords = [(float(pt["lon"]), float(pt["lat"])) for pt in g]
            except (KeyError, TypeError, ValueError):
                continue
            if coords:
                xs, ys = zip(*coords)
                extent[0] = min(extent[0], min(xs))
                extent[1] = min(extent[1], min(ys))
                extent[2] = max(extent[2], max(xs))
                extent[3] = max(extent[3], max(ys))
            try:
                if geom_type == "point":
                    if len(coords) >= 1:
                        geom = Point(coords[0])
                    else:
                        continue
                elif geom_type == "polygon":
                    if len(coords) < 3:
                        continue
                    if coords[0] != coords[-1]:
                        coords = coords + [coords[0]]
                    geom = Polygon(coords)
                else:
                    if len(coords) < 2:
                        continue
                    geom = LineString(coords)
            except Exception:
                continue
            if geom.is_empty:
                continue
            usable_geometry_count += 1
            if box_poly.intersects(geom):
                out.append((geom, tags))

    if extent[0] != float("inf"):
        metadata["observed_feature_bounds"] = tuple(extent)
    metadata["geometry_count_in_bbox"] = len(out)
    if out:
        metadata.update(status="available", coverage_status="features_observed")
    elif usable_geometry_count == 0:
        metadata.update(status="no_usable_geometry", coverage_status="unknown")
    elif (extent[2] < bbox[0] or extent[0] > bbox[2]
          or extent[3] < bbox[1] or extent[1] > bbox[3]):
        metadata.update(status="uncovered_bbox", coverage_status="outside_observed_feature_extent")
    else:
        metadata.update(status="empty_in_bbox", coverage_status="unverified_within_feature_extent")
    return out, metadata


def load_osm_geoms(pkl_path, geom_type, bbox):
    """兼容接口：仅返回 bbox 内几何列表。"""
    geoms, _ = load_osm_geoms_with_status(pkl_path, geom_type, bbox)
    return geoms


def rasterize_on(geoms, transform, shape):
    if not geoms:
        return np.zeros(shape, np.uint8)
    shapes = [(g, 1) for g, _ in geoms]
    try:
        return rio_rasterize(shapes, out_shape=shape, transform=transform,
                             fill=0, all_touched=True, dtype=np.uint8)
    except Exception:
        return np.zeros(shape, np.uint8)


def compute_distance_band(geoms, transform, shape, max_dist_m=5000.0):
    on = rasterize_on(geoms, transform, shape)
    if on.sum() == 0:
        return np.full(shape, max_dist_m, np.float32)
    cell_m = pixel_meters(transform, transform.f + transform.e * (shape[0] / 2.0))
    dist_px = distance_transform_edt(1 - on)
    return np.clip(dist_px * cell_m, 0, max_dist_m).astype(np.float32)


def compute_building_density(geoms, transform, shape):
    on = rasterize_on(geoms, transform, shape)
    if on.sum() == 0:
        return np.zeros(shape, np.float32)
    cell_m = pixel_meters(transform, transform.f + transform.e * (shape[0] / 2.0))
    sigma = max(2, int(500 / cell_m))
    density = gaussian_filter(on.astype(np.float32), sigma=sigma)
    cell_area_km2 = (cell_m / 1000.0) ** 2
    return (density / cell_area_km2).astype(np.float32)


def compute_landuse_code(geoms, transform, shape):
    # 无数据/空 band 用 0 作为中性哨兵(与 v3 集成脚本一致); 归一化 landuse_code/8.0 时为 0.0(中性)。
    result = np.full(shape, 0, np.uint8)
    if not geoms:
        return result
    from collections import defaultdict
    groups = defaultdict(list)
    for g, tags in geoms:
        try:
            code = _landuse_tag_to_code(tags)
            groups[code].append(g)
        except Exception:
            continue
    for code in sorted(groups.keys()):
        shapes = [(g, code) for g in groups[code]]
        try:
            m = rio_rasterize(shapes, out_shape=shape, transform=transform,
                              fill=0, all_touched=True, dtype=np.uint8)
            result = np.where(m == code, code, result).astype(np.uint8)
        except Exception:
            continue
    return result


def compute_residential_landuse_mask(geoms, transform, shape):
    """Rasterize mapped ``landuse=residential`` polygons as a soft settlement proxy.

    The mask indicates mapped residential land use only. It is neither a building
    footprint/population estimate nor a legally reliable exclusion zone.
    """
    residential = [
        (geom, tags) for geom, tags in geoms
        if (tags.get("landuse") or "").strip().lower() == "residential"
    ]
    return rasterize_on(residential, transform, shape)


def load_user_constraint_overlays(geojson_path, transform, shape):
    """Rasterize user-supplied WGS84 polygon constraints into hard and soft layers."""
    if not HAS_RASTERIO or not HAS_SHAPELY:
        raise RuntimeError("读取自定义约束图层需要 rasterio 和 shapely")
    path = Path(geojson_path)
    if not path.is_file():
        raise FileNotFoundError(f"自定义约束 GeoJSON 不存在: {path}")
    with path.open(encoding="utf-8-sig") as source_file:
        document = json.load(source_file)
    if not isinstance(document, dict) or document.get("type") != "FeatureCollection" \
            or not isinstance(document.get("features"), list):
        raise ValueError("自定义约束文件必须是 WGS84 GeoJSON FeatureCollection")
    dataset_metadata = document.get("metadata")
    data_sources = document.get("data_sources")
    if dataset_metadata is not None and not isinstance(dataset_metadata, dict):
        raise ValueError("自定义约束 GeoJSON 的 metadata 必须是对象")
    if data_sources is not None and not isinstance(data_sources, list):
        raise ValueError("自定义约束 GeoJSON 的 data_sources 必须是数组")
    declared_crs = document.get("crs")
    if declared_crs:
        if not isinstance(declared_crs, dict):
            raise ValueError("自定义约束 GeoJSON 的 crs 声明无效")
        crs_properties = declared_crs.get("properties") or {}
        if not isinstance(crs_properties, dict):
            raise ValueError("自定义约束 GeoJSON 的 crs 声明无效")
        crs_name = str(crs_properties.get("name", "")).upper()
        if "4326" not in crs_name and "CRS84" not in crs_name:
            raise ValueError("自定义约束 GeoJSON 必须使用 WGS84 经纬度坐标")
    if not document["features"]:
        raise ValueError("自定义约束 GeoJSON 不包含要素")

    from shapely.geometry import shape as shapely_shape

    hard_shapes, soft_shapes = [], []
    sources, categories = set(), set()
    for index, feature in enumerate(document["features"], start=1):
        if not isinstance(feature, dict):
            raise ValueError(f"约束要素 {index} 格式无效")
        properties = feature.get("properties") or {}
        if not isinstance(properties, dict):
            raise ValueError(f"约束要素 {index} 的 properties 格式无效")
        mode = str(properties.get("mode", "")).strip().lower()
        source = str(properties.get("source", "")).strip()
        if mode not in {"hard", "soft"}:
            raise ValueError(f"约束要素 {index} 的 mode 必须是 hard 或 soft")
        if not source:
            raise ValueError(f"约束要素 {index} 缺少 source 来源字段")
        geometry_data = feature.get("geometry")
        if not geometry_data:
            raise ValueError(f"约束要素 {index} 缺少 geometry")
        try:
            geometry = shapely_shape(geometry_data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"约束要素 {index} 的 geometry 格式无效") from exc
        if geometry.is_empty or not geometry.is_valid or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"约束要素 {index} 必须是有效的 Polygon 或 MultiPolygon")
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
            raise ValueError(f"约束要素 {index} 坐标超出 WGS84 经纬度范围")
        if mode == "hard":
            hard_shapes.append((geometry, 1))
        else:
            try:
                weight = float(properties["weight"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"软约束要素 {index} 必须提供非负数值 weight") from exc
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"软约束要素 {index} 的 weight 必须是非负有限数值")
            soft_shapes.append((geometry, weight))
        sources.add(source)
        category = str(properties.get("category", "")).strip()
        if category:
            categories.add(category)

    height, width = shape
    hard_mask = rio_rasterize(
        hard_shapes, out_shape=(height, width), transform=transform,
        fill=0, all_touched=True, dtype=np.uint8,
    ) if hard_shapes else np.zeros((height, width), dtype=np.uint8)
    soft_cost = rio_rasterize(
        soft_shapes, out_shape=(height, width), transform=transform,
        fill=0, all_touched=True, dtype=np.float32,
        merge_alg=rasterio.enums.MergeAlg.add,
    ) if soft_shapes else np.zeros((height, width), dtype=np.float32)
    metadata = {
        "path": str(path.resolve()),
        "status": "loaded",
        "feature_count": len(document["features"]),
        "hard_feature_count": len(hard_shapes),
        "soft_feature_count": len(soft_shapes),
        "sources": sorted(sources),
        "categories": sorted(categories),
        "hard_cell_count": int(hard_mask.sum()),
        "soft_cell_count": int(np.count_nonzero(soft_cost)),
    }
    if dataset_metadata is not None:
        metadata["dataset_metadata"] = dataset_metadata
    if data_sources is not None:
        metadata["data_sources"] = data_sources
    return {"hard_mask": hard_mask, "soft_cost": soft_cost, "metadata": metadata}


def compute_vegetation_height(geoms, transform, shape):
    """从植被/林地类 OSM 多边形推导植被高度代理 (m)。"""
    result = np.zeros(shape, np.float32)
    if not geoms:
        return result
    from collections import defaultdict
    groups = defaultdict(list)
    for g, tags in geoms:
        lu = (tags.get("landuse") or "").lower()
        nat = (tags.get("natural") or "").lower()
        if "wood" in nat or "forest" in nat or "tree" in nat:
            h = 20.0
        elif "scrub" in nat or "grass" in nat or "grassland" in nat or "heath" in nat:
            h = 5.0
        elif "wetland" in nat or "marsh" in nat:
            h = 2.0
        elif "forest" in lu or "wood" in lu:
            h = 20.0
        else:
            h = 3.0
        groups[h].append(g)
    for h, glist in groups.items():
        m = rasterize_on([(g, {}) for g in glist], transform, shape)
        result = np.where(m > 0, h, result).astype(np.float32)
    return result


def compute_osm_bands(dem, transform, bbox):
    """计算 OSM 波段与风险代理；图层状态可从返回值的 ``metadata`` 属性读取。

    Building footprints provide a hard footprint mask and settlement proxy;
    residential land use provides a soft settlement proxy and model feature.
    Other layers are model features only and are not used by the default rule
    cost. Protected areas are reported for inspection only.
    """
    H, W = dem.shape
    layer_specs = {
        "roads": ("taiwan_roads.pkl", "line", "feature_model_input_only_not_default_rule_cost"),
        "water": ("taiwan_water.pkl", "line", "feature_model_input_only_not_default_rule_cost"),
        "railways": ("taiwan_railways.pkl", "line", "feature_model_input_only_not_default_rule_cost"),
        "faults": ("taiwan_faults.pkl", "line", "feature_model_input_only_not_default_rule_cost"),
        "landuse": ("taiwan_landuse.pkl", "polygon", "residential_soft_proxy_and_model_feature"),
        "buildings": ("taiwan_buildings.pkl", "polygon", "hard_footprint_and_settlement_proxy"),
        "vegetation": ("taiwan_vegetation.pkl", "polygon", "feature_model_input_only_not_default_rule_cost"),
        "protected_areas": ("taiwan_protected.pkl", "polygon", "informational_only"),
    }
    layers = {}
    layer_metadata = {}
    for name, (filename, geom_type, role) in layer_specs.items():
        layers[name], layer_metadata[name] = load_osm_geoms_with_status(
            DOWNLOADED_DIR / filename, geom_type, bbox)
        layer_metadata[name]["planning_role"] = role
    roads = layers["roads"]
    water = layers["water"]
    railways = layers["railways"]
    faults = layers["faults"]
    landuse = layers["landuse"]
    buildings = layers["buildings"]
    vegetation = layers["vegetation"]
    print(f"  OSM 几何数: 道路{len(roads)} 水域{len(water)} 铁路{len(railways)} "
          f"断裂{len(faults)} 土地利用{len(landuse)} 建筑{len(buildings)} 植被{len(vegetation)}")

    bands = OSMBands(metadata={"layers": layer_metadata})
    bands["dist_road"] = compute_distance_band(roads, transform, (H, W), 5000.0)
    bands["dist_water"] = compute_distance_band(water, transform, (H, W), 5000.0)
    bands["dist_railway"] = compute_distance_band(railways, transform, (H, W), 5000.0)
    bands["dist_fault"] = compute_distance_band(faults, transform, (H, W), 10000.0)
    bands["landuse_code"] = compute_landuse_code(landuse, transform, (H, W))
    bands["residential_landuse_mask"] = compute_residential_landuse_mask(landuse, transform, (H, W))
    bands["building_density"] = compute_building_density(buildings, transform, (H, W))
    bands["building_mask"] = rasterize_on(buildings, transform, (H, W))
    bands["building_distance_m"] = compute_distance_band(buildings, transform, (H, W), 5000.0)
    bands["vegetation_height"] = compute_vegetation_height(vegetation, transform, (H, W))

    # 风险代理波段（21-25）— 由地形/位置推导
    z = _fill(dem).astype(np.float32)
    lat_center = transform.f + transform.e * (H / 2.0)  # 窗口纬度中心（修正：之前误用 0）
    row_m, col_m = pixel_spacing_meters(transform, lat_center)
    dy, dx = np.gradient(z, row_m, col_m)
    slope = np.degrees(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
    aspect = np.degrees(np.arctan2(-dy, dx)) % 360
    elev_n = np.clip(z / 3000.0, 0, 1)
    slope_n = np.clip(slope / 45.0, 0, 1)
    # 台风：东南坡 + 低海拔 + 陡坡
    aspect_score = np.where((aspect >= 45) & (aspect < 225), 1.0, 0.3)
    bands["typhoon_risk"] = np.clip(0.4 * slope_n + 0.35 * aspect_score + 0.25 * (1 - elev_n), 0, 1).astype(np.float32)
    # 地震：基础 0.6 + 坡度
    bands["seismic_risk"] = np.clip(0.6 + 0.3 * slope_n, 0, 1).astype(np.float32)
    # 滑坡：坡度 + 粗糙度
    rough = np.nan_to_num(ndi.generic_filter(z, np.nanstd, size=3), nan=0.0) if HAS_SCIPY else np.zeros_like(z)
    bands["landslide_risk"] = np.clip(0.5 * slope_n + 0.3 * np.clip(rough / 200.0, 0, 1), 0, 1).astype(np.float32)
    # 覆冰：高海拔
    bands["ice_cover_risk"] = np.clip((z - 2000) / 2000.0, 0, 1).astype(np.float32)
    # 雷击：坡度 + 海拔
    bands["lightning_risk"] = np.clip(0.6 * slope_n + 0.4 * elev_n, 0, 1).astype(np.float32)
    return bands
