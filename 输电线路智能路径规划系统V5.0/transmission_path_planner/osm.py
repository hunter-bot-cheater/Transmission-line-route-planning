"""OpenStreetMap 矢量数据的栅格化与距离/密度/用地波段。"""

import ast
import json
from shapely import make_valid
from shapely.ops import unary_union

from .config import *
from .geometry import *
from .data import _fill, _terrain_gradient, _landuse_tag_to_code

_PKG_CACHE = {}
_PKL_CACHE = _PKG_CACHE
_OFFICIAL_BUILDING_INFO_CACHE = {}


def _read_geojson_tail_value(path, key):
    """Read a small top-level metadata value placed after a large GeoJSON feature array."""
    with Path(path).open("rb") as source_file:
        source_file.seek(max(0, Path(path).stat().st_size - 1024 * 1024))
        tail = source_file.read().decode("utf-8", errors="ignore")
    marker = json.dumps(key)
    position = tail.rfind(marker)
    if position < 0:
        raise ValueError(f"官方建筑 GeoJSON 缺少顶层 {key}")
    colon = tail.find(":", position + len(marker))
    if colon < 0:
        raise ValueError(f"官方建筑 GeoJSON 的 {key} 格式无效")
    value_start = colon + 1
    while value_start < len(tail) and tail[value_start].isspace():
        value_start += 1
    try:
        value, _ = json.JSONDecoder().raw_decode(tail, value_start)
    except json.JSONDecodeError as exc:
        raise ValueError(f"官方建筑 GeoJSON 的 {key} 格式无效") from exc
    return value

class OSMBands(dict):
    """栅格波段字典；状态元数据放在属性中，不影响 ``items()`` 的数组接口。"""

    def __init__(self, *args, metadata=None, geometries=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = metadata or {}
        self.geometries = geometries or {}


def load_osm_geoms_with_status(pkl_path, geom_type, bbox):
    """读取 Overpass 缓存并返回 (bbox 内几何, 图层状态元数据)。

    ``uncovered_bbox`` 表示请求框完全位于缓存中观察到的要素范围之外；
    ``empty_in_bbox`` 表示范围框相交但没有要素命中。跳过范围外非有限坐标要素时，
    ``coverage_status`` 标记为 ``partial``，因为无法确认缺失坐标是否影响请求范围。
    OSM 覆盖完整性仍需另行核验。
    """
    pkl_path = Path(pkl_path)
    metadata = {
        "path": str(pkl_path), "status": None, "source_element_count": 0,
        "geometry_count_in_bbox": 0, "observed_feature_bounds": None,
        "coverage_basis": "observed_feature_extent",
        "cache_generator": None, "source_snapshot_timestamp": None,
        "source_snapshot_status": "unknown_not_recorded",
        "tag_wrapper_count": 0, "tag_parse_error_count": 0,
        "geometry_repaired_count": 0,
        "nonfinite_geometry_outside_bbox_count": 0,
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
    metadata["cache_generator"] = data.get("generator")
    osm_metadata = data.get("osm3s")
    if isinstance(osm_metadata, dict):
        snapshot_timestamp = osm_metadata.get("timestamp_osm_base")
        if snapshot_timestamp:
            metadata.update(
                source_snapshot_timestamp=str(snapshot_timestamp),
                source_snapshot_status="recorded",
            )
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
        raw_tags = el.get("tags", {}) or {}
        tags = raw_tags
        if isinstance(raw_tags, dict) and set(raw_tags) == {"tags"}:
            raw_tags = raw_tags["tags"]
            metadata["tag_wrapper_count"] += 1
        if isinstance(raw_tags, str):
            try:
                raw_tags = ast.literal_eval(raw_tags)
            except (SyntaxError, ValueError):
                raw_tags = None
            if not isinstance(raw_tags, dict):
                metadata["tag_parse_error_count"] += 1
                tags = {}
            else:
                tags = raw_tags
        elif isinstance(raw_tags, dict):
            tags = raw_tags
        elif raw_tags:
            metadata["tag_parse_error_count"] += 1
            tags = {}
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
            finite_coords = [point for point in coords
                             if math.isfinite(point[0]) and math.isfinite(point[1])]
            has_nonfinite = len(finite_coords) != len(coords)
            if has_nonfinite and not finite_coords:
                raise ValueError(
                    f"OSM 要素 {el.get('id', 'unknown')} 含无法定位的非有限坐标，拒绝规划")
            bounds_coords = finite_coords if has_nonfinite else coords
            if bounds_coords:
                min_x = min(point[0] for point in bounds_coords)
                min_y = min(point[1] for point in bounds_coords)
                max_x = max(point[0] for point in bounds_coords)
                max_y = max(point[1] for point in bounds_coords)
                extent[0] = min(extent[0], min_x)
                extent[1] = min(extent[1], min_y)
                extent[2] = max(extent[2], max_x)
                extent[3] = max(extent[3], max_y)

            if has_nonfinite:
                if (max_x < bbox[0] or min_x > bbox[2]
                        or max_y < bbox[1] or min_y > bbox[3]):
                    metadata["nonfinite_geometry_outside_bbox_count"] += 1
                    continue
                raise ValueError(
                    f"OSM 要素 {el.get('id', 'unknown')} 在请求范围内含非有限坐标，拒绝规划")

            min_coords = 1 if geom_type == "point" else 3 if geom_type == "polygon" else 2
            if len(coords) < min_coords:
                continue
            usable_geometry_count += 1
            if (max_x < bbox[0] or min_x > bbox[2]
                    or max_y < bbox[1] or min_y > bbox[3]):
                continue
            try:
                if geom_type == "point":
                    geom = Point(coords[0])
                elif geom_type == "polygon":
                    if coords[0] != coords[-1]:
                        coords = coords + [coords[0]]
                    geom = Polygon(coords)
                else:
                    geom = LineString(coords)
            except Exception:
                usable_geometry_count -= 1
                continue
            if geom.is_empty:
                usable_geometry_count -= 1
                continue
            if geom_type == "polygon" and not geom.is_valid:
                try:
                    geom = make_valid(geom)
                    if geom.geom_type == "GeometryCollection":
                        polygons = []
                        for part in geom.geoms:
                            if part.geom_type == "Polygon":
                                polygons.append(part)
                            elif part.geom_type == "MultiPolygon":
                                polygons.extend(part.geoms)
                        geom = unary_union(polygons) if polygons else geom
                except Exception as exc:
                    raise ValueError(
                        f"OSM 面要素 {el.get('id', 'unknown')} 几何修复失败") from exc
                if (geom.is_empty or not geom.is_valid
                        or geom.geom_type not in {"Polygon", "MultiPolygon"}):
                    raise ValueError(
                        f"OSM 面要素 {el.get('id', 'unknown')} 几何修复后仍不可用")
                metadata["geometry_repaired_count"] += 1
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
    if metadata["nonfinite_geometry_outside_bbox_count"]:
        metadata.update(
            coverage_status="partial",
            coverage_reason="nonfinite_geometry_skipped_outside_bbox",
        )
    return out, metadata


def load_osm_geoms(pkl_path, geom_type, bbox):
    """兼容接口：仅返回 bbox 内几何列表。"""
    geoms, _ = load_osm_geoms_with_status(pkl_path, geom_type, bbox)
    return geoms


def rasterize_on(geoms, transform, shape):
    if not geoms:
        return np.zeros(shape, np.uint8)
    shapes = [(g, 1) for g, _ in geoms]
    return rio_rasterize(shapes, out_shape=shape, transform=transform,
                         fill=0, all_touched=True, dtype=np.uint8)


def _pixel_spacing_for_shape(transform, shape, crs=None):
    lat_center = transform.f + transform.b * (shape[1] / 2.0) + transform.e * (shape[0] / 2.0)
    return pixel_spacing_meters(transform, lat_center, crs)


def compute_distance_band(geoms, transform, shape, max_dist_m=5000.0, *, crs=None):
    row_m, col_m = _pixel_spacing_for_shape(transform, shape, crs)
    pad_rows = int(math.ceil(max_dist_m / row_m)) + 1
    pad_cols = int(math.ceil(max_dist_m / col_m)) + 1
    padded_shape = (shape[0] + 2 * pad_rows, shape[1] + 2 * pad_cols)
    padded_transform = transform * Affine.translation(-pad_cols, -pad_rows)
    on = rasterize_on(geoms, padded_transform, padded_shape)
    if on.sum() == 0:
        return np.full(shape, max_dist_m, np.float32)
    distance_m = distance_transform_edt(1 - on, sampling=(row_m, col_m))
    cropped = distance_m[pad_rows:pad_rows + shape[0],
                         pad_cols:pad_cols + shape[1]]
    return np.clip(cropped, 0, max_dist_m).astype(np.float32)


def compute_building_density(geoms, transform, shape, *, crs=None):
    """Return a 500 m-smoothed share of cells touched by mapped building footprints."""
    row_m, col_m = _pixel_spacing_for_shape(transform, shape, crs)
    sigma = (500.0 / row_m, 500.0 / col_m)
    pad_rows = int(math.ceil(4.0 * sigma[0]))
    pad_cols = int(math.ceil(4.0 * sigma[1]))
    padded_shape = (shape[0] + 2 * pad_rows, shape[1] + 2 * pad_cols)
    padded_transform = transform * Affine.translation(-pad_cols, -pad_rows)
    on = rasterize_on(geoms, padded_transform, padded_shape)
    if on.sum() == 0:
        return np.zeros(shape, np.float32)
    smooth = gaussian_filter(on.astype(np.float32), sigma=sigma, mode="constant")
    return np.clip(smooth[pad_rows:pad_rows + shape[0],
                          pad_cols:pad_cols + shape[1]], 0.0, 1.0).astype(np.float32)


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


def load_land_mask_geojson(geojson_path, transform, shape):
    """Rasterize WGS84 land polygons as an optional hard traversability mask."""
    if not HAS_RASTERIO or not HAS_SHAPELY:
        raise RuntimeError("读取陆域 GeoJSON 需要 rasterio 和 shapely")
    path = Path(geojson_path)
    if not path.is_file():
        raise FileNotFoundError(f"陆域 GeoJSON 不存在: {path}")
    with path.open(encoding="utf-8-sig") as source_file:
        document = json.load(source_file)
    if (not isinstance(document, dict)
            or document.get("type") != "FeatureCollection"
            or not isinstance(document.get("features"), list)
            or not document["features"]):
        raise ValueError("陆域文件必须是包含要素的 WGS84 GeoJSON FeatureCollection")
    declared_crs = document.get("crs")
    if declared_crs:
        if not isinstance(declared_crs, dict):
            raise ValueError("陆域 GeoJSON 的 crs 声明无效")
        crs_properties = declared_crs.get("properties") or {}
        if not isinstance(crs_properties, dict):
            raise ValueError("陆域 GeoJSON 的 crs 声明无效")
        crs_name = str(crs_properties.get("name", "")).strip()
        try:
            parsed_crs = rasterio.crs.CRS.from_user_input(crs_name)
        except (TypeError, ValueError) as exc:
            raise ValueError("陆域 GeoJSON 必须使用 WGS84 经纬度坐标") from exc
        if (parsed_crs.to_epsg() != 4326
                and parsed_crs.to_string().upper() != "OGC:CRS84"):
            raise ValueError("陆域 GeoJSON 必须使用 WGS84 经纬度坐标")

    from shapely.geometry import shape as shapely_shape

    land_shapes = []
    repaired_feature_indices = []
    for index, feature in enumerate(document["features"], start=1):
        if not isinstance(feature, dict) or not feature.get("geometry"):
            raise ValueError(f"陆域要素 {index} 缺少有效 geometry")
        try:
            geometry = shapely_shape(feature["geometry"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"陆域要素 {index} 的 geometry 格式无效") from exc
        if (geometry.is_empty
                or geometry.geom_type not in {"Polygon", "MultiPolygon"}):
            raise ValueError(f"陆域要素 {index} 必须是有效的 Polygon 或 MultiPolygon")
        if not geometry.is_valid:
            if (geometry.geom_type != "MultiPolygon"
                    or not all(part.is_valid for part in geometry.geoms)):
                raise ValueError(f"陆域要素 {index} 必须是有效的 Polygon 或 MultiPolygon")
            geometry = unary_union(list(geometry.geoms))
            if (geometry.is_empty or not geometry.is_valid
                    or geometry.geom_type not in {"Polygon", "MultiPolygon"}):
                raise ValueError(f"陆域要素 {index} 的重叠多边形无法安全合并")
            repaired_feature_indices.append(index)
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        if (not all(math.isfinite(value) for value in geometry.bounds)
                or min_lon < -180 or max_lon > 180
                or min_lat < -90 or max_lat > 90):
            raise ValueError(f"陆域要素 {index} 坐标超出 WGS84 经纬度范围")
        land_shapes.append((geometry, 1))

    land_mask = rio_rasterize(
        land_shapes, out_shape=shape, transform=transform,
        fill=0, all_touched=False, dtype=np.uint8,
    )
    land_cell_count = int(np.count_nonzero(land_mask))
    if land_cell_count == 0:
        raise ValueError("陆域多边形没有覆盖当前规划网格")
    metadata = {
        "path": str(path.resolve()),
        "status": "loaded",
        "feature_count": len(land_shapes),
        "repaired_feature_count": len(repaired_feature_indices),
        "repaired_feature_indices": repaired_feature_indices,
        "land_cell_count": land_cell_count,
        "rasterization": "pixel_center_inside",
        "boundary_semantics": "administrative_land_polygon_screen_not_surveyed_shoreline",
    }
    if isinstance(document.get("metadata"), dict):
        metadata["dataset_metadata"] = document["metadata"]
    if isinstance(document.get("data_sources"), list):
        metadata["data_sources"] = document["data_sources"]
    return {"land_mask": land_mask, "metadata": metadata}


def load_user_constraint_overlays(geojson_path, transform, shape, *,
                                  require_verified_hard_constraints=False):
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
    hard_constraint_evidence = []
    source_shapes = {}
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
            verification_status = str(
                properties.get("verification_status", "")).strip().casefold()
            verification_reference = str(
                properties.get("verification_reference", "")).strip()
            hard_constraint_evidence.append({
                "feature_index": index,
                "source": source,
                "verification_status": verification_status or "unknown",
                "verification_reference": verification_reference or None,
            })
            if (require_verified_hard_constraints
                    and (verification_status != "verified" or not verification_reference)):
                raise ValueError(
                    f"硬约束要素 {index} 必须声明 verification_status=verified "
                    "并提供 verification_reference 才能选择塔位图方案")
            hard_shapes.append((geometry, 1))
        else:
            try:
                weight = float(properties["weight"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"软约束要素 {index} 必须提供非负数值 weight") from exc
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"软约束要素 {index} 的 weight 必须是非负有限数值")
            soft_shapes.append((geometry, weight))
        source_layer_shapes = source_shapes.setdefault(source, {"hard": [], "soft": []})
        source_layer_shapes[mode].append((geometry, 1 if mode == "hard" else weight))
        sources.add(source)
        category = str(properties.get("category", "")).strip()
        if category:
            categories.add(category)

    if require_verified_hard_constraints and not hard_shapes:
        raise ValueError("选择塔位图方案至少需要一个带核验凭证的硬约束要素")

    height, width = shape
    hard_mask = rio_rasterize(
        hard_shapes, out_shape=(height, width), transform=transform,
        fill=0, all_touched=True, dtype=np.uint8,
    ) if hard_shapes else np.zeros((height, width), dtype=np.uint8)
    if require_verified_hard_constraints and not np.any(hard_mask):
        raise ValueError("核验硬约束要素没有覆盖当前规划网格")
    soft_cost = rio_rasterize(
        soft_shapes, out_shape=(height, width), transform=transform,
        fill=0, all_touched=True, dtype=np.float32,
        merge_alg=rasterio.enums.MergeAlg.add,
    ) if soft_shapes else np.zeros((height, width), dtype=np.float32)
    source_layers = []
    for source, modes in sorted(source_shapes.items()):
        for mode, geometries in modes.items():
            if not geometries:
                continue
            coverage = rio_rasterize(
                [(geometry, 1) for geometry, _ in geometries],
                out_shape=(height, width), transform=transform,
                fill=0, all_touched=True, dtype=np.uint8,
            )
            covered_cell_count = int(np.count_nonzero(coverage))
            affected_cell_count = covered_cell_count
            if mode == "soft":
                del coverage
                positive_weight_shapes = [
                    (geometry, 1) for geometry, weight in geometries if weight > 0
                ]
                if positive_weight_shapes:
                    affected = rio_rasterize(
                        positive_weight_shapes, out_shape=(height, width), transform=transform,
                        fill=0, all_touched=True, dtype=np.uint8,
                    )
                    affected_cell_count = int(np.count_nonzero(affected))
                else:
                    affected_cell_count = 0
            source_layers.append({
                "source": source,
                "mode": mode,
                "feature_count": len(geometries),
                "covered_cell_count": covered_cell_count,
                "affected_cell_count": affected_cell_count,
            })
    metadata = {
        "path": str(path.resolve()),
        "status": "loaded",
        "feature_count": len(document["features"]),
        "hard_feature_count": len(hard_shapes),
        "soft_feature_count": len(soft_shapes),
        "hard_constraint_evidence": hard_constraint_evidence,
        "hard_constraint_declarations_complete": bool(hard_shapes) and all(
            item["verification_status"] == "verified"
            and item["verification_reference"]
            for item in hard_constraint_evidence),
        "hard_constraint_evidence_validation_scope": "caller_declarations_only",
        "sources": sorted(sources),
        "categories": sorted(categories),
        "hard_cell_count": int(hard_mask.sum()),
        "soft_cell_count": int(np.count_nonzero(soft_cost)),
        "source_layers": source_layers,
    }
    if dataset_metadata is not None:
        metadata["dataset_metadata"] = dataset_metadata
    if data_sources is not None:
        metadata["data_sources"] = data_sources
    return {"hard_mask": hard_mask, "soft_cost": soft_cost, "metadata": metadata}


def load_official_buildings_geojson(geojson_path, bbox_wgs84, transform, shape,
                                   *, buffer_m=0.0):
    """Load authorized WGS84 building footprints as an all-touched hard mask.

    Coverage is deliberately left unverified: an empty clipped result does not
    mean that the planning window contains no buildings.
    """
    if not HAS_RASTERIO or not HAS_SHAPELY:
        raise RuntimeError("读取官方建筑图层需要 rasterio 和 shapely")
    path = Path(geojson_path)
    if not path.is_file():
        raise FileNotFoundError(f"官方建筑 GeoJSON 不存在: {path}")
    if (len(bbox_wgs84) != 4
            or not all(math.isfinite(float(value)) for value in bbox_wgs84)):
        raise ValueError("bbox_wgs84 必须是四个有限坐标")
    west, south, east, north = map(float, bbox_wgs84)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox_wgs84 必须是有效 WGS84 范围")
    if not math.isfinite(float(buffer_m)) or buffer_m < 0:
        raise ValueError("buffer_m 必须是非负有限数值")
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError("shape 必须是正值 (height, width)")
    planning_bbox = (west, south, east, north)
    if buffer_m:
        lat_buffer = float(buffer_m) / 110574.0
        max_abs_lat = max(abs(south), abs(north))
        lon_buffer = float(buffer_m) / max(
            111320.0 * math.cos(math.radians(max_abs_lat)), 1e-6)
        west = max(-180.0, west - lon_buffer)
        east = min(180.0, east + lon_buffer)
        south = max(-90.0, south - lat_buffer)
        north = min(90.0, north + lat_buffer)

    dataset_metadata = _read_geojson_tail_value(path, "metadata")
    data_sources = _read_geojson_tail_value(path, "data_sources")
    if not isinstance(dataset_metadata, dict):
        raise ValueError("官方建筑 GeoJSON 的 metadata 必须是对象")
    if not isinstance(data_sources, list):
        raise ValueError("官方建筑 GeoJSON 的 data_sources 必须是数组")
    coordinate_system = str(dataset_metadata.get("coordinate_system", "")).upper()
    if "4326" not in coordinate_system and "CRS84" not in coordinate_system:
        raise ValueError("官方建筑 GeoJSON 的 metadata 必须声明 WGS84 经纬度坐标")
    if dataset_metadata.get("constraint_mode") != "hard":
        raise ValueError("官方建筑 GeoJSON 必须声明 constraint_mode=hard")

    try:
        import geopandas as gpd
        import pyogrio
    except ImportError as exc:
        raise RuntimeError("读取官方建筑 GeoJSON 需要 geopandas 和 pyogrio") from exc
    stat = path.stat()
    cache_key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    info = _OFFICIAL_BUILDING_INFO_CACHE.get(cache_key)
    if info is None:
        info = pyogrio.read_info(path)
        _OFFICIAL_BUILDING_INFO_CACHE[cache_key] = info
    if info.get("driver") != "GeoJSON":
        raise ValueError("官方建筑输入必须是 GeoJSON")
    try:
        metadata_feature_count = int(dataset_metadata["feature_count"])
        source_feature_count = int(info.get("features", -1))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("官方建筑 GeoJSON 的 metadata.feature_count 无效") from exc
    if metadata_feature_count < 1 or source_feature_count != metadata_feature_count:
        raise ValueError("官方建筑 GeoJSON 要素数与 metadata.feature_count 不一致")
    fields = {str(field) for field in info.get("fields", ())}
    required_fields = {"mode", "source", "category"}
    if not required_fields.issubset(fields):
        raise ValueError("官方建筑 GeoJSON 缺少 mode/source/category 字段")
    source_crs = str(info.get("crs") or "").upper()
    if "4326" not in source_crs and "CRS84" not in source_crs:
        raise ValueError("官方建筑 GeoJSON 必须使用 WGS84 经纬度坐标")
    frame = gpd.read_file(
        path, bbox=(west, south, east, north),
        columns=["mode", "source", "category", "dataset_id", "source_updated_at"],
        engine="pyogrio",
    )
    frame_crs = frame.crs.to_string().upper() if frame.crs is not None else ""
    if "4326" not in frame_crs and "CRS84" not in frame_crs:
        raise ValueError("官方建筑 GeoJSON 必须使用 WGS84 经纬度坐标")

    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

    query_window = box(west, south, east, north)
    planning_window = box(*planning_bbox)
    geometries = []
    intersecting_feature_count = 0
    planning_feature_count = 0
    route_geometries = []
    feature_sources = set()

    def polygon_parts(geometry):
        if isinstance(geometry, Polygon):
            return [geometry] if not geometry.is_empty else []
        if isinstance(geometry, MultiPolygon):
            return [part for part in geometry.geoms if not part.is_empty]
        if isinstance(geometry, GeometryCollection):
            return [part for item in geometry.geoms for part in polygon_parts(item)]
        return []

    for index, row in enumerate(frame.itertuples(), start=1):
        geometry = row.geometry
        source = getattr(row, "source", None)
        mode = getattr(row, "mode", None)
        category = getattr(row, "category", None)
        if mode != "hard" or category != "building_footprint":
            raise ValueError(f"官方建筑要素 {index} 必须声明 hard/building_footprint")
        source_name = str(source or "").strip()
        if source_name:
            feature_sources.add(source_name)
        properties = {
            "mode": mode,
            "source": source_name,
            "category": category,
            "dataset_id": getattr(row, "dataset_id", None),
            "source_updated_at": getattr(row, "source_updated_at", None),
        }
        if (geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}
                or not geometry.is_valid):
            raise ValueError(f"官方建筑要素 {index} 必须是有效 Polygon 或 MultiPolygon")
        min_lon, min_lat, max_lon, max_lat = geometry.bounds
        if (not all(math.isfinite(value) for value in geometry.bounds)
                or min_lon < -180 or max_lon > 180
                or min_lat < -90 or max_lat > 90):
            raise ValueError(f"官方建筑要素 {index} 坐标超出 WGS84 经纬度范围")
        if not geometry.intersects(query_window):
            continue
        intersecting_feature_count += 1
        clipped = geometry.intersection(query_window)
        parts = polygon_parts(clipped)
        geometries.extend((part, properties) for part in parts)
        if geometry.intersects(planning_window):
            planning_feature_count += 1
            route_clipped = geometry.intersection(planning_window)
            route_geometries.extend(
                (part, properties) for part in polygon_parts(route_clipped))

    mask = rasterize_on(geometries, transform, tuple(map(int, shape)))
    metadata = {
        "path": str(path.resolve()),
        "status": "loaded" if planning_feature_count else "loaded_no_features_in_window",
        "source": "NLSC Taiwan e-Map BUILD",
        "feature_sources": sorted(feature_sources),
        "source_feature_count": metadata_feature_count,
        "source_features_intersecting_window": planning_feature_count,
        "source_features_intersecting_query_bbox": intersecting_feature_count,
        "clipped_polygon_count": len(geometries),
        "covered_cell_count": int(np.count_nonzero(mask)),
        "rasterization": "all_touched",
        "requested_bbox_wgs84": [west, south, east, north],
        "planning_window_bbox_wgs84": list(planning_bbox),
        "load_buffer_m": float(buffer_m),
        "coverage_status": "not_independently_verified",
        "coverage_reason": (
            "Input file coverage is not independently certified; zero features in the window "
            "does not establish absence of buildings."),
        "check_scope": "loaded_official_building_features_only",
        "data_sources": data_sources,
        "dataset_metadata": dataset_metadata,
    }
    return {
        "geometries": geometries, "route_geometries": route_geometries,
        "mask": mask, "metadata": metadata,
    }


def merge_official_buildings_into_bands(osm_bands, official_layer, transform, shape,
                                        *, include_building_distance=False):
    """Union official footprints with OSM footprints for hard screening only."""
    official_mask = np.asarray(official_layer["mask"], dtype=np.uint8)
    if official_mask.shape != tuple(shape):
        raise ValueError("official building mask must match the DEM")
    existing_mask = np.asarray(osm_bands.get("building_mask"), dtype=np.uint8)
    if existing_mask.shape != tuple(shape):
        raise ValueError("OSM building mask must match the DEM")
    geometries = list(osm_bands.geometries.get("buildings", []))
    geometries.extend(official_layer["geometries"])
    osm_bands.geometries["buildings"] = geometries
    osm_bands["building_mask"] = np.maximum(existing_mask, official_mask)
    if include_building_distance and official_layer["geometries"]:
        osm_bands["building_distance_m"] = compute_distance_band(
            geometries, transform, shape)
    return geometries


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


def compute_osm_bands(dem, transform, bbox, cache_dir=None, *,
                      include_model_features=True, include_building_distance=True):
    """计算 OSM 波段与风险代理；图层状态可从返回值的 ``metadata`` 属性读取。

    Building footprints provide a hard footprint mask and settlement proxy;
    residential land use provides a soft settlement proxy and model feature.
    Other layers are model features only and are not used by the default rule
    cost. Protected areas are reported for inspection only.
    """
    H, W = dem.shape
    cache_dir = Path(cache_dir) if cache_dir else DOWNLOADED_DIR
    layer_specs = {
        "roads": ("taiwan_roads.pkl", "line", "feature_model_input_only_not_default_rule_cost"),
        "water": ("taiwan_water.pkl", "line", "feature_model_input_only_not_default_rule_cost"),
        "railways": ("taiwan_railways.pkl", "line", "feature_model_input_only_not_default_rule_cost"),
        "faults": ("taiwan_faults.pkl", "line", "feature_model_input_only_not_default_rule_cost"),
        "landuse": ("taiwan_landuse.pkl", "polygon", "settlement_and_optional_landcover_soft_proxy"),
        "buildings": ("taiwan_buildings.pkl", "polygon", "hard_footprint_and_settlement_proxy"),
        "vegetation": ("taiwan_vegetation.pkl", "polygon", "feature_model_input_only_not_default_rule_cost"),
        "protected_areas": ("taiwan_protected.pkl", "polygon", "informational_only"),
    }
    layers = {}
    layer_metadata = {}
    for name, (filename, geom_type, role) in layer_specs.items():
        if not include_model_features and name not in ("landuse", "buildings"):
            layers[name] = []
            layer_metadata[name] = {
                "path": str(cache_dir / filename), "status": "not_loaded",
                "reason": "not_required_by_selected_planner_mode",
                "coverage_status": "not_checked", "planning_role": role,
            }
            continue
        query_bbox = bbox
        if name == "buildings":
            latitude = 0.5 * (bbox[1] + bbox[3])
            buffer_m = 5000.0 if include_building_distance else 2000.0
            lat_buffer = buffer_m / 111320.0
            lon_buffer = lat_buffer / max(math.cos(math.radians(latitude)), 1e-6)
            query_bbox = (bbox[0] - lon_buffer, bbox[1] - lat_buffer,
                          bbox[2] + lon_buffer, bbox[3] + lat_buffer)
        layers[name], layer_metadata[name] = load_osm_geoms_with_status(
            cache_dir / filename, geom_type, query_bbox)
        layer_metadata[name]["planning_role"] = role
    roads = layers["roads"]
    water = layers["water"]
    railways = layers["railways"]
    faults = layers["faults"]
    landuse = layers["landuse"]
    buildings = layers["buildings"]
    vegetation = layers["vegetation"]
    if include_model_features:
        print(f"  OSM 几何数: 道路{len(roads)} 水域{len(water)} 铁路{len(railways)} "
              f"断裂{len(faults)} 土地利用{len(landuse)} 建筑{len(buildings)} 植被{len(vegetation)}")
    else:
        print(f"  OSM 规划图层: 土地利用{len(landuse)} 建筑{len(buildings)}；"
              "未加载仅供模型或诊断的可选图层")

    bands = OSMBands(metadata={"layers": layer_metadata},
                     geometries={"buildings": buildings})
    if include_model_features:
        bands["dist_road"] = compute_distance_band(roads, transform, (H, W), 5000.0)
        bands["dist_water"] = compute_distance_band(water, transform, (H, W), 5000.0)
        bands["dist_railway"] = compute_distance_band(railways, transform, (H, W), 5000.0)
        bands["dist_fault"] = compute_distance_band(faults, transform, (H, W), 10000.0)
    bands["landuse_code"] = compute_landuse_code(landuse, transform, (H, W))
    bands["residential_landuse_mask"] = compute_residential_landuse_mask(landuse, transform, (H, W))
    bands["building_density"] = compute_building_density(buildings, transform, (H, W))
    bands["building_mask"] = rasterize_on(buildings, transform, (H, W))
    if include_building_distance:
        bands["building_distance_m"] = compute_distance_band(buildings, transform, (H, W), 5000.0)
    if include_model_features:
        bands["vegetation_height"] = compute_vegetation_height(vegetation, transform, (H, W))

    if include_model_features:
        # 风险代理波段（21-25）— 由地形/位置推导
        z = _fill(dem).astype(np.float32)
        lat_center = transform.f + transform.e * (H / 2.0)  # 窗口纬度中心（修正：之前误用 0）
        row_m, col_m = pixel_spacing_meters(transform, lat_center)
        dy, dx = _terrain_gradient(z, row_m, col_m)
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
