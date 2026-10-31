"""Read official ARDSWC MVT and sanitized SHP data for WGS84 planning windows.

The returned masks are screening inputs only. In particular, mapped potential
landslide and debris-flow impact areas are never interpreted as legal exclusions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


LAYER_SPECS = {
    "debris_flow_impact": "DEBRIS1753_20260126",
    "landslide_potential": "POTENTIAL_LANDSLIDE_20260106",
    "landslide_impact": "POTENTIAL_LANDSLIDE_AFFECT_20260106",
}

VECTOR_LAYER_SPECS = {
    "debris_flow_impact": ("debris_flow_impact_115_2026_sanitized.zip", "176526"),
    "landslide_potential": ("landslide_potential_115_2026_sanitized.zip", "176527"),
    "landslide_impact": ("landslide_impact_115_2026_sanitized.zip", "176528"),
}


class _DriverUnavailable(RuntimeError):
    pass


class _DependencyUnavailable(RuntimeError):
    pass


def _zero_masks(shape):
    return {name: np.zeros(shape, dtype=np.uint8) for name in LAYER_SPECS}


def _tile_xy(lon, lat, zoom):
    n = 1 << zoom
    x = math.floor((lon + 180.0) / 360.0 * n)
    lat = max(min(lat, 85.0511287798066), -85.0511287798066)
    lat_rad = math.radians(lat)
    y = math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) * 0.5 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def _tiles_for_bbox(bbox, zoom):
    west, south, east, north = bbox
    if not all(math.isfinite(float(value)) for value in bbox):
        raise ValueError("bbox_wgs84 坐标必须是有限数值")
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox_wgs84 必须是 west,south,east,north 顺序的有效范围")

    # Work in fractional tile coordinates before applying nextafter: at a
    # geographic tile edge, the ULP in degrees can vanish in the Mercator math.
    n = 1 << zoom
    x_west = (west + 180.0) / 360.0 * n
    x_east = (east + 180.0) / 360.0 * n

    def y_fraction(lat):
        lat = max(min(lat, 85.0511287798066), -85.0511287798066)
        lat_rad = math.radians(lat)
        return (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) * 0.5 * n

    y_north = y_fraction(north)
    y_south = y_fraction(south)
    x0 = math.floor(math.nextafter(x_west, math.inf))
    x1 = math.floor(math.nextafter(x_east, -math.inf))
    y0 = math.floor(math.nextafter(y_north, math.inf))
    y1 = math.floor(math.nextafter(y_south, -math.inf))
    x0, x1 = min(max(x0, 0), n - 1), min(max(x1, 0), n - 1)
    y0, y1 = min(max(y0, 0), n - 1), min(max(y1, 0), n - 1)
    return [(x, y) for y in range(min(y0, y1), max(y0, y1) + 1)
            for x in range(min(x0, x1), max(x0, x1) + 1)]


def _bbox_is_covered(requested, source):
    west, south, east, north = requested
    src_west, src_south, src_east, src_north = source
    return (src_west <= west and src_south <= south
            and src_east >= east and src_north >= north)


def _failed_tile_info(raw_failures, layer_dir, zoom):
    """Return identifiable failed (x,y) tiles and count of unlocated failures."""
    failed = set()
    unlocated = 0
    for item in raw_failures or []:
        values = item if isinstance(item, dict) else {}
        if values:
            try:
                if int(values.get("z", zoom)) == zoom:
                    failed.add((int(values["x"]), int(values["y"])))
                    continue
            except (KeyError, TypeError, ValueError):
                pass
        match = re.search(r"(?:^|[/\\])\d+[/\\](\d+)[/\\](\d+)\.pbf(?:$|[?#])", str(item))
        if match:
            y, x = map(int, match.groups())
            failed.add((x, y))
        else:
            unlocated += 1
    return failed, unlocated


def _load_checksums(path):
    if not path.is_file():
        return None
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$", line)
        if match:
            checksums[match.group(2).replace("\\", "/")] = match.group(1).lower()
    return checksums


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_tile_geometries(tile_path, bbox_wgs84, tile_x, tile_y, zoom):
    """Read polygon fragments through GDAL's MVT driver and return WGS84 shapes."""
    try:
        import pyogrio
    except ImportError as exc:
        raise _DriverUnavailable("pyogrio is not installed") from exc

    # The downloaded BigGIS package stores paths as z/y/x, while GDAL's MVT
    # driver assumes z/x/y from a path unless explicit tile indices are given.
    tile_options = {"X": int(tile_x), "Y": int(tile_y), "Z": int(zoom)}
    info = pyogrio.read_info(tile_path, layer=0, **tile_options)
    if info.get("driver") != "MVT":
        raise _DriverUnavailable(f"GDAL opened tile with {info.get('driver')!r}, not MVT")
    source_crs = info.get("crs")
    if not source_crs:
        raise ValueError("MVT tile does not declare a CRS")

    try:
        from pyproj import CRS, Transformer
        from shapely.geometry import box
        from shapely.ops import transform as transform_geometry
        from shapely import from_wkb
    except ImportError as exc:
        raise _DependencyUnavailable("pyproj and shapely are required to reproject MVT polygons") from exc

    source = CRS.from_user_input(source_crs)
    target = CRS.from_epsg(4326)
    forward = Transformer.from_crs(target, source, always_xy=True)
    backward = Transformer.from_crs(source, target, always_xy=True)
    source_bounds = forward.transform_bounds(*bbox_wgs84, densify_pts=21)
    layer_name = info.get("layer_name") or 0
    _, _, geometries, _ = pyogrio.raw.read(
        tile_path, layer=layer_name, columns=[], bbox=source_bounds, **tile_options,
    )
    if geometries is None:
        return []

    window = box(*bbox_wgs84)
    result = []
    for value in geometries:
        if value is None:
            continue
        geometry = from_wkb(bytes(value))
        if geometry.is_empty:
            continue
        if geometry.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        if not source.equals(target):
            geometry = transform_geometry(backward.transform, geometry)
        if geometry.is_empty or not geometry.intersects(window):
            continue
        clipped = geometry.intersection(window)
        if clipped.geom_type in ("Polygon", "MultiPolygon") and not clipped.is_empty:
            result.append(clipped)
        elif clipped.geom_type == "GeometryCollection":
            result.extend(g for g in clipped.geoms
                          if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_empty)
    return result


def _rasterize_union(geometries, transform, shape):
    if not geometries:
        return np.zeros(shape, dtype=np.uint8)
    try:
        from rasterio.features import rasterize
    except ImportError as exc:
        raise _DependencyUnavailable("rasterio is required to rasterize official hazard tiles") from exc
    return rasterize(
        ((geometry, 1) for geometry in geometries),
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )


def load_official_hazard_vector_masks(source_dir, bbox_wgs84, transform, shape):
    """Load validated, sanitized nationwide 2026 ARDSWC SHP risk layers.

    The vector files are projected from TWD97 / TM2 zone 121 to WGS84,
    clipped to the route window, then rasterized on the planner's DEM grid.
    Their masks remain soft screening signals; they are not legal exclusions.
    """
    bbox = tuple(float(value) for value in bbox_wgs84)
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError("bbox_wgs84 必须包含四个有限坐标")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError("bbox_wgs84 必须是有效 west,south,east,north 范围")
    if len(shape) != 2 or any(int(value) <= 0 for value in shape):
        raise ValueError("shape 必须是正值 (height, width)")

    root = Path(source_dir)
    if root.is_symlink():
        raise ValueError("官方灾害矢量目录不能是符号链接")
    if not root.is_dir():
        raise FileNotFoundError(f"官方灾害矢量目录不存在: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("官方灾害矢量清单不能是符号链接")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"官方灾害矢量清单不存在: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = {item["name"]: item for item in manifest["sanitized_files"]}
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("官方灾害矢量清单格式无效") from exc
    if manifest.get("address_values_read_or_emitted") is not False:
        raise ValueError("官方灾害矢量清单未声明地址值未读取或导出")

    try:
        import geopandas as gpd
        import pyogrio
        from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
        from shapely import force_2d, make_valid
    except ImportError as exc:
        raise RuntimeError("读取官方灾害矢量需要 geopandas、pyogrio 和 shapely") from exc

    masks = _zero_masks(tuple(map(int, shape)))
    clip = box(*bbox)
    metadata = {
        "status": "available",
        "coverage_status": "unknown",
        "source_integrity_status": "validated_against_local_manifest",
        "spatial_coverage_status": "not_independently_verified",
        "source_dir": str(root.resolve()),
        "source": "Taiwan MOA ARDSWC official 2026 sanitized SHP",
        "source_type": "sanitized_shapefile",
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": _sha256(manifest_path),
        "requested_bbox_wgs84": list(bbox),
        "raster_crs": "EPSG:4326",
        "risk_mode": "soft_only_screening; no hard exclusion",
        "risk_semantics": "binary_union; source Risk attributes are not interpreted",
        "address_values_read_or_emitted": False,
        "warning": "All source archives are checked against the local manifest; spatial inventory completeness is unverified. Polygons are screening data, not surveyed exclusions.",
        "layers": {},
    }

    for layer_name, (filename, dataset_id) in VECTOR_LAYER_SPECS.items():
        archive = root / filename
        entry = files.get(filename)
        if archive.is_symlink():
            raise ValueError(f"官方灾害矢量文件不能是符号链接: {filename}")
        if entry is None or not archive.is_file():
            raise FileNotFoundError(f"官方灾害矢量文件缺失: {filename}")
        actual_hash = _sha256(archive)
        if actual_hash != str(entry.get("sha256", "")).lower():
            raise ValueError(f"官方灾害矢量 SHA-256 不匹配: {filename}")
        if entry.get("address_field_removed") is not True:
            raise ValueError(f"官方灾害矢量未声明移除地址字段: {filename}")

        vector_path = f"/vsizip/{archive.resolve().as_posix()}"
        layer_info = pyogrio.read_info(vector_path)
        fields = {str(column).strip().casefold()
                  for column in layer_info.get("fields", ())}
        if "address" in fields or "地址" in fields:
            raise ValueError(f"脱敏灾害图层仍含地址字段: {filename}")
        try:
            expected_count = int(entry.get("feature_count", -1))
            observed_count = int(layer_info.get("features", -1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"官方灾害矢量要素数清单无效: {filename}") from exc
        if expected_count < 0 or observed_count != expected_count:
            raise ValueError(f"官方灾害矢量要素数与清单不一致: {filename}")
        geometry_type = str(layer_info.get("geometry_type", "")).upper()
        if not geometry_type.startswith(("POLYGON", "MULTIPOLYGON")):
            raise ValueError(f"官方灾害矢量必须仅包含面几何: {filename}")
        frame = gpd.read_file(vector_path, columns=[], engine="pyogrio")
        geometry_column = getattr(frame.geometry, "name", "geometry")
        if any(column != geometry_column for column in frame.columns):
            raise RuntimeError("pyogrio 几何读取意外加载了属性字段")
        if frame.crs is None or frame.crs.to_epsg() != 3826:
            raise ValueError(f"官方灾害矢量必须使用 EPSG:3826: {filename}")
        if len(frame) != expected_count:
            raise ValueError(f"官方灾害矢量要素数与清单不一致: {filename}")
        if not bool(frame.geometry.is_valid.all()):
            raise ValueError(f"官方灾害矢量含无效几何: {filename}")

        frame = frame.to_crs("EPSG:4326")
        geometries = []
        intersecting = 0
        for geometry in frame.geometry:
            if geometry is None or geometry.is_empty:
                raise ValueError(f"官方灾害矢量含空几何: {filename}")
            if geometry.geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError(f"官方灾害矢量含非面要素: {filename}")
            if not geometry.intersects(clip):
                continue
            intersecting += 1
            geometry = force_2d(geometry)
            clipped = make_valid(geometry.intersection(clip))
            if isinstance(clipped, Polygon):
                geometries.append(clipped)
            elif isinstance(clipped, MultiPolygon):
                geometries.extend(clipped.geoms)
            elif isinstance(clipped, GeometryCollection):
                parts = [part for part in clipped.geoms
                         if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty]
                for part in parts:
                    geometries.extend([part] if isinstance(part, Polygon) else part.geoms)

        masks[layer_name] = _rasterize_union(geometries, transform, tuple(map(int, shape)))
        metadata["layers"][layer_name] = {
            "source_layer": layer_name,
            "source_dataset_id": dataset_id,
            "source_archive": filename,
            "source_archive_sha256": actual_hash,
            "source_crs": "EPSG:3826",
            "planning_role": "soft_risk_only",
            "hard_exclusion_allowed": False,
            "source_integrity_status": "validated_against_local_manifest",
            "spatial_coverage_status": "not_independently_verified",
            "coverage_status": "unknown",
            "status": "loaded" if geometries else "official_empty_for_window",
            "source_feature_count": len(frame),
            "feature_fragment_count": len(geometries),
            "source_features_intersecting_window": intersecting,
            "covered_cell_count": int(np.count_nonzero(masks[layer_name])),
        }

    metadata["covered_cell_count"] = int(np.count_nonzero(
        np.maximum.reduce(list(masks.values()))))
    return masks, metadata


def load_official_hazard_masks(source_dir, bbox_wgs84, transform, shape):
    """Load three official hazard masks for a WGS84 DEM window.

    Parameters
    ----------
    source_dir:
        Directory containing the downloaded ARDSWC MVT package and manifest.
    bbox_wgs84:
        ``(west, south, east, north)`` in longitude/latitude degrees.
    transform, shape:
        Affine transform and ``(height, width)`` of the DEM window. The grid is
        assumed to use EPSG:4326, matching the planner's current DEM contract.

    Returns
    -------
    (masks, metadata):
        uint8 0/1 masks keyed by ``debris_flow_impact``,
        ``landslide_potential``, and ``landslide_impact``. Duplicate MVT tile
        fragments are combined by raster union. Potential and impact masks are
        screening inputs only and never imply a hard exclusion.
    """
    bbox = tuple(float(v) for v in bbox_wgs84)
    if len(bbox) != 4:
        raise ValueError("bbox_wgs84 必须包含 west,south,east,north 四个坐标")
    if len(shape) != 2 or any(int(v) <= 0 for v in shape):
        raise ValueError("shape 必须是正值 (height, width)")
    masks = _zero_masks(tuple(map(int, shape)))
    root = Path(source_dir) if source_dir is not None else None
    metadata = {
        "status": "missing_source",
        "coverage_status": "unknown",
        "source_dir": str(root.resolve()) if root is not None and root.exists() else str(root) if root else None,
        "source": None,
        "downloaded_at_utc": None,
        "source_bbox_wgs84": None,
        "requested_bbox_wgs84": list(bbox),
        "zoom": None,
        "raster_crs": "EPSG:4326",
        "risk_mode": "soft_only_screening; no hard exclusion",
        "warning": "MVT tiles are scale-generalized and clipped at tile boundaries; not original survey polygons.",
        "tiles": {"requested": 0, "read": 0, "official_empty": 0,
                  "failed": 0, "unknown": 0, "read_error": 0,
                  "checksum_mismatch": 0},
        "layers": {},
    }
    for key, layer_dir in LAYER_SPECS.items():
        metadata["layers"][key] = {
            "source_layer": layer_dir,
            "planning_role": "soft_risk_only",
            "default_soft_weight": 0.0,
            "hard_exclusion_allowed": False,
            "status": "not_loaded",
            "coverage_status": "unknown",
            "feature_fragment_count": 0,
            "non_polygon_feature_count": 0,
            "covered_cell_count": 0,
            "tile_records": [],
            "tiles": {"requested": 0, "read": 0, "official_empty": 0,
                      "failed": 0, "unknown": 0, "read_error": 0,
                      "checksum_mismatch": 0},
        }

    if root is None or not root.is_dir():
        metadata["warning"] = "Official hazard source directory is missing; all masks are zero and coverage is unknown."
        return masks, metadata

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        metadata.update(status="no_manifest", coverage_status="unknown")
        metadata["warning"] = "Manifest is missing; absent tiles cannot be distinguished from official empty tiles."
        return masks, metadata
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        zoom = int(manifest["zoom"])
        source_bbox = (float(manifest["bbox_wgs84"]["west"]),
                       float(manifest["bbox_wgs84"]["south"]),
                       float(manifest["bbox_wgs84"]["east"]),
                       float(manifest["bbox_wgs84"]["north"]))
        tile_x_range = tuple(map(int, manifest["tile_x_range_inclusive"]))
        tile_y_range = tuple(map(int, manifest["tile_y_range_inclusive"]))
        layer_manifest = manifest["layers"]
        if zoom < 0 or len(source_bbox) != 4 or len(tile_x_range) != 2 or len(tile_y_range) != 2:
            raise ValueError("invalid manifest fields")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        metadata.update(status="invalid_manifest", coverage_status="unknown")
        metadata["warning"] = f"Official hazard manifest is invalid: {exc}"
        return masks, metadata

    metadata.update(
        status="available",
        source=manifest.get("source"),
        downloaded_at_utc=manifest.get("downloaded_at_utc"),
        source_bbox_wgs84=list(source_bbox),
        zoom=zoom,
        manifest_path=str(manifest_path.resolve()),
    )
    requested_tiles = _tiles_for_bbox(bbox, zoom)
    metadata["tiles"]["requested"] = len(requested_tiles) * len(LAYER_SPECS)
    source_bbox_covers = _bbox_is_covered(bbox, source_bbox)
    checksums = _load_checksums(root / "tiles.sha256")
    all_layers_complete = True
    saw_data = False
    saw_read_error = False
    saw_driver_unavailable = False
    saw_dependency_unavailable = False

    for key, layer_dir in LAYER_SPECS.items():
        layer_meta = layer_manifest.get(layer_dir)
        layer_out = metadata["layers"][key]
        if not isinstance(layer_meta, dict):
            layer_out.update(status="missing_manifest_layer", coverage_status="unknown")
            layer_out["tiles"]["unknown"] = len(requested_tiles)
            metadata["tiles"]["unknown"] += len(requested_tiles)
            all_layers_complete = False
            continue

        layer_out.update(
            description=layer_meta.get("description"),
            source_date=(re.search(r"_(\d{8})$", layer_dir).group(1)
                         if re.search(r"_(\d{8})$", layer_dir) else None),
        )
        counts_valid = False
        checksum_count_valid = None
        checksummed_nonempty_count = None
        try:
            requested_count = int(layer_meta["tiles_requested"])
            nonempty_count = int(layer_meta["nonempty_tiles"])
            empty_count = int(layer_meta["empty_tiles"])
            raw_failures = layer_meta.get("failures", layer_meta.get("failed_tiles", [])) or []
            failed_declared = int(layer_meta.get("failed_tile_count", len(raw_failures)))
            failed_coords, unlocated_failures = _failed_tile_info(raw_failures, layer_dir, zoom)
            expected_rectangle_count = ((tile_x_range[1] - tile_x_range[0] + 1)
                                        * (tile_y_range[1] - tile_y_range[0] + 1))
            counts_valid = (requested_count == expected_rectangle_count
                            and nonempty_count + empty_count + failed_declared == requested_count)
            if checksums is not None:
                layer_checksum_prefix = f"{layer_dir}/"
                layer_checksum_paths = [
                    path for path in checksums if path.startswith(layer_checksum_prefix)
                ]
                checksummed_coords = set()
                checksum_paths_valid = True
                for path in layer_checksum_paths:
                    match = re.fullmatch(
                        rf"{re.escape(layer_dir)}/{zoom}/(\d+)/(\d+)\.pbf", path)
                    if not match:
                        checksum_paths_valid = False
                        continue
                    tile_y, tile_x = map(int, match.groups())
                    if not (tile_x_range[0] <= tile_x <= tile_x_range[1]
                            and tile_y_range[0] <= tile_y <= tile_y_range[1]):
                        checksum_paths_valid = False
                        continue
                    checksummed_coords.add((tile_x, tile_y))
                checksummed_nonempty_count = len(checksummed_coords)
                checksum_count_valid = (
                    checksum_paths_valid
                    and len(layer_checksum_paths) == checksummed_nonempty_count
                    and checksummed_nonempty_count == nonempty_count
                )
        except (KeyError, TypeError, ValueError):
            requested_count = nonempty_count = empty_count = failed_declared = 0
            failed_coords, unlocated_failures = set(), 0
            checksum_count_valid = False if checksums is not None else None
            checksummed_nonempty_count = 0 if checksums is not None else None

        layer_complete = (source_bbox_covers and counts_valid
                          and checksum_count_valid is not False)
        layer_geometries = []
        tile_counts = layer_out["tiles"]
        tile_counts["requested"] = len(requested_tiles)
        metadata["layers"][key]["manifest_counts_valid"] = counts_valid
        metadata["layers"][key]["tiles"]["manifest_nonempty"] = nonempty_count
        metadata["layers"][key]["tiles"]["manifest_official_empty"] = empty_count
        metadata["layers"][key]["tiles"]["manifest_failed"] = failed_declared
        metadata["layers"][key]["tiles"]["checksummed_nonempty"] = checksummed_nonempty_count
        metadata["layers"][key]["manifest_checksum_count_match"] = checksum_count_valid
        layer_root = root / layer_dir

        def record_tile(x, y, status, **details):
            layer_out["tile_records"].append({
                "z": zoom, "x": x, "y": y,
                "path": f"{layer_dir}/{zoom}/{y}/{x}.pbf",
                "status": status,
                **details,
            })

        for x, y in requested_tiles:
            tile_key = (x, y)
            relpath = f"{layer_dir}/{zoom}/{y}/{x}.pbf"
            tile_path = layer_root / str(zoom) / str(y) / f"{x}.pbf"
            if (x < tile_x_range[0] or x > tile_x_range[1]
                    or y < tile_y_range[0] or y > tile_y_range[1]):
                tile_counts["unknown"] += 1
                record_tile(x, y, "unknown_outside_declared_coverage")
                layer_complete = False
                continue
            if tile_key in failed_coords:
                tile_counts["failed"] += 1
                record_tile(x, y, "download_failed")
                layer_complete = False
                continue
            if not tile_path.is_file():
                if (source_bbox_covers and counts_valid and checksum_count_valid is True
                        and relpath not in checksums
                        and unlocated_failures == 0):
                    tile_counts["official_empty"] += 1
                    record_tile(x, y, "official_empty")
                else:
                    tile_counts["unknown"] += 1
                    record_tile(x, y, "missing_unknown")
                    layer_complete = False
                continue

            checksum_status = "not_available"
            if checksums is not None:
                expected_hash = checksums.get(relpath)
                if expected_hash and _sha256(tile_path) != expected_hash:
                    tile_counts["checksum_mismatch"] += 1
                    record_tile(x, y, "checksum_mismatch")
                    layer_complete = False
                    continue
                if expected_hash is None:
                    checksum_status = "not_listed"
                    layer_complete = False
                else:
                    checksum_status = "verified"
            try:
                tile_geometries = _read_tile_geometries(tile_path, bbox, x, y, zoom)
            except _DriverUnavailable as exc:
                saw_driver_unavailable = True
                layer_out["warning"] = str(exc)
                tile_counts["read_error"] += 1
                record_tile(x, y, "driver_unavailable", checksum_status=checksum_status,
                            error=str(exc))
                layer_complete = False
                continue
            except _DependencyUnavailable as exc:
                saw_dependency_unavailable = True
                layer_out["warning"] = str(exc)
                tile_counts["read_error"] += 1
                record_tile(x, y, "dependency_unavailable", checksum_status=checksum_status,
                            error=str(exc))
                layer_complete = False
                continue
            except Exception as exc:
                saw_read_error = True
                layer_out.setdefault("warnings", []).append(
                    f"{relpath}: {type(exc).__name__}: {exc}")
                tile_counts["read_error"] += 1
                record_tile(x, y, "read_error", checksum_status=checksum_status,
                            error=f"{type(exc).__name__}: {exc}")
                layer_complete = False
                continue
            tile_counts["read"] += 1
            metadata["tiles"]["read"] += 1
            layer_geometries.extend(tile_geometries)
            record_tile(x, y, "read", checksum_status=checksum_status,
                        feature_fragment_count=len(tile_geometries))

        layer_out["feature_fragment_count"] = len(layer_geometries)
        if layer_geometries:
            saw_data = True
            try:
                layer_mask = _rasterize_union(layer_geometries, transform, tuple(map(int, shape)))
            except _DriverUnavailable as exc:
                saw_driver_unavailable = True
                layer_out["warning"] = str(exc)
                layer_mask = np.zeros(shape, dtype=np.uint8)
                layer_complete = False
            except _DependencyUnavailable as exc:
                saw_dependency_unavailable = True
                layer_out["warning"] = str(exc)
                layer_mask = np.zeros(shape, dtype=np.uint8)
                layer_complete = False
            except Exception as exc:
                saw_read_error = True
                layer_out["warning"] = f"rasterization failed: {type(exc).__name__}: {exc}"
                layer_mask = np.zeros(shape, dtype=np.uint8)
                layer_complete = False
            masks[key] = (np.asarray(layer_mask) > 0).astype(np.uint8)
            layer_out["covered_cell_count"] = int(np.count_nonzero(masks[key]))

        for count_name in ("official_empty", "failed", "unknown", "read_error", "checksum_mismatch"):
            metadata["tiles"][count_name] += tile_counts[count_name]
        layer_out["coverage_status"] = "complete" if layer_complete else "partial"
        if layer_complete:
            if layer_geometries:
                layer_out["status"] = "available"
            elif tile_counts["official_empty"] == len(requested_tiles):
                layer_out["status"] = "official_empty"
            else:
                layer_out["status"] = "empty_in_window"
        else:
            layer_out["status"] = "partial"
            all_layers_complete = False

    metadata["coverage_status"] = "complete" if all_layers_complete else "partial"
    if saw_driver_unavailable:
        metadata["status"] = "driver_unavailable"
    elif saw_dependency_unavailable:
        metadata["status"] = "dependency_unavailable"
    elif saw_read_error:
        metadata["status"] = "partial"
    elif saw_data:
        metadata["status"] = "available" if all_layers_complete else "partial"
    elif all_layers_complete:
        metadata["status"] = "official_empty"
    else:
        metadata["status"] = "partial"
    return masks, metadata
