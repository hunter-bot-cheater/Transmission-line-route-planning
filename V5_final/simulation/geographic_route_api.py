"""Server-side bridge from the simulator UI to the V5.0 geographic planner."""
from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import os
import re
import subprocess
import threading
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

_ALLOWED_ROOT = Path("/data1/B24041626/pp")
_TAIWAN_LAND_MASK_SHA256 = "7e3d24cb13b7f1125c1c967d9d436ee556567516b0eb9894cfa3e763fe8874dd"
_JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_ARTIFACTS = {"route": "geojson", "map": "png", "profile": "csv", "provenance": "json", "log": "txt"}


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _declared_inside(path: Path, root: Path) -> bool:
    """Check the configured path itself; venv/python is commonly a symlink."""
    try:
        Path(os.path.abspath(path)).relative_to(root.absolute())
        return True
    except ValueError:
        return False


def _number(value: Any, name: str, lower: float, upper: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name}必须是数值") from error
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name}超出允许范围")
    return result


def _dem_bounds_wgs84(dem_path: Path) -> dict[str, float]:
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(dem_path) as dataset:
        if dataset.crs is None:
            raise RuntimeError("服务器 DEM 未声明坐标系，无法限定台湾数据范围")
        west, south, east, north = transform_bounds(
            dataset.crs, "EPSG:4326", *dataset.bounds, densify_pts=21,
        )
    bounds = {"west": west, "south": south, "east": east, "north": north}
    if (not all(math.isfinite(value) for value in bounds.values())
            or not -180 <= west < east <= 180
            or not -90 <= south < north <= 90):
        raise RuntimeError("服务器 DEM 范围无效")
    return bounds


def _require_supported_planner_dem_crs(dem_path: Path) -> None:
    import rasterio

    with rasterio.open(dem_path) as dataset:
        supported_crs = rasterio.crs.CRS.from_epsg(4326)
        if dataset.crs != supported_crs:
            current_crs = str(dataset.crs) if dataset.crs is not None else "未声明"
            raise RuntimeError(
                "当前真实线路 CLI 仅支持 EPSG:4326 DEM；"
                f"当前 DEM 坐标系为 {current_crs}，请先重投影"
            )


def _engineering_gate_metadata() -> dict[str, Any]:
    return {
        "status": "not_evaluated",
        "scope": "terrain_and_configured_constraint_screening_only",
        "blockers": [
            {
                "code": "station_terminals_unverified",
                "message": "缺少权威线路侧站端接入点名录；当前 Web 任务只能使用探索性坐标。",
            },
            {
                "code": "tower_and_span_graph_not_run",
                "message": "Web 任务未运行塔位与档距图搜索。",
            },
            {
                "code": "approved_line_design_profile_missing",
                "message": "电压等级仅作记录；没有经批准的杆塔、导线和档距设计配置。",
            },
            {
                "code": "sag_and_clearance_not_evaluated",
                "message": "未提供导线荷载、弧垂及净空计算输入，未完成逐跨校核。",
            },
            {
                "code": "crossing_and_permit_review_not_evaluated",
                "message": "交叉跨越审批、用地许可和现场地质条件未评估。",
            },
        ],
    }


def _dem_has_valid_taiwan_land_cell(dem_path: Path, land_geometry) -> bool:
    import numpy as np
    import rasterio
    from rasterio.errors import RasterioError, WindowError
    from rasterio.features import geometry_mask, geometry_window
    from rasterio.warp import transform_geom
    from rasterio.windows import Window
    from shapely.geometry import mapping

    with rasterio.open(dem_path) as dataset:
        if dataset.crs is None:
            raise RuntimeError("服务器 DEM 未声明坐标系，无法检查台湾陆域有效像元")
        try:
            geometry = transform_geom(
                "EPSG:4326", dataset.crs, mapping(land_geometry), precision=-1,
            )
        except RasterioError as error:
            raise RuntimeError("无法将台湾陆域边界投影到 DEM 坐标系") from error
        try:
            window = geometry_window(dataset, [geometry])
        except WindowError:
            return False
        width = int(window.width)
        first_row = int(window.row_off)
        last_row = first_row + int(window.height)
        for row in range(first_row, last_row, 256):
            height = min(256, last_row - row)
            chunk = Window(window.col_off, row, window.width, height)
            elevations = dataset.read(1, window=chunk)
            valid = (dataset.read_masks(1, window=chunk) > 0) & np.isfinite(elevations)
            land = geometry_mask(
                [geometry], out_shape=(height, width),
                transform=dataset.window_transform(chunk), invert=True,
            )
            if np.any(valid & land):
                return True
    return False


@lru_cache(maxsize=4)
def _load_taiwan_land_geometry_cached(
        path_text: str, content_sha256: str):
    import rasterio
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import unary_union

    try:
        content = Path(path_text).read_bytes()
        if (hashlib.sha256(content).hexdigest() != content_sha256
                or content_sha256 != _TAIWAN_LAND_MASK_SHA256):
            raise RuntimeError("台湾陆域边界文件 SHA-256 与已核验的内政部县市界版本不一致")
        document = json.loads(content.decode("utf-8-sig"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("台湾陆域边界 GeoJSON 无法读取") from error
    features = document.get("features") if isinstance(document, dict) else None
    if (not isinstance(document, dict) or document.get("type") != "FeatureCollection"
            or not isinstance(features, list) or not features):
        raise RuntimeError("台湾陆域边界必须是非空 WGS84 GeoJSON FeatureCollection")

    if "crs" in document:
        declared_crs = document["crs"]
        properties = declared_crs.get("properties") if isinstance(declared_crs, dict) else None
        crs_name = properties.get("name") if isinstance(properties, dict) else None
        try:
            parsed_crs = rasterio.crs.CRS.from_user_input(crs_name)
        except (TypeError, ValueError) as error:
            raise RuntimeError("台湾陆域边界 CRS 声明无效") from error
        if parsed_crs.to_epsg() != 4326 and parsed_crs.to_string().upper() != "OGC:CRS84":
            raise RuntimeError("台湾陆域边界必须使用 WGS84 经纬度坐标")

    polygons = []
    for index, feature in enumerate(features, start=1):
        if (not isinstance(feature, dict) or feature.get("type") != "Feature"
                or not isinstance(feature.get("geometry"), dict)):
            raise RuntimeError(f"台湾陆域边界要素 {index} 缺少 geometry")
        try:
            geometry = shapely_shape(feature["geometry"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"台湾陆域边界要素 {index} 的 geometry 无效") from error
        if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
            raise RuntimeError(f"台湾陆域边界要素 {index} 必须是有效 Polygon 或 MultiPolygon")
        if not geometry.is_valid:
            if (geometry.geom_type != "MultiPolygon"
                    or not all(part.is_valid for part in geometry.geoms)):
                raise RuntimeError(f"台湾陆域边界要素 {index} 必须是有效 Polygon 或 MultiPolygon")
            geometry = unary_union(list(geometry.geoms))
            if (geometry.is_empty or not geometry.is_valid
                    or geometry.geom_type not in {"Polygon", "MultiPolygon"}):
                raise RuntimeError(f"台湾陆域边界要素 {index} 无法安全修复")
        bounds = geometry.bounds
        if (not all(math.isfinite(value) for value in bounds)
                or bounds[0] < -180 or bounds[2] > 180
                or bounds[1] < -90 or bounds[3] > 90):
            raise RuntimeError(f"台湾陆域边界要素 {index} 坐标超出 WGS84 范围")
        polygons.append(geometry)

    land_geometry = unary_union(polygons)
    if (land_geometry.is_empty or not land_geometry.is_valid
            or land_geometry.geom_type not in {"Polygon", "MultiPolygon"}):
        raise RuntimeError("台湾陆域边界无法安全合并")
    return land_geometry


def _taiwan_land_geometry(path: Path):
    try:
        path_text = str(path.resolve())
        content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise RuntimeError("台湾陆域边界文件不存在或不可读") from error
    if content_sha256 != _TAIWAN_LAND_MASK_SHA256:
        raise RuntimeError("台湾陆域边界文件 SHA-256 与已核验的内政部县市界版本不一致")
    return _load_taiwan_land_geometry_cached(path_text, content_sha256)


def _validate_taiwan_land_point(land_mask_path: Path, lon: float, lat: float, label: str) -> None:
    from shapely.geometry import Point

    if not _taiwan_land_geometry(land_mask_path).covers(Point(lon, lat)):
        raise ValueError(f"{label}坐标位于台湾县市陆域边界外")


def _validate_route_taiwan_land_coverage(
        land_mask_path: Path, coordinates: list[list[float]]) -> None:
    from shapely.geometry import LineString

    line = LineString([(float(point[0]), float(point[1])) for point in coordinates])
    if not _taiwan_land_geometry(land_mask_path).covers(line):
        raise RuntimeError("final_route.geojson 路线超出台湾县市陆域边界")


def _validate_route_dem_coverage(dem_path: Path, coordinates: list[list[float]]) -> None:
    """Reject a route that leaves valid DEM cells, including cells touched at corners."""
    import numpy as np
    import rasterio
    from rasterio.warp import transform

    max_route_vertex_count = 250_000
    max_sample_count = 2_000_000
    chunk_size = 10_000
    raster_segment_chunk_size = 256
    pixel_epsilon = 1e-9
    with rasterio.open(dem_path) as dataset:
        if dataset.crs is None:
            raise RuntimeError("服务器 DEM 未声明坐标系，无法核验完整路线覆盖")
        if len(coordinates) > max_route_vertex_count:
            raise RuntimeError("final_route.geojson 过长，无法安全完成 DEM 覆盖核验")

        longitudes = [float(point[0]) for point in coordinates]
        latitudes = [float(point[1]) for point in coordinates]
        projected_x, projected_y = transform(
            "EPSG:4326", dataset.crs, longitudes, latitudes,
        )
        inverse = ~dataset.transform
        pixel_points = [inverse * (x, y) for x, y in zip(projected_x, projected_y)]
        steps_by_segment = []
        sample_count = 1
        for index, (first, second) in enumerate(zip(coordinates, coordinates[1:])):
            first_col, first_row = pixel_points[index]
            second_col, second_row = pixel_points[index + 1]
            if not all(math.isfinite(value) for value in (
                    first_col, first_row, second_col, second_row)):
                raise RuntimeError("final_route.geojson 无法投影到服务器 DEM")
            steps = max(1, math.ceil(4 * max(
                abs(second_col - first_col), abs(second_row - first_row),
            )))
            sample_count += steps
            if sample_count > max_sample_count:
                raise RuntimeError("final_route.geojson 过长，无法安全完成 DEM 覆盖核验")
            steps_by_segment.append(steps)

        previous_pixel: tuple[float, float] | None = None

        def segment_intersects_cell(
                first: tuple[float, float], second: tuple[float, float], col: int, row: int,
        ) -> bool:
            x1, y1 = first
            x2, y2 = second
            dx, dy = x2 - x1, y2 - y1
            left, right = col - pixel_epsilon, col + 1 + pixel_epsilon
            top, bottom = row - pixel_epsilon, row + 1 + pixel_epsilon
            lower, upper = 0.0, 1.0
            for direction, distance in (
                    (-dx, x1 - left), (dx, right - x1),
                    (-dy, y1 - top), (dy, bottom - y1)):
                if direction == 0:
                    if distance < 0:
                        return False
                    continue
                fraction = distance / direction
                if direction < 0:
                    if fraction > upper:
                        return False
                    lower = max(lower, fraction)
                else:
                    if fraction < lower:
                        return False
                    upper = min(upper, fraction)
            return lower <= upper

        def validate_raster_segments(pixel_points: list[tuple[float, float]]) -> None:
            if len(pixel_points) < 2:
                return

            touched_cells: set[tuple[int, int]] = set()
            for first, second in zip(pixel_points, pixel_points[1:]):
                col_start = math.ceil(min(first[0], second[0]) - 1 - pixel_epsilon)
                col_stop = math.floor(max(first[0], second[0]) + pixel_epsilon)
                row_start = math.ceil(min(first[1], second[1]) - 1 - pixel_epsilon)
                row_stop = math.floor(max(first[1], second[1]) + pixel_epsilon)
                for row in range(row_start, row_stop + 1):
                    for col in range(col_start, col_stop + 1):
                        if not segment_intersects_cell(first, second, col, row):
                            continue
                        if not (0 <= col < dataset.width and 0 <= row < dataset.height):
                            raise RuntimeError(
                                "final_route.geojson 路线触及 DEM 数据范围之外的栅格像元"
                            )
                        touched_cells.add((row, col))

            if not touched_cells:
                raise RuntimeError("final_route.geojson 路线无法映射到 DEM 栅格")

            cell_centers = [
                dataset.transform * (col + 0.5, row + 0.5)
                for row, col in sorted(touched_cells)
            ]
            values = dataset.sample(cell_centers, indexes=1, masked=True)
            for value in values:
                sample = np.ma.asarray(value).reshape(-1)[0]
                if np.ma.is_masked(sample) or not math.isfinite(float(sample)):
                    raise RuntimeError(
                        "final_route.geojson 路线穿过无效 DEM 像元（包含像元边界接触）"
                    )

        def validate_chunk(chunk: list[tuple[float, float]], first_sample_index: int) -> None:
            nonlocal previous_pixel
            sample_x, sample_y = transform(
                "EPSG:4326", dataset.crs,
                [point[0] for point in chunk], [point[1] for point in chunk],
            )
            sample_pixel_points = [inverse * (x, y) for x, y in zip(sample_x, sample_y)]
            for local_index, (col, row) in enumerate(sample_pixel_points):
                if not (0 <= col < dataset.width and 0 <= row < dataset.height):
                    raise RuntimeError(
                        f"final_route.geojson 路线第 {first_sample_index + local_index + 1} "
                        "个 DEM 样点超出数据范围"
                    )

            values = dataset.sample(zip(sample_x, sample_y), indexes=1, masked=True)
            for local_index, value in enumerate(values):
                sample = np.ma.asarray(value).reshape(-1)[0]
                if np.ma.is_masked(sample) or not math.isfinite(float(sample)):
                    raise RuntimeError(
                        "final_route.geojson 路线第 "
                        f"{first_sample_index + local_index + 1} 个 DEM 样点穿过无效数据区"
                    )

            if previous_pixel is not None:
                sample_pixel_points.insert(0, previous_pixel)
            for offset in range(0, len(sample_pixel_points) - 1, raster_segment_chunk_size):
                end = min(offset + raster_segment_chunk_size + 1, len(sample_pixel_points))
                validate_raster_segments(sample_pixel_points[offset:end])
            previous_pixel = sample_pixel_points[-1]

        chunk: list[tuple[float, float]] = []
        processed_samples = 0
        for index, (first, second) in enumerate(zip(coordinates, coordinates[1:])):
            steps = steps_by_segment[index]
            for step in range(steps):
                fraction = step / steps
                chunk.append((
                    float(first[0]) + (float(second[0]) - float(first[0])) * fraction,
                    float(first[1]) + (float(second[1]) - float(first[1])) * fraction,
                ))
                if len(chunk) == chunk_size:
                    validate_chunk(chunk, processed_samples)
                    processed_samples += len(chunk)
                    chunk.clear()
        chunk.append((float(coordinates[-1][0]), float(coordinates[-1][1])))
        validate_chunk(chunk, processed_samples)


def _load_final_route_properties(
        output_dir: Path, start: dict[str, str], end: dict[str, str], dem_path: Path,
        land_mask_path: Path,
) -> tuple[Path, dict[str, Any]]:
    route_files = list(output_dir.rglob("final_route.geojson"))
    if len(route_files) != 1:
        raise RuntimeError("规划器未生成唯一 final_route.geojson")

    route_path = route_files[0]
    try:
        route_data = json.loads(route_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("规划器生成的 final_route.geojson 无法读取") from error
    features = route_data.get("features") if isinstance(route_data, dict) else None
    if (not isinstance(route_data, dict) or route_data.get("type") != "FeatureCollection"
            or not isinstance(features, list) or len(features) != 1):
        raise RuntimeError("final_route.geojson 必须包含唯一路线要素")

    feature = features[0]
    geometry = feature.get("geometry") if isinstance(feature, dict) else None
    coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
    if (not isinstance(feature, dict) or feature.get("type") != "Feature"
            or not isinstance(geometry, dict) or geometry.get("type") != "LineString"
            or not isinstance(coordinates, list) or len(coordinates) < 2):
        raise RuntimeError("final_route.geojson 未包含有效 LineString 路线")
    for point in coordinates:
        if (not isinstance(point, list) or len(point) < 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) for value in point)):
            raise RuntimeError("final_route.geojson 含无效经纬度坐标")
        lon, lat = map(float, point[:2])
        if not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise RuntimeError("final_route.geojson 含无效经纬度坐标")
    _validate_route_dem_coverage(dem_path, coordinates)
    _validate_route_taiwan_land_coverage(land_mask_path, coordinates)

    properties = feature.get("properties")
    required = {
        "planning_status", "hard_constraint_status", "hard_violations", "length_km",
        "dem_sample_coverage_ratio", "dem_sample_count", "dem_valid_sample_count",
        "engineering_feasibility", "from_station_id", "to_station_id",
    }
    if not isinstance(properties, dict) or not required.issubset(properties):
        raise RuntimeError("final_route.geojson 缺少路线状态、约束或 DEM 覆盖字段")
    if properties["engineering_feasibility"] != "not_evaluated":
        raise RuntimeError("final_route.geojson 工程可行性必须明确标记为未评估")
    if properties["from_station_id"] != start["station_id"]:
        raise RuntimeError("final_route.geojson 起点站点编号与请求不一致")
    if properties["to_station_id"] != end["station_id"]:
        raise RuntimeError("final_route.geojson 终点站点编号与请求不一致")
    for label, point, station in (("起点", coordinates[0], start), ("终点", coordinates[-1], end)):
        if (not math.isclose(float(point[0]), float(station["lon"]), rel_tol=0.0, abs_tol=1e-6)
                or not math.isclose(float(point[1]), float(station["lat"]), rel_tol=0.0, abs_tol=1e-6)):
            raise RuntimeError(f"final_route.geojson {label}首尾坐标与请求端点不一致")
    planning_status = properties["planning_status"]
    if not isinstance(planning_status, str) or planning_status not in {
        "exploratory_only_unverified_endpoints", "screening_candidate",
        "screening_candidate_data_incomplete", "screening_candidate_coverage_unverified",
        "no_hard_constraint_compliant_route",
    }:
        raise RuntimeError("final_route.geojson 路线状态无效")
    if planning_status == "no_hard_constraint_compliant_route":
        raise RuntimeError("规划器未生成满足硬约束的路线")
    hard_status = properties["hard_constraint_status"]
    if not isinstance(hard_status, str) or hard_status != "pass":
        raise RuntimeError("final_route.geojson 硬约束状态无效")
    if (isinstance(properties["hard_violations"], bool)
            or not isinstance(properties["hard_violations"], int)
            or properties["hard_violations"] < 0):
        raise RuntimeError("final_route.geojson 硬约束违规数无效")
    if properties["hard_violations"] != 0:
        raise RuntimeError("final_route.geojson 存在硬约束违规")
    try:
        length_km = float(properties["length_km"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("final_route.geojson 路线长度无效") from error
    if isinstance(properties["length_km"], bool) or not math.isfinite(length_km) or length_km <= 0:
        raise RuntimeError("final_route.geojson 路线长度无效")
    sample_count = properties["dem_sample_count"]
    valid_sample_count = properties["dem_valid_sample_count"]
    coverage = properties["dem_sample_coverage_ratio"]
    if (isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0
            or isinstance(valid_sample_count, bool) or not isinstance(valid_sample_count, int)
            or not 0 <= valid_sample_count <= sample_count
            or isinstance(coverage, bool) or not isinstance(coverage, (int, float))
            or not math.isfinite(float(coverage)) or not 0 <= float(coverage) <= 1
            or abs(round(valid_sample_count / sample_count, 4) - float(coverage)) > 0.0001):
        raise RuntimeError("final_route.geojson DEM 覆盖统计无效或不一致")
    return route_path, properties


class GeographicRouteJobs:
    """Run one geographic DEM-planning job at a time in the approved server root."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_job: str | None = None

    def _config(self) -> dict[str, Any] | None:
        package_root = Path(os.environ.get("TRANSMISSION_PLANNER_ROOT", ""))
        python = Path(os.environ.get("TRANSMISSION_PLANNER_PYTHON", ""))
        dem = Path(os.environ.get("TRANSMISSION_PLANNER_DEM", ""))
        job_root = Path(os.environ.get("TRANSMISSION_PLANNER_JOB_ROOT", ""))
        access_token = os.environ.get("TRANSMISSION_PLANNER_ACCESS_TOKEN", "")
        if not all(str(value) for value in (package_root, python, dem, job_root)):
            return None
        if len(access_token) < 32:
            return None
        if not package_root.is_absolute() or not python.is_absolute() or not dem.is_absolute() or not job_root.is_absolute():
            return None
        if (not all(_inside(path, _ALLOWED_ROOT) for path in (package_root, dem, job_root))
                or not _declared_inside(python, _ALLOWED_ROOT)):
            return None
        package = package_root / "transmission_path_planner"
        if not package.is_dir() or not python.is_file() or not dem.is_file():
            return None
        try:
            max_side = int(os.environ.get("TRANSMISSION_PLANNER_MAX_SIDE", "4096"))
            timeout = int(os.environ.get("TRANSMISSION_PLANNER_TIMEOUT_SECONDS", "3600"))
        except ValueError:
            return None
        if not 1024 <= max_side <= 6400 or not 60 <= timeout <= 10800:
            return None
        config: dict[str, Any] = {
            "package_root": package_root.resolve(),
            "python": Path(os.path.abspath(python)),
            "dem": dem.resolve(),
            "job_root": job_root.resolve(),
            "access_token": access_token,
            "max_side": max_side,
            "timeout": timeout,
        }
        optional_paths = (
            ("osm", "TRANSMISSION_PLANNER_OSM_CACHE_DIR", "dir"),
            ("constraints", "TRANSMISSION_PLANNER_CONSTRAINT_GEOJSON", "file"),
            ("land_mask", "TRANSMISSION_PLANNER_LAND_MASK_GEOJSON", "file"),
            ("hazards", "TRANSMISSION_PLANNER_OFFICIAL_HAZARDS_DIR", "dir"),
            ("population", "TRANSMISSION_PLANNER_POPULATION_RASTER", "file"),
        )
        for key, variable, path_type in optional_paths:
            raw = os.environ.get(variable)
            if raw:
                path = Path(raw)
                if not path.is_absolute() or not _inside(path, _ALLOWED_ROOT) or not path.exists():
                    return None
                if path_type == "dir" and not path.is_dir():
                    return None
                if path_type == "file" and not path.is_file():
                    return None
                config[key] = path.resolve()
        return config

    def public_status(self) -> dict[str, Any]:
        config = self._config()
        bounds = None
        dem_crs_supported = False
        dem_crs_error = None
        land_mask_ready = False
        valid_land_dem_data = False
        if config:
            try:
                _require_supported_planner_dem_crs(config["dem"])
                dem_crs_supported = True
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                dem_crs_error = str(error)
            try:
                bounds = _dem_bounds_wgs84(config["dem"])
            except (ImportError, OSError, RuntimeError, ValueError):
                pass
            if "land_mask" in config:
                try:
                    from shapely.geometry import box

                    land_geometry = _taiwan_land_geometry(config["land_mask"])
                    overlap = land_geometry.intersection(box(
                        bounds["west"], bounds["south"], bounds["east"], bounds["north"],
                    )) if bounds else None
                    land_mask_ready = bool(overlap is not None and overlap.area > 0)
                    if land_mask_ready and dem_crs_supported:
                        valid_land_dem_data = _dem_has_valid_taiwan_land_cell(
                            config["dem"], land_geometry,
                        )
                except (ImportError, OSError, RuntimeError, ValueError):
                    pass
        ready = (config is not None and dem_crs_supported and bounds is not None and land_mask_ready
                 and valid_land_dem_data)
        if ready:
            message = "台湾 DEM 与陆域边界已就绪"
        elif config is None:
            message = "真实 DEM 规划未启用；请在服务器配置 pp 内的规划器、DEM 和任务输出目录"
        elif bounds is None:
            message = "无法读取服务器 DEM 覆盖范围；规划未启用"
        elif not dem_crs_supported:
            message = dem_crs_error or "DEM 坐标系不受当前规划器支持；请先重投影到 EPSG:4326"
        elif land_mask_ready and not valid_land_dem_data:
            message = "台湾陆域范围内没有有效 DEM 像元；台湾范围规划已禁用"
        else:
            message = "台湾陆域边界 GeoJSON 未配置、版本未核验或与 DEM 不相交；台湾范围规划已禁用"
        return {
            "ready": ready,
            "mode": "server_dem_station_route" if ready else "disabled",
            "message": message,
            "max_side": config["max_side"] if config else None,
            "dem_bounds_wgs84": bounds,
            "dem_crs_supported": dem_crs_supported,
            "engineering_feasibility": "not_evaluated",
            "engineering_gate": _engineering_gate_metadata(),
            "access_token_required": True,
            "data_layers": ({
                "dem": dem_crs_supported, "osm": "osm" in config,
                "project_constraints": "constraints" in config,
                "land_mask": land_mask_ready,
                "official_hazards": "hazards" in config,
                "population": "population" in config,
            } if config else {}),
        }

    def authorize(self, supplied: str | None) -> bool:
        config = self._config()
        return bool(config and isinstance(supplied, str)
                    and hmac.compare_digest(supplied, config["access_token"]))

    @staticmethod
    def _station(value: Any, label: str, mode: str) -> dict[str, str]:
        if not isinstance(value, dict):
            raise ValueError(f"{label}站点信息无效")
        station_id = str(value.get("station_id", "")).strip()
        name = str(value.get("name", "")).strip()
        if not station_id or len(station_id) > 64 or not name or len(name) > 120:
            raise ValueError(f"{label}站点编号和名称必填，长度不能超过 120")
        lat = _number(value.get("lat"), f"{label}纬度", -90, 90)
        lon = _number(value.get("lon"), f"{label}经度", -180, 180)
        if mode == "verified":
            raise ValueError("当前服务没有权威站点登记名录，无法独立核验用户填写的来源或凭证；请使用探索性端点")
        point_role = "unknown"
        source = "用户录入的探索性端点"
        reference = "未核验"
        verification = "unverified"
        return {
            "station_id": station_id, "name": name, "lat": str(lat), "lon": str(lon),
            "coordinate_crs": "EPSG:4326", "voltage_kv": "", "station_type": "",
            "source": source, "verified": verification, "point_role": point_role,
            "verification_status": verification, "verification_reference": reference,
            "ground_elevation_m": "", "ground_elevation_source": "", "vertical_datum": "",
        }

    @staticmethod
    def _validate_dem_point(dem_path: Path, station: dict[str, str], label: str) -> None:
        import numpy as np
        import rasterio
        from rasterio.warp import transform

        with rasterio.open(dem_path) as dataset:
            if dataset.crs is None:
                raise RuntimeError("服务器 DEM 未声明坐标系，无法校验端点")
            x, y = transform("EPSG:4326", dataset.crs,
                             [float(station["lon"])], [float(station["lat"])])
            projected_x, projected_y = x[0], y[0]
            bounds = dataset.bounds
            if not (bounds.left <= projected_x <= bounds.right
                    and bounds.bottom <= projected_y <= bounds.top):
                raise ValueError(f"{label}坐标超出当前台湾 DEM 数据范围")
            sample = next(dataset.sample([(projected_x, projected_y)], masked=True))[0]
            if np.ma.is_masked(sample) or not np.isfinite(float(sample)):
                raise ValueError(f"{label}坐标落在台湾 DEM 无有效数据区域，不能用于规划")

    def validate_point(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._config()
        if config is None:
            raise RuntimeError("服务器真实 DEM 规划未配置")
        if "land_mask" not in config:
            raise RuntimeError("未配置台湾陆域边界 GeoJSON，无法进行台湾范围点选")
        lat = _number(payload.get("lat"), "纬度", -90, 90)
        lon = _number(payload.get("lon"), "经度", -180, 180)
        _require_supported_planner_dem_crs(config["dem"])
        self._validate_dem_point(
            config["dem"], {"lat": str(lat), "lon": str(lon)}, "地图所选位置",
        )
        _validate_taiwan_land_point(config["land_mask"], lon, lat, "地图所选位置")
        return {"valid": True, "lat": lat, "lon": lon}

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self._config()
        if config is None:
            raise RuntimeError("服务器真实 DEM 规划未配置或配置路径不在 /data1/B24041626/pp 内")
        if "land_mask" not in config:
            raise RuntimeError("未配置台湾陆域边界 GeoJSON，台湾范围规划已禁用")
        _require_supported_planner_dem_crs(config["dem"])
        mode = payload.get("endpoint_mode")
        if mode not in {"verified", "exploratory"}:
            raise ValueError("endpoint_mode 只能是 verified 或 exploratory")
        start = self._station(payload.get("start"), "起点", mode)
        end = self._station(payload.get("end"), "终点", mode)
        if start["station_id"] == end["station_id"]:
            raise ValueError("起点与终点编号不能相同")
        self._validate_dem_point(config["dem"], start, "起点")
        self._validate_dem_point(config["dem"], end, "终点")
        _validate_taiwan_land_point(
            config["land_mask"], float(start["lon"]), float(start["lat"]), "起点",
        )
        _validate_taiwan_land_point(
            config["land_mask"], float(end["lon"]), float(end["lat"]), "终点",
        )
        voltage = str(payload.get("voltage_kv", "")).strip()
        if len(voltage) > 32:
            raise ValueError("电压等级文本过长")
        job_id = uuid.uuid4().hex
        job_dir = config["job_root"] / job_id
        with self._lock:
            if self._active_job is not None:
                raise RuntimeError("服务器正在运行另一条真实 DEM 线路规划，请等待其结束")
            self._active_job = job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
            self._write_state(job_dir, {
                "job_id": job_id, "state": "queued", "created_at": self._now(),
                "endpoint_qualification": "declared_verified_line_connection_points" if mode == "verified" else "exploratory_only_unverified_endpoints",
                "planning_status": "queued", "engineering_feasibility": "not_evaluated",
                "engineering_gate": _engineering_gate_metadata(),
                "constraints_configured": "constraints" in config,
                "data_layers": self.public_status()["data_layers"],
                "message": "任务已排队，规划仅在服务器执行",
            })
            threading.Thread(target=self._run, args=(job_id, config, job_dir, start, end, voltage, mode), daemon=True).start()
        except Exception:
            with self._lock:
                if self._active_job == job_id:
                    self._active_job = None
            raise
        return self.get_status(job_id)

    @staticmethod
    def _now() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _write_state(job_dir: Path, state: dict[str, Any]) -> None:
        temporary = job_dir / f".state-{threading.get_ident()}.tmp"
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(job_dir / "state.json")

    def _run(self, job_id: str, config: dict[str, Any], job_dir: Path,
             start: dict[str, str], end: dict[str, str], voltage: str, mode: str) -> None:
        output_dir = job_dir / "results"
        stations_file = job_dir / "stations.csv"
        tasks_file = job_dir / "route_tasks.csv"
        state: dict[str, Any] = {
            "job_id": job_id, "state": "running", "created_at": self._now(),
            "endpoint_qualification": "declared_verified_line_connection_points" if mode == "verified" else "exploratory_only_unverified_endpoints",
            "planning_status": "running", "engineering_feasibility": "not_evaluated",
            "engineering_gate": _engineering_gate_metadata(),
            "constraints_configured": "constraints" in config,
            "data_layers": self.public_status()["data_layers"],
            "message": "DEM 读取、地形代价计算和路线搜索均在服务器端运行",
        }
        try:
            self._write_state(job_dir, state)
            with stations_file.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(start))
                writer.writeheader()
                writer.writerows((start, end))
            with tasks_file.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("route_id", "from_station_id", "to_station_id", "voltage_kv"))
                writer.writeheader()
                writer.writerow({"route_id": f"web-{job_id[:12]}", "from_station_id": start["station_id"], "to_station_id": end["station_id"], "voltage_kv": voltage})
            output_dir.mkdir()
            argv = [str(config["python"]), "-m", "transmission_path_planner",
                    "--dem", str(config["dem"]), "--stations-csv", str(stations_file),
                    "--route-tasks-csv", str(tasks_file), "--out-dir", str(output_dir),
                    "--max-side", str(config["max_side"]), "--no-ppp",
                    "--official-hazard-weight", "1.0", "--settlement-weight", "2.0",
                    "--ridge-penalty-weight", "0.1", "--landcover-penalty-weight", "0.0",
                    "--grade-penalty-weight", "0.0", "--terrain-slope-weight", "0.75",
                    "--roughness-weight", "0.25", "--astar-heuristic-weight", "1.0",
                    "--building-clearance-m", "0", "--window-margin-deg", "0.3"]
            if mode == "exploratory":
                argv.append("--allow-unverified-endpoints")
            if "osm" in config:
                argv.extend(("--osm-cache-dir", str(config["osm"])))
            if "constraints" in config:
                argv.extend(("--constraint-geojson", str(config["constraints"])))
            if "land_mask" in config:
                argv.extend(("--land-mask-geojson", str(config["land_mask"])))
            if "hazards" in config:
                argv.extend(("--official-hazards-dir", str(config["hazards"])))
            if "population" in config:
                argv.extend(("--population-raster", str(config["population"]),
                             "--population-weight", "1.0",
                             "--population-nodata-policy", "maximum"))
            env = os.environ.copy()
            env["PYTHONPATH"] = str(config["package_root"]) + os.pathsep + env.get("PYTHONPATH", "")
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["OMP_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            with (job_dir / "run.log").open("w", encoding="utf-8") as log_stream:
                result = subprocess.run(argv, cwd=config["package_root"], env=env,
                                        stdout=log_stream, stderr=subprocess.STDOUT,
                                        timeout=config["timeout"], check=False)
            if result.returncode != 0:
                raise RuntimeError(f"规划命令返回 {result.returncode}；请查看任务日志")
            map_files = list(output_dir.rglob("path_comparison_*.png"))
            profile_files = list(output_dir.rglob("elevation_profile.csv"))
            provenance_files = list(output_dir.rglob("run_provenance.json"))
            route_path, properties = _load_final_route_properties(
                output_dir, start, end, config["dem"], config["land_mask"],
            )
            state.update({
                "state": "succeeded", "finished_at": self._now(),
                "planning_status": properties["planning_status"],
                "hard_constraint_status": properties.get("hard_constraint_status"),
                "hard_violations": properties.get("hard_violations"),
                "length_km": properties.get("length_km"),
                "data_coverage": properties.get("dem_sample_coverage_ratio"),
                "dem_sample_count": properties.get("dem_sample_count"),
                "dem_valid_sample_count": properties.get("dem_valid_sample_count"),
                "engineering_feasibility": "not_evaluated",
                "message": "规划计算完成；这表示任务执行完毕，不表示工程可行",
                "artifacts": {"route": bool(route_path), "map": bool(map_files),
                              "profile": bool(profile_files), "provenance": bool(provenance_files), "log": True},
            })
            self._write_state(job_dir, state)
        except subprocess.TimeoutExpired:
            if not (job_dir / "run.log").is_file():
                (job_dir / "run.log").write_text("规划进程超过运行时限。\n", encoding="utf-8")
            state.update({"state": "failed", "finished_at": self._now(), "planning_status": "timeout",
                          "message": "服务器规划超过允许时长；请查看任务日志",
                          "artifacts": {"log": True}})
            self._write_state(job_dir, state)
        except Exception as error:
            state.update({"state": "failed", "finished_at": self._now(), "planning_status": "failed",
                          "message": str(error)[:500]})
            if (job_dir / "run.log").is_file():
                state["artifacts"] = {"log": True}
            self._write_state(job_dir, state)
        finally:
            with self._lock:
                if self._active_job == job_id:
                    self._active_job = None

    def get_status(self, job_id: str) -> dict[str, Any]:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise ValueError("任务编号无效")
        config = self._config()
        if config is None:
            raise RuntimeError("服务器真实 DEM 规划未配置")
        job_dir = config["job_root"] / job_id
        state_file = job_dir / "state.json"
        if not _inside(job_dir, config["job_root"]) or not state_file.is_file():
            raise KeyError(job_id)
        state = json.loads(state_file.read_text(encoding="utf-8"))
        with self._lock:
            active = self._active_job == job_id
        if state.get("state") in {"queued", "running"} and not active:
            state.update({"state": "interrupted", "finished_at": self._now(),
                          "planning_status": "interrupted",
                          "message": "服务进程已重启，任务状态无法继续；原输出保留供检查"})
            self._write_state(job_dir, state)
        return state

    def artifact(self, job_id: str, kind: str) -> tuple[Path, str]:
        if kind not in _ARTIFACTS:
            raise KeyError(kind)
        state = self.get_status(job_id)
        config = self._config()
        assert config is not None
        job_dir = config["job_root"] / job_id
        if state["state"] != "succeeded" and kind != "log":
            raise FileNotFoundError(kind)
        output_dir = job_dir / "results"
        candidates: list[Path]
        if kind == "route":
            candidates = list(output_dir.rglob("final_route.geojson"))
        elif kind == "map":
            candidates = list(output_dir.rglob("path_comparison_*.png"))
        elif kind == "profile":
            candidates = list(output_dir.rglob("elevation_profile.csv"))
        elif kind == "provenance":
            candidates = list(output_dir.rglob("run_provenance.json"))
        else:
            candidates = [job_dir / "run.log"]
        if len(candidates) != 1 or not candidates[0].is_file() or not _inside(candidates[0], job_dir):
            raise FileNotFoundError(kind)
        return candidates[0], _ARTIFACTS[kind]

    def read_artifact(self, job_id: str, kind: str) -> tuple[bytes, str]:
        path, extension = self.artifact(job_id, kind)
        body = path.read_bytes()
        if kind != "route":
            return body, extension

        try:
            route_data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("final_route.geojson 无法读取") from error
        features = route_data.get("features") if isinstance(route_data, dict) else None
        if (not isinstance(route_data, dict) or route_data.get("type") != "FeatureCollection"
                or not isinstance(features, list)):
            raise RuntimeError("final_route.geojson 不是有效 FeatureCollection")

        gate = self.get_status(job_id).get("engineering_gate")
        if not isinstance(gate, dict):
            gate = _engineering_gate_metadata()
        for feature in features:
            properties = feature.get("properties") if isinstance(feature, dict) else None
            if not isinstance(properties, dict):
                raise RuntimeError("final_route.geojson 要素缺少 properties")
            properties["engineering_gate"] = gate
        return json.dumps(route_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), extension


GEOGRAPHIC_ROUTE_JOBS = GeographicRouteJobs()
