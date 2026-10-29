# transmission_path_planner/cli.py
"""命令行入口与多走廊批量编排。"""

import hashlib
import re

from .config import *
from .geometry import *
from .data import *
from .osm import *
from .cost import *
from .methods import *
from .smoothing import *
from .metrics import *
from .visualization import *
from .io_export import *
from .stations import *
from .official_hazards import load_official_hazard_masks
from .provenance import capture_run_provenance, write_run_provenance
from .experimental_theta_star import (
    integrate_polyline_cost,
    theta_star_cost_path,
)
from .experimental_tower_graph import span_clearance_status, tower_graph_route
from .experimental_astar_16 import (
    NEIGHBORS_8,
    NEIGHBORS_16,
    astar_exact_integral_path,
    integrate_path_cost,
)
from .diverse_routes import sampled_pareto_astar_routes
from shapely import make_valid


FINAL_ROUTE_LABEL = "最终路线"
ASTAR_16_METHOD = "A*(16-neighbor experimental)"
ASTAR_8_EXACT_METHOD = "A*(8-neighbor exact experimental)"
ASTAR_8_SHARED_METHOD = "A*(8-neighbor shared objective experimental)"
ASTAR_16_SHARED_METHOD = "A*(16-neighbor shared objective experimental)"
THETA_SHARED_METHOD = "Theta*(shared objective experimental)"
TOWER_GRAPH_METHOD = "Tower graph (experimental)"
SHARED_SEARCH_ROUTE_TAGS = {
    ASTAR_8_SHARED_METHOD: "astar_8_shared_objective",
    ASTAR_16_SHARED_METHOD: "astar_16_shared_objective",
    THETA_SHARED_METHOD: "theta_shared_objective",
}


def _select_final_route(methods, case_metrics, select_tower_graph=False):
    if select_tower_graph:
        metrics = case_metrics.get(TOWER_GRAPH_METHOD, {})
        site_slope_limit = metrics.get("tower_site_max_surface_slope_deg")
        site_slope_assessed = (
            metrics.get("tower_site_slope_status") == "interior_candidates_screened"
            and isinstance(site_slope_limit, (int, float))
            and not isinstance(site_slope_limit, bool)
            and math.isfinite(float(site_slope_limit))
            and 0 < float(site_slope_limit) <= 90
        )
        clearance_spans = metrics.get("clearance_profile_spans", [])
        clearance_passed = (
            metrics.get("clearance_profile_status") in {
                "dem_only_profile_screening",
                "station_elevation_anchored_profile_screening",
            }
            and bool(clearance_spans)
            and all(span.get("status") == "sampled_clearance_ok"
                    for span in clearance_spans)
        )
        if (methods.get(TOWER_GRAPH_METHOD)
                and metrics.get("candidate_validation_status") == "survived_configured_screening"
                and metrics.get("hard_constraint_status") == "pass"
                and metrics.get("hard_violations") == 0
                and metrics.get("building_geometry_status") == "pass"
                and metrics.get("tower_span_status") == "pass"
                and metrics.get("tower_span_violation_count") == 0
                and site_slope_assessed
                and metrics.get("tower_site_pad_status") == "site_pad_footprint_screened"
                and isinstance(metrics.get("tower_site_pad_radius_m"), (int, float))
                and not isinstance(metrics.get("tower_site_pad_radius_m"), bool)
                and math.isfinite(float(metrics["tower_site_pad_radius_m"]))
                and metrics["tower_site_pad_radius_m"] > 0
                and clearance_passed):
            return TOWER_GRAPH_METHOD, None, "explicit_tower_graph_configured_screens_passed"
        return None, None, "no_hard_constraint_compliant_route"

    for method, rule in (
        ("A*(turn-aware)", "turn_penalized_screening_candidate"),
        ("A*(baseline)", "astar_min_cost_priority"),
        ("A*(weighted)", "weighted_astar_only_candidate"),
    ):
        metrics = case_metrics.get(method, {})
        if methods.get(method) and metrics.get("hard_violations") == 0:
            return method, None, rule
    return None, None, "no_hard_constraint_compliant_route"


def _candidate_validation_status(search_path, validated_path):
    if validated_path:
        return "survived_validation"
    if search_path:
        return "rejected_after_search"
    return "no_path"


def _shared_search_route_exports(neighbor_results, theta_results, transform):
    experiments = {**neighbor_results, **theta_results}
    routes = []
    for method, tag in SHARED_SEARCH_ROUTE_TAGS.items():
        path = experiments.get(method, {}).get("path")
        if path and len(path) >= 2:
            geo = [rowcol_to_geo(row, col, transform) for row, col in path]
            routes.append((method, tag, geo))
    return routes


def _route_method_export_tag(method):
    return re.sub(r"[^a-z0-9]+", "_", method.lower().replace("*", "star")).strip("_")


def _shared_route_artifact_paths(case_name, method, search_path, validated_path):
    tag = SHARED_SEARCH_ROUTE_TAGS[method]
    case_dir = route_output_name(case_name)
    return {
        "search_route_artifact": (
            f"{case_dir}/planned_route_{tag}_search_raw.geojson"
            if search_path and len(search_path) >= 2 else None),
        "validated_route_artifact": (
            f"{case_dir}/planned_route_{tag}_validated.geojson"
            if validated_path and len(validated_path) >= 2 else None),
    }


def _assert_route_artifacts_absent(directory, stems):
    existing = [
        path for stem in stems for path in Path(directory).glob(f"{stem}.*")
    ]
    if existing:
        raise FileExistsError(
            "refusing to overwrite route artifacts: "
            + ", ".join(str(path) for path in existing))


def _write_comparison_outputs(all_results, output_dir):
    output_dir = Path(output_dir)
    try:
        plot_algo_comparison(all_results, output_dir / "algo_comparison.png")
    except Exception as exc:
        print(f"[WARN] 聚合对比图写出失败(不影响指标写出): {exc}")
    try:
        with open(output_dir / "comparison_metrics.json", "w", encoding="utf-8") as stream:
            json.dump(all_results, stream, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[WARN] 聚合指标JSON写出失败: {exc}")


def _metric_grade_penalty_weight(method, experimental_neighbor_methods,
                                 configured_weight, experimental_theta_methods=None):
    if method in experimental_neighbor_methods:
        return experimental_neighbor_methods[method].get("grade_penalty_weight", 0.0)
    if experimental_theta_methods and method in experimental_theta_methods:
        return experimental_theta_methods[method].get("grade_penalty_weight", 0.0)
    if method in (TOWER_GRAPH_METHOD, "Theta*(experimental)"):
        return 0.0
    return configured_weight


def _method_route_metrics(results, method):
    route_metrics = []
    for result in results:
        metrics = result.get("metrics", {}).get(method)
        if isinstance(metrics, dict) and "length_km" in metrics and "mean_cost" in metrics:
            route_metrics.append(metrics)
    return route_metrics


def attach_station_endpoints(geo, station_task):
    """Replace snapped raster-center endpoints with the supplied station coordinates."""
    if not station_task or not geo:
        return geo, {}
    start, end = station_task["from_station"], station_task["to_station"]
    start_point = (float(start["lon"]), float(start["lat"]))
    end_point = (float(end["lon"]), float(end["lat"]))
    offsets = {
        "from_station_to_grid_center_offset_m": round(haversine_km(*start_point, *geo[0]) * 1000.0, 2),
        "to_station_to_grid_center_offset_m": round(haversine_km(*end_point, *geo[-1]) * 1000.0, 2),
    }
    if len(geo) == 1:
        return [start_point, end_point], offsets
    adjusted = list(geo)
    adjusted[0], adjusted[-1] = start_point, end_point
    return adjusted, offsets


def expand_single_cell_route(geo, start_ll, end_ll):
    """Turn a one-cell route into the short segment between distinct input endpoints."""
    if not geo or len(geo) != 1:
        return geo
    if tuple(start_ll) == tuple(end_ll):
        raise ValueError("起终点坐标相同，不能生成零长度线路")
    return [(start_ll[1], start_ll[0]), (end_ll[1], end_ll[0])]


def attach_input_endpoints(geo, start_ll, end_ll):
    """Restore both exact input endpoints after routing on snapped raster cells."""
    if not geo:
        return geo
    start_xy = (float(start_ll[1]), float(start_ll[0]))
    end_xy = (float(end_ll[1]), float(end_ll[0]))
    if len(geo) == 1:
        if start_xy == end_xy:
            raise ValueError("起终点坐标相同，不能生成零长度线路")
        return [start_xy, end_xy]
    adjusted = list(geo)
    adjusted[0], adjusted[-1] = start_xy, end_xy
    return adjusted


def validate_route_hard_constraints(geo, transform, hard_mask):
    """Reject a candidate whose exported geometry crosses a blocked grid cell."""
    if not geo:
        return None, {"hard_constraint_status": "no_path", "hard_violations": 0}
    violations = path_hard_violations(geo, transform, hard_mask)
    report = {
        "hard_constraint_status": "pass" if violations == 0 else "fail",
        "hard_violations": int(violations),
        "hard_constraint_validation_scope": "loaded_raster_masks_only",
        "hard_constraint_completeness_status": "not_independently_verified",
    }
    return (geo if violations == 0 else None), report


def validate_tower_span_lengths(geo, min_span_m, max_span_m, tolerance_m=0.5):
    """Recheck every final tower span after exact endpoint restoration."""
    if not geo or len(geo) < 2:
        return None, {"tower_span_status": "no_path", "tower_span_violation_count": 0,
                      "tower_span_lengths_m": []}
    lengths = [haversine_km(*first, *second) * 1000.0
               for first, second in zip(geo, geo[1:])]
    violations = [index for index, length in enumerate(lengths)
                  if length < min_span_m - tolerance_m
                  or length > max_span_m + tolerance_m]
    report = {
        "tower_span_status": "pass" if not violations else "fail",
        "tower_span_violation_count": len(violations),
        "tower_span_lengths_m": [round(length, 2) for length in lengths],
        "tower_span_invalid_indices": violations,
        "tower_span_min_m": min_span_m,
        "tower_span_max_m": max_span_m,
        "tower_span_length_method": "haversine_horizontal_distance_after_endpoint_restore",
    }
    return (geo if not violations else None), report


def tower_candidate_validation_status(geo, searched_path, clearance_status,
                                     building_geometry_status, tower_site_slope_status=None,
                                     tower_site_pad_status=None,
                                     tower_site_pad_radius_m=None):
    if str(clearance_status).startswith("unknown_"):
        return "unknown_clearance_reference"
    if not geo:
        return "rejected_after_search" if searched_path else "no_path"
    if building_geometry_status != "pass":
        return "incomplete_building_geometry_check"
    if tower_site_slope_status != "interior_candidates_screened":
        return "site_slope_not_evaluated"
    if (tower_site_pad_status != "site_pad_footprint_screened"
            or isinstance(tower_site_pad_radius_m, bool)
            or not isinstance(tower_site_pad_radius_m, (int, float))
            or not math.isfinite(float(tower_site_pad_radius_m))
            or tower_site_pad_radius_m <= 0):
        return "tower_site_pad_not_screened"
    return "survived_configured_screening"


def overhead_span_grade_report():
    """Mark ground-grade as unassessed for overhead conductor spans."""
    return {
        "grade_limit_status": "not_applicable_overhead_spans",
        "grade_limit_assessed": False,
        "grade_limit_violation_count": None,
        "grade_limit_unassessed_count": None,
        "grade_limit_max_sampled_grade_ratio": None,
        "grade_limit_threshold_ratio": None,
    }


def route_search_window_clearance_m(geo, transform, shape, pixel_spacing):
    """Return the closest route-vertex distance to the raster window edge."""
    if not geo:
        return None
    height, width = shape
    row_m, col_m = map(float, pixel_spacing)
    inverse = ~transform
    clearance = math.inf
    for lon, lat in geo:
        col, row = inverse * (lon, lat)
        clearance = min(
            clearance,
            row * row_m,
            (height - row) * row_m,
            col * col_m,
            (width - col) * col_m,
        )
    return round(float(clearance), 2)


def screening_data_quality(osm_layer_metadata):
    """Describe proxy-layer availability and coverage without treating gaps as empty land."""
    required_layers = ("buildings", "landuse")
    missing = [
        name for name in required_layers
        if osm_layer_metadata.get(name, {}).get("status") != "available"
    ]
    if missing:
        return "incomplete_osm_proxies", missing

    coverage = [osm_layer_metadata[name].get("coverage_status")
                for name in required_layers]
    if all(status == "complete" for status in coverage):
        return "osm_proxy_coverage_complete", missing
    if "features_observed" in coverage:
        return "osm_proxy_features_observed_coverage_unverified", missing
    return "osm_proxy_coverage_unverified", missing


def reject_osm_planning_layer_read_errors(osm_layer_metadata):
    """Do not plan with empty geometries produced by a failed cache read."""
    errors = [
        f"{name} ({osm_layer_metadata.get(name, {}).get('reason', 'unknown')})"
        for name in ("buildings", "landuse")
        if osm_layer_metadata.get(name, {}).get("status") == "read_error"
    ]
    if errors:
        raise RuntimeError("OSM规划图层读取失败，拒绝按空图层继续规划: " + ", ".join(errors))


def route_hazard_exposure(geo, hazard_masks, transform, shape):
    """Report raster-cell contact with hazard masks for the selected route."""
    if not geo or len(geo) < 2:
        return {"status": "route_too_short", "route_cell_count": 0, "layers": {}}
    route_mask = rasterize_on([(LineString(geo), {})], transform, shape) > 0
    route_cells = int(np.count_nonzero(route_mask))
    layers = {}
    for name, mask in hazard_masks.items():
        hazard = np.asarray(mask) > 0
        cells = int(np.count_nonzero(route_mask & hazard))
        layers[name] = {
            "route_cell_count": cells,
            "route_cell_fraction": cells / route_cells if route_cells else None,
        }
    return {
        "status": "measured" if route_cells else "route_not_rasterized",
        "route_cell_count": route_cells,
        "cell_fraction_is_approximate": True,
        "layers": layers,
    }


def route_custom_constraint_exposure(geo, constraints, transform, shape):
    """Measure rasterized route contact with configured soft constraint costs."""
    if constraints is None:
        return {"status": "disabled", "route_cell_count": 0,
                "soft_affected_route_cell_count": 0,
                "soft_affected_route_cell_fraction": None,
                "summed_soft_weight_cells": None}
    if not geo or len(geo) < 2:
        return {"status": "route_too_short", "route_cell_count": 0,
                "soft_affected_route_cell_count": 0,
                "soft_affected_route_cell_fraction": None,
                "summed_soft_weight_cells": 0.0}
    soft_cost = np.asarray(constraints["soft_cost"], dtype=np.float64)
    if soft_cost.shape != tuple(shape):
        raise ValueError("custom constraint raster does not match route grid")
    route_mask = rasterize_on([(LineString(geo), {})], transform, shape) > 0
    route_cell_count = int(np.count_nonzero(route_mask))
    touched_soft_cost = soft_cost[route_mask]
    affected_count = int(np.count_nonzero(touched_soft_cost > 0.0))
    return {
        "status": "measured" if route_cell_count else "route_not_rasterized",
        "route_cell_count": route_cell_count,
        "soft_affected_route_cell_count": affected_count,
        "soft_affected_route_cell_fraction": (
            affected_count / route_cell_count if route_cell_count else None),
        "summed_soft_weight_cells": float(touched_soft_cost.sum()),
        "fraction_denominator": "all_touched_route_cells; approximate, not length weighted",
    }


def route_population_coverage(geo, valid_population_mask, transform, shape):
    """Report whether a route crosses cells with unknown population values."""
    if valid_population_mask is None:
        return {"status": "disabled", "route_cell_count": 0,
                "unknown_route_cell_count": 0, "coverage_ratio": None}
    if not geo or len(geo) < 2:
        return {"status": "route_too_short", "route_cell_count": 0,
                "unknown_route_cell_count": 0, "coverage_ratio": None}
    route_mask = rasterize_on([(LineString(geo), {})], transform, shape) > 0
    route_cells = int(np.count_nonzero(route_mask))
    unknown_cells = int(np.count_nonzero(
        route_mask & ~np.asarray(valid_population_mask, dtype=bool)))
    return {
        "status": "complete" if unknown_cells == 0 else "incomplete_unknown_population_cells",
        "route_cell_count": route_cells,
        "unknown_route_cell_count": unknown_cells,
        "coverage_ratio": (1.0 - unknown_cells / route_cells if route_cells else None),
        "cell_fraction_is_approximate": True,
    }


def validate_endpoint_cells(start, end, shape, hard_mask, blocking_layers=None):
    """Require fixed endpoints to lie inside the DEM and the complete passable mask."""
    height, width = shape
    for label, (row, col) in (("起点", start), ("终点", end)):
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"{label}超出DEM范围")
        if not hard_mask[row, col]:
            reasons = [name for name, mask in (blocking_layers or {}).items()
                       if mask is not None and bool(mask[row, col])]
            if reasons:
                reason_text = "、".join(reasons)
                raise ValueError(f"{label}位于不可通行像元（{reason_text}）；请核对接入点坐标或约束")
            raise ValueError(f"{label}位于NoData/建筑/配置禁区内；请核对接入点坐标或约束")
    return start, end


def _tower_site_pad_structure(radius_m, pixel_spacing_m):
    radius = float(radius_m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("tower_site_pad_radius_m must be finite and positive")
    if pixel_spacing_m is None or len(pixel_spacing_m) != 2:
        raise ValueError("pixel_spacing_m is required for tower-site pad screening")
    row_m, col_m = map(float, pixel_spacing_m)
    if (not math.isfinite(row_m) or not math.isfinite(col_m)
            or row_m <= 0 or col_m <= 0):
        raise ValueError("pixel_spacing_m values must be finite and positive")

    row_extent = math.ceil(radius / row_m + 0.5)
    col_extent = math.ceil(radius / col_m + 0.5)
    row_offsets = np.arange(-row_extent, row_extent + 1)[:, None]
    col_offsets = np.arange(-col_extent, col_extent + 1)[None, :]
    row_gap = np.maximum(np.abs(row_offsets) * row_m - row_m / 2.0, 0.0)
    col_gap = np.maximum(np.abs(col_offsets) * col_m - col_m / 2.0, 0.0)
    touched_cells = row_gap ** 2 + col_gap ** 2 <= radius ** 2 + 1e-9
    rows, cols = np.nonzero(touched_cells)
    return touched_cells[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def tower_site_hazard_exposure_diagnostic(
        tower_path, hazard_masks, hazard_metadata, pixel_spacing_m,
        pad_radius_m, grid_shape=None):
    """Report known hazard-mask overlaps for intermediate tower-site pads only.

    This is diagnostic metadata: it does not change route cost, search, geometry,
    candidate selection, or identify zero-overlap sites as safe.
    """
    layer_keys = (
        "debris_flow_impact", "landslide_impact", "landslide_potential")
    intermediate_sites = list(tower_path[1:-1]) if tower_path is not None else []
    if pad_radius_m is None:
        footprint = np.ones((1, 1), dtype=bool)
        footprint_status = "center_cell_only_pad_not_configured"
    else:
        footprint = _tower_site_pad_structure(pad_radius_m, pixel_spacing_m)
        footprint_status = "configured_pad_footprint"
    center_row, center_col = footprint.shape[0] // 2, footprint.shape[1] // 2
    layer_metadata = (hazard_metadata or {}).get("layers", {})
    layer_results = {}

    for layer in layer_keys:
        mask = hazard_masks.get(layer) if isinstance(hazard_masks, dict) else None
        coverage = layer_metadata.get(layer, {}).get("coverage_status", "unknown")
        overlap_sites = 0
        overlap_cells = 0
        pads_outside_grid = 0
        mask_available = mask is not None and np.asarray(mask).ndim == 2
        if mask_available:
            mask = np.asarray(mask) > 0
            if grid_shape is not None and mask.shape != tuple(grid_shape):
                mask_available = False

        for site in intermediate_sites:
            row, col = map(int, site)
            row0, col0 = row - center_row, col - center_col
            row1, col1 = row0 + footprint.shape[0], col0 + footprint.shape[1]
            if not mask_available:
                continue
            height, width = mask.shape
            r0, c0 = max(row0, 0), max(col0, 0)
            r1, c1 = min(row1, height), min(col1, width)
            if r0 >= r1 or c0 >= c1:
                pads_outside_grid += 1
                continue
            footprint_r0, footprint_c0 = r0 - row0, c0 - col0
            footprint_r1 = footprint_r0 + r1 - r0
            footprint_c1 = footprint_c0 + c1 - c0
            touched = (mask[r0:r1, c0:c1]
                       & footprint[footprint_r0:footprint_r1,
                                   footprint_c0:footprint_c1])
            count = int(np.count_nonzero(touched))
            overlap_cells += count
            overlap_sites += int(count > 0)
            if (row0 < 0 or col0 < 0 or row1 > height or col1 > width):
                pads_outside_grid += 1

        all_pads_inside_grid = pads_outside_grid == 0
        if not intermediate_sites:
            status = "not_assessed_no_intermediate_sites"
        elif overlap_cells:
            status = "known_overlap"
        elif pad_radius_m is None:
            status = "center_cell_only_pad_not_configured"
        elif (mask_available and coverage == "complete"
              and all_pads_inside_grid):
            status = "complete_coverage_no_overlap"
        else:
            status = "unknown_coverage"
        layer_results[layer] = {
            "status": status,
            "coverage_status": coverage,
            "mask_available": mask_available,
            "overlap_intermediate_site_count": overlap_sites,
            "overlap_pad_cell_count_across_sites": overlap_cells,
            "pads_not_fully_inside_raster_grid": pads_outside_grid,
        }

    statuses = [item["status"] for item in layer_results.values()]
    if not intermediate_sites:
        overall_status = "not_assessed_no_intermediate_sites"
    elif "known_overlap" in statuses:
        overall_status = "known_overlap"
    elif all(status == "complete_coverage_no_overlap" for status in statuses):
        overall_status = "complete_coverage_no_overlap"
    else:
        overall_status = "unknown_coverage"
    return {
        "status": overall_status,
        "scope": "intermediate_tower_site_pad_footprints_only",
        "pad_radius_m": pad_radius_m,
        "pad_footprint_status": footprint_status,
        "intermediate_site_count": len(intermediate_sites),
        "layers": layer_results,
        "selection_effect": "none_diagnostic_only",
    }


def build_tower_masks(dem, slope, custom_hard_mask=None,
                      max_surface_slope_deg=None, max_terrain_elevation_m=None,
                      building_mask=None, building_distance_m=None,
                      building_clearance_m=0.0, tower_site_pad_radius_m=None,
                      pixel_spacing_m=None):
    """Separate overhead-span traversability from candidate tower-site limits.

    A configured pad radius screens every touched terrain cell conservatively;
    it is a surface-support proxy, not a geotechnical foundation assessment.
    """
    elevation = np.asarray(dem)
    surface_slope = np.asarray(slope)
    if elevation.ndim != 2 or surface_slope.shape != elevation.shape:
        raise ValueError("dem and slope must be same-shaped 2D arrays")
    span_mask = np.isfinite(elevation)
    for mask, label in ((custom_hard_mask, "custom_hard_mask"),
                        (building_mask, "building_mask")):
        if mask is not None:
            values = np.asarray(mask)
            if values.shape != elevation.shape:
                raise ValueError(f"{label} must match the DEM")
            span_mask &= values == 0
    if max_terrain_elevation_m is not None:
        span_mask &= elevation <= max_terrain_elevation_m
    if building_clearance_m > 0 and building_distance_m is not None:
        distance = np.asarray(building_distance_m)
        if distance.shape != elevation.shape:
            raise ValueError("building_distance_m must match the DEM")
        span_mask &= distance > building_clearance_m

    site_mask = span_mask.copy()
    if max_surface_slope_deg is not None:
        site_mask &= surface_slope <= max_surface_slope_deg
    if tower_site_pad_radius_m is not None:
        radius = float(tower_site_pad_radius_m)
        touched_cells = _tower_site_pad_structure(radius, pixel_spacing_m)
        if not HAS_SCIPY:
            raise RuntimeError("SciPy is required for tower-site pad screening")

        pad_data_valid = span_mask & np.isfinite(surface_slope)
        if max_surface_slope_deg is not None:
            pad_data_valid &= surface_slope <= max_surface_slope_deg
        site_mask &= binary_erosion(pad_data_valid, structure=touched_cells, border_value=0)
    return span_mask.astype(np.uint8), site_mask.astype(np.uint8)


def _geometry_intersects_features(geometry, features):
    """Check exact intersections against OSM geometry rows or bare geometries."""
    return _geometry_intersection_count(geometry, features) > 0


def _geometry_intersection_summary(geometry, features):
    """Count intersections and repaired nearby OSM geometries."""
    minx, miny, maxx, maxy = geometry.bounds
    count = 0
    repaired = 0
    for feature in features:
        obstacle = feature[0] if isinstance(feature, (tuple, list)) else feature
        left, bottom, right, top = obstacle.bounds
        if left <= maxx and right >= minx and bottom <= maxy and top >= miny:
            if not obstacle.is_valid:
                obstacle = make_valid(obstacle)
                if obstacle.is_empty or not obstacle.is_valid:
                    raise ValueError("相交建筑几何修复失败，拒绝通过精确建筑复核")
                repaired += 1
            if geometry.intersects(obstacle):
                count += 1
    return count, repaired


def _geometry_intersection_count(geometry, features):
    """Count mapped features intersecting a geometry after validity repair."""
    return _geometry_intersection_summary(geometry, features)[0]


def open_safe_building_endpoint_pixels(endpoints, hard_mask, blocking_layers,
                                       building_geometries, transform,
                                       building_clearance_m=0.0):
    """Open only endpoint cells blocked by rasterization when exact geometry is clear."""
    if not building_geometries or building_clearance_m > 0:
        return []

    exceptions = []
    for label, (row, col), point_xy in endpoints:
        if not (0 <= row < hard_mask.shape[0] and 0 <= col < hard_mask.shape[1]):
            continue
        if hard_mask[row, col]:
            continue
        reasons = [name for name, mask in blocking_layers.items()
                   if mask is not None and bool(mask[row, col])]
        if reasons != ["OSM建筑足迹"]:
            continue

        point = Point(*point_xy)
        if _geometry_intersects_features(point, building_geometries):
            continue
        center_xy = transform * (col + 0.5, row + 0.5)
        connector = LineString([point_xy, center_xy])
        if _geometry_intersects_features(connector, building_geometries):
            continue

        probe_mask = hard_mask.copy()
        probe_mask[row, col] = 1
        if path_hard_violations([point_xy, center_xy], transform, probe_mask):
            continue
        exceptions.append({"label": label, "row": row, "col": col,
                           "point_xy": tuple(point_xy)})

    for exception in exceptions:
        hard_mask[exception["row"], exception["col"]] = 1
    return exceptions


def validate_route_against_buildings(geo, building_geometries):
    """Fail closed when an exported route intersects an available building footprint."""
    if not geo:
        return None, {"building_geometry_status": "no_path",
                      "building_geometry_intersections": 0}
    route = LineString(geo) if len(geo) > 1 else Point(*geo[0])
    intersections, repaired = _geometry_intersection_summary(route, building_geometries)
    report = {
        "building_geometry_status": "fail" if intersections else "pass",
        "building_geometry_intersections": intersections,
        "building_geometry_repaired_features": repaired,
    }
    return (None if intersections else geo), report


def annotate_route_validation_context(report, endpoint_exceptions, building_metadata):
    """Record endpoint exceptions and the scope of the available building check."""
    report["endpoint_rasterization_exceptions"] = [
        {"label": item["label"], "row": int(item["row"]),
         "col": int(item["col"]), "point_xy": list(item["point_xy"])}
        for item in endpoint_exceptions
    ]
    report["building_geometry_coverage_status"] = building_metadata.get(
        "coverage_status", "unknown")
    report["building_geometry_coverage_reason"] = building_metadata.get(
        "coverage_reason")
    report["building_geometry_check_scope"] = (
        "loaded_osm_features_only"
        if building_metadata.get("status") == "available" else "unavailable")
    report["building_geometry_source_repair_count"] = int(
        building_metadata.get("geometry_repaired_count", 0))
    report["building_geometry_skipped_nonfinite_count"] = int(
        building_metadata.get("nonfinite_geometry_outside_bbox_count", 0))
    return report


def run_method_or_skip_on_oom(method_name, run_method):
    """Skip a neural candidate only for PyTorch OOM; let other errors surface."""
    try:
        return run_method(), None
    except torch.OutOfMemoryError as exc:
        message = str(exc)
    print(f"  [{method_name}] 显存不足，跳过该方法并继续其余规划: {message}")
    if cuda_is_usable(torch):
        torch.cuda.empty_cache()
    return None, message


def train_vin_or_skip_on_oom(*args, **kwargs):
    """Skip VIN on PyTorch OOM without suppressing unrelated failures."""
    result, _ = run_method_or_skip_on_oom("VIN-Grad", lambda: train_vin(*args, **kwargs))
    return result


def validate_unique_case_names(cases, check_reference_names=False):
    names = [case[0] for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("用例名称必须唯一，重复名称会覆盖输出目录")
    if check_reference_names:
        # Reference files must resolve uniquely on Windows as well as POSIX.
        reference_names = [safe_reference_name(name).casefold() for name in names]
        if len(set(reference_names)) != len(reference_names):
            raise ValueError("用例名净化后发生重复，无法安全匹配逐任务参考线路文件")


def resolve_turn_penalty_weights(legacy_weight, search_weight=None,
                                 smoothing_weight=None):
    """Resolve the legacy shared weight and optional stage-specific overrides."""
    return (
        legacy_weight if search_weight is None else search_weight,
        legacy_weight if smoothing_weight is None else smoothing_weight,
    )


def validate_cases_available(cases):
    if not cases:
        raise ValueError("未找到有效规划任务；请检查输入 CSV 或站点任务")
    if any(tuple(start) == tuple(end) for _, start, end in cases):
        raise ValueError("起终点坐标相同，不能生成零长度线路")


def load_case_reference(path, require_single_line=True):
    if not path or not Path(path).is_file():
        return None, "reference_missing"
    try:
        real_geo = load_real_path(
            path, require_single_line=require_single_line, raise_errors=True)
    except ValueError as exc:
        print(f"  [warn] 目标线路参考文件无效: {exc}")
        return None, "reference_invalid_or_empty"
    except Exception as exc:
        print(f"  [warn] 目标线路参考文件无法读取: {exc}")
        return None, "reference_invalid_or_unreadable"
    return (real_geo, "reference_loaded" if real_geo else "reference_invalid_or_empty")


def summarize_run_completion(expected_cases, results, failed_cases):
    failed_names = {item["case"] for item in failed_cases}
    final_routes = [record["metrics"].get(FINAL_ROUTE_LABEL, {})
                    for record in results]
    no_route_cases = [
        record["case"] for record, final_route in zip(results, final_routes)
        if final_route.get("planning_status") == "no_hard_constraint_compliant_route"
    ]
    no_route_names = set(no_route_cases)
    completed = {
        record["case"] for record in results
        if record["case"] not in failed_names and record["case"] not in no_route_names
    }
    exploratory_cases = [
        record["case"] for record, final_route in zip(results, final_routes)
        if record["case"] in completed
        and final_route.get("planning_status") == "exploratory_only_unverified_endpoints"
    ]
    coverage_unverified_candidates = [
        record["case"] for record, final_route in zip(results, final_routes)
        if final_route.get("planning_status", "").startswith("screening_candidate_")
    ]
    engineering_status_counts = {}
    for final_route in final_routes:
        status = final_route.get("engineering_feasibility", "not_reported")
        engineering_status_counts[status] = engineering_status_counts.get(status, 0) + 1
    return {
        "expected_cases": expected_cases,
        "completed_cases": len(completed),
        "recorded_cases": len(results),
        "failed_cases": failed_cases,
        "no_route_cases": no_route_cases,
        "exploratory_cases": exploratory_cases,
        "coverage_unverified_candidate_cases": coverage_unverified_candidates,
        "engineering_feasibility_status_counts": engineering_status_counts,
        "complete": expected_cases > 0 and len(completed) == expected_cases
        and not failed_cases and not no_route_cases,
    }


def route_planning_status(final_geo, station_task, missing_screening_layers,
                          osm_coverage_unverified=False):
    if not final_geo:
        return "no_hard_constraint_compliant_route"
    if station_task and station_task["endpoint_qualification"] != \
            "declared_verified_line_connection_points":
        return "exploratory_only_unverified_endpoints"
    if missing_screening_layers:
        return "screening_candidate_data_incomplete"
    if osm_coverage_unverified:
        return "screening_candidate_coverage_unverified"
    if station_task is None:
        return "exploratory_only_unverified_endpoints"
    return "screening_candidate"


def format_case_error(exc):
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def method_visual_similarity_summary(results, method, expected_cases=None, failed_case_names=None):
    excluded = set(failed_case_names or ())
    declared_independent = []
    exploratory = []
    for record in results:
        if record["case"] in excluded:
            continue
        final_route = record["metrics"].get(FINAL_ROUTE_LABEL, {})
        if final_route.get("planning_status") == "no_hard_constraint_compliant_route":
            continue
        has_declared_independent_endpoints = (
            final_route.get("station_endpoint_qualification")
            == "declared_verified_line_connection_points"
            and final_route.get("station_endpoint_reference_independence")
            == "declared_independent_of_reference"
        )
        (declared_independent if has_declared_independent_endpoints else exploratory).append(record)
    total_cases = max(len(results), expected_cases or 0)

    def summarize(records):
        comparisons = [record["metrics"].get(method, {}).get("route_similarity")
                       for record in records]
        comparisons = [value for value in comparisons if value]
        p95 = [value["symmetric"]["p95_m"] for value in comparisons]
        hausdorff = [value["symmetric"]["sampled_hausdorff_m"] for value in comparisons]
        coverage = [value["symmetric"]["coverage_f1_within_m"]["1000"]
                    for value in comparisons]
        coverage_100m = [value["symmetric"]["coverage_f1_within_m"].get("100")
                         for value in comparisons]
        coverage_100m = [value for value in coverage_100m if value is not None]
        length_error = [value["length"]["absolute_difference_pct"]
                        for value in comparisons]
        return {
            "endpoint_cases": len(records),
            "measured": len(comparisons),
            "unmeasured": max(0, len(records) - len(comparisons)),
            "median_symmetric_p95_m": float(np.median(p95)) if p95 else None,
            "median_sampled_hausdorff_m": float(np.median(hausdorff)) if hausdorff else None,
            "median_coverage_f1_100m": float(np.median(coverage_100m)) if coverage_100m else None,
            "median_coverage_f1_1km": float(np.median(coverage)) if coverage else None,
            "median_length_difference_pct": float(np.median(length_error)) if length_error else None,
        }
    return {
        "declared_independent_endpoints": summarize(declared_independent),
        "exploratory_or_independence_unproven": summarize(exploratory),
        "unclassified_cases": max(0, total_cases - len(declared_independent) - len(exploratory)),
    }


def validate_real_shp_argument(real_shp, cases):
    if real_shp is None:
        return
    if not Path(real_shp).is_file():
        raise ValueError("--real-shp 必须指向一个存在的目标线路文件")
    if len(cases) != 1:
        raise ValueError("--real-shp 只支持单任务目标线路；多任务请使用 --real-dir 提供逐任务线路")


def configure_console_output():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(errors="replace")


def parse_lat_lon(value, label):
    try:
        parts = [float(part) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError(f"{label}必须是纬度,经度数值") from exc
    if (len(parts) != 2 or not all(math.isfinite(part) for part in parts)
            or not -90 <= parts[0] <= 90 or not -180 <= parts[1] <= 180):
        raise ValueError(f"{label}经纬度无效；纬度须在[-90,90]、经度须在[-180,180]")
    return tuple(parts)


def main():
    configure_console_output()
    t0 = time.time()
    ap = argparse.ArgumentParser(description="复杂山区输电线路启发式路径规划 (V5.0)")
    ap.add_argument("--dem", type=str, default=None)
    ap.add_argument("--dem-vertical-datum", type=str, default=None,
                    help="用于净空筛查的 DEM 垂直基准声明；须与站点实测高程基准一致才会使用实测端点高程")
    ap.add_argument("--start", type=str, default=None, help="起点 lat,lon")
    ap.add_argument("--end", type=str, default=None, help="终点 lat,lon")
    ap.add_argument("--stations-csv", type=str, default=None,
                    help="站点表需含 station_id,lat,lon,coordinate_crs；正式任务还需声明核验过的线路接入点")
    ap.add_argument("--route-tasks-csv", type=str, default=None,
                    help="站间任务表: route_id,from_station_id,to_station_id")
    ap.add_argument("--allow-unverified-endpoints", action="store_true",
                    help="允许未核验端点仅作探索性规划；输出不得解释为正式站间候选")
    ap.add_argument("--constraint-geojson", type=str, default=None,
                    help="可选 WGS84 多边形图层; properties 需含 mode=hard/soft、source；soft 还需 weight。选塔时硬约束另需核验声明和凭证")
    ap.add_argument("--land-mask-geojson", type=str, default=None,
                    help="可选 WGS84 陆域 Polygon/MultiPolygon；多边形外区域作为硬不可通行区")
    default_hazard_dir = PACKAGE_DIR / "data" / "official_sources" / "ardswc_115_mvt_z10_20261023"
    ap.add_argument("--official-hazards-dir", type=str, default=str(default_hazard_dir),
                    help="农业部 115 年土石流/大规模崩塌 MVT 数据目录；默认读取软件包内离线图层")
    ap.add_argument("--official-hazard-weight", type=float, default=1.0,
                    help="官方土石流/崩塌图层联合软代价权重；重叠类别不重复累计，0 表示关闭")
    ap.add_argument("--turn-penalty-weight", type=float, default=0.0,
                    help="兼容参数：同时设置搜索和平滑转弯代价；可分别由下面两个参数覆盖")
    ap.add_argument("--search-turn-penalty-weight", type=float, default=None,
                    help="A* 搜索转弯代价系数；启用后使用8方向状态、内存占用显著增加")
    ap.add_argument("--smoothing-turn-penalty-weight", type=float, default=None,
                    help="约束感知平滑器转弯代价系数；默认沿用 --turn-penalty-weight")
    ap.add_argument("--settlement-weight", type=float, default=2.0,
                    help="OSM建筑密度/住宅与商业用地代理的软避让权重(方案筛选参数, 默认2.0)")
    ap.add_argument("--population-raster", type=str, default=None,
                    help="可选单波段 EPSG:4326 人口数栅格；值须为每像元人口估算数")
    ap.add_argument("--population-weight", type=float, default=0.0,
                    help="WorldPop 人口密度软成本权重；默认0关闭，需同时指定 --population-raster")
    ap.add_argument("--population-nodata-policy", choices=("maximum", "median", "zero"),
                    default="maximum",
                    help="人口栅格 NoData 搜索代价填充值；默认 maximum 保守惩罚，median/zero 仅作缺失敏感性分析")
    ap.add_argument("--ridge-penalty-weight", type=float, default=0.10,
                    help="基于300米TPI的高脊软惩罚权重；仅用地形，不读取线路几何")
    ap.add_argument("--landcover-penalty-weight", type=float, default=0.0,
                    help="OSM林地/湿地软代价权重；默认0关闭，不完整环境图层仅作可选筛查")
    ap.add_argument("--building-clearance-m", type=float, default=0.0,
                    help="建筑物栅格硬避让距离(米, 必须小于5000); 0仅屏蔽已映射建筑足迹，不代表规范净距")
    ap.add_argument("--grade-penalty-weight", type=float, default=1.0,
                    help="沿线相邻DEM高差软惩罚权重(以30%%地面坡度作为一个参考级)")
    ap.add_argument("--terrain-slope-weight", type=float, default=0.75,
                    help="DEM坡度面软代价权重；默认0.75，0可用于消融测试")
    ap.add_argument("--roughness-weight", type=float, default=0.25,
                    help="900米地形粗糙度软代价权重；默认0.25，0可用于消融测试")
    ap.add_argument("--astar-heuristic-weight", type=float, default=1.0,
                    help="A*启发函数权重；1.0对当前离散代价面求最优，大于1时为近似搜索且不保证误差界")
    ap.add_argument("--experimental-theta-star", action="store_true",
                    help="额外生成 Theta* 实验候选；复用 A* 栅格代价面，不计软坡度/转弯代价，"
                         "经常规约束复核但不参与最终选路")
    ap.add_argument("--experimental-theta-star-shared-objective", action="store_true",
                    help="额外生成 Theta* 搜索空间对照；需设 --grade-penalty-weight 0 以保持路径级目标一致；"
                         "地形坡度栅格与硬坡度限制仍生效；不计转弯代价，仅作对照且不参与最终选路")
    ap.add_argument("--experimental-astar-16", action="store_true",
                    help="额外运行同目标的 8/16 邻域精确积分 A* 对照；复用含地形与约束图层的代价面，"
                         "关闭相邻地面坡度附加惩罚和转弯惩罚，"
                         "配置的地面坡度上限仍作硬约束；均不参与最终选路")
    ap.add_argument("--experimental-astar-16-shared-objective", action="store_true",
                    help="额外运行 8/16 邻域 A* 搜索空间对照；需设 --grade-penalty-weight 0 以保持路径级目标一致；"
                         "地形坡度栅格与硬坡度限制仍生效；不含转弯惩罚，仅作对照且不参与最终选路")
    ap.add_argument("--experimental-sampled-pareto", action="store_true",
                    help="对同一地形/约束代价面采样生成 A* 多目标候选；只输出实验候选，不参与最终选路")
    ap.add_argument("--experimental-tower-graph", action="store_true",
                    help="额外生成带候选塔位和档距约束的实验路线；默认仅作筛选对照")
    ap.add_argument("--select-experimental-tower-graph", action="store_true",
                    help="显式把通过塔位坡度、档距、硬约束、建筑及逐跨 DEM 净空筛查的塔位图候选作为最终筛查方案")
    ap.add_argument("--tower-min-span-m", type=float, default=None,
                    help="实验塔位图最小水平档距；必须按具体线路工程参数提供")
    ap.add_argument("--tower-max-span-m", type=float, default=None,
                    help="实验塔位图最大水平档距；必须按具体线路工程参数提供")
    ap.add_argument("--tower-candidate-stride", type=int, default=2,
                    help="塔位候选像元步长；会影响可达性，默认2，仅为栅格采样参数")
    ap.add_argument("--tower-site-pad-radius-m", type=float, default=None,
                    help="显式塔位地表平台筛查半径；需与 experimental tower graph 同用；仅筛查已知DEM/硬掩膜/坡度")
    ap.add_argument("--tower-attachment-height-m", type=float, default=None,
                    help="可选净空筛查参数：固定挂点高度；需与弧垂和最小净空同时提供")
    ap.add_argument("--tower-max-midspan-sag-m", type=float, default=None,
                    help="可选净空筛查参数：固定最大跨中弧垂；不代表导线荷载计算")
    ap.add_argument("--tower-min-clearance-m", type=float, default=None,
                    help="可选净空筛查参数：固定最小地面净空；通过筛查不代表工程认证")
    ap.add_argument("--max-ground-grade", type=float, default=None,
                    help="可选地面相邻栅格最大坡度比，如0.30; 属于筛选阈值，不是线路设计限值")
    ap.add_argument("--max-surface-slope-deg", type=float, default=None,
                    help="可选候选塔位地表坡度筛查阈值(度); 不限制架空走廊跨越；默认不设工程阈值")
    ap.add_argument("--max-terrain-elevation-m", type=float, default=None,
                    help="可选地形海拔硬限制(米); 默认不设工程阈值")
    ap.add_argument("--use-costunet", action="store_true",
                    help="把CostUNet作为50%%软先验混入规则代价; 默认关闭，避免未知训练来源影响基线")
    ap.add_argument("--max-side", type=int, default=WORK_MAX_SIDE,
                    help=f"规划网格最长边；默认{WORK_MAX_SIDE}，尽量保留 DEM 原始分辨率，内存不足时可调低")
    ap.add_argument("--window-margin-deg", type=float, default=0.30,
                    help="起终点经纬度包围框外扩度数；扩大可检查候选路线是否受搜索窗口边界限制")
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--include-ai-diagnostics", action="store_true",
                    help="可选运行 VIN/PPO 作为诊断对照；AI 候选不会被选为最终路线")
    ap.add_argument("--no-ppp", action="store_true", help="在 --include-ai-diagnostics 下跳过 PPO")
    ap.add_argument("--no-vin", action="store_true", help="在 --include-ai-diagnostics 下跳过 VIN-Grad")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--with-real", action="store_true",
                    help="加载参考线路，仅在规划选路完成后计算 v2 双向空间相似度")
    ap.add_argument("--real-cases", action="store_true",
                    help="改用 v3 的 4 条真实走廊(R01~R04)并启用真实线对照")
    ap.add_argument("--real-cases-csv", type=str, default=None,
                    help="从真实走廊 CSV(列: name,start_lat,start_lon,end_lat,end_lon)载入测试用例；"
                         "默认在 CSV 同目录查找参考线，可用 --real-dir 指定其他目录。真实线只用于规划后的对照。")
    ap.add_argument("--real-dir", type=str, default=None,
                    help="逐任务参考线目录；每个匹配 SHP 必须恰含一个 LineString 资产")
    ap.add_argument("--real-shp", type=str, default=None,
                    help="单任务指定只含一个 LineString 且 CRS 明确的目标线路 SHP 作为事后对照; "
                         "多任务请用 --real-dir 按任务分别提供线路。规划阶段不会读取该线。")
    ap.add_argument("--osm-cache-dir", type=str, default=None,
                    help="OSM 环境图层缓存目录，内含 taiwan_buildings.pkl、taiwan_landuse.pkl 等文件")
    ap.add_argument("--rebuild-osm-cache", action="store_true",
                    help="调用 shared.data_acquisition 重新生成台湾 OSM 缓存 pkl(覆盖错误区域数据), "
                         "然后继续正常流程。首次使用或土地利用波段恒为0时请启用。")
    # PPO 强度(显存/耗时敏感, 默认中等; RTX 4060 8GB 可调低防 OOM)
    ap.add_argument("--ppo-updates", type=int, default=60, help="PPO 训练更新轮数(默认60)")
    ap.add_argument("--ppo-episodes", type=int, default=8, help="PPO 每轮采样的轨迹数(默认8)")
    ap.add_argument("--ppo-epochs", type=int, default=4, help="PPO 每轮策略优化 epoch(默认4)")
    ap.add_argument("--ppo-bc", type=int, default=60, help="PPO 行为克隆预热步数(默认60)")
    args = ap.parse_args()
    (args.search_turn_penalty_weight,
     args.smoothing_turn_penalty_weight) = resolve_turn_penalty_weights(
         args.turn_penalty_weight, args.search_turn_penalty_weight,
         args.smoothing_turn_penalty_weight)
    if bool(args.stations_csv) != bool(args.route_tasks_csv):
        ap.error("--stations-csv 和 --route-tasks-csv 必须同时提供")
    if args.select_experimental_tower_graph:
        if not args.experimental_tower_graph:
            ap.error("--select-experimental-tower-graph 必须同时启用 --experimental-tower-graph")
        if not args.stations_csv:
            ap.error("选择塔位图方案必须使用站点 CSV 任务")
        if not args.constraint_geojson:
            ap.error("选择塔位图方案必须提供含核验凭证的 --constraint-geojson 硬约束")
        if (args.max_surface_slope_deg is None
                or not 0 < args.max_surface_slope_deg <= 90):
            ap.error("选择塔位图方案必须提供有效的 --max-surface-slope-deg 塔位坡度上限")
        if args.tower_site_pad_radius_m is None:
            ap.error("选择塔位图方案必须显式提供 --tower-site-pad-radius-m")
        if any(value is None for value in (
                args.tower_attachment_height_m,
                args.tower_max_midspan_sag_m,
                args.tower_min_clearance_m)) or not (
                    args.dem_vertical_datum and args.dem_vertical_datum.strip()):
            ap.error("选择塔位图方案必须提供完整的塔位净空参数和 --dem-vertical-datum")
    if args.allow_unverified_endpoints and not args.stations_csv:
        ap.error("--allow-unverified-endpoints 仅适用于站点 CSV 任务")
    if bool(args.start) != bool(args.end):
        ap.error("--start 和 --end 必须同时提供")
    task_sources = sum((
        bool(args.stations_csv), bool(args.start or args.end),
        bool(args.real_cases_csv), bool(args.real_cases),
    ))
    if task_sources > 1:
        ap.error("任务输入来源只能选一种：站点 CSV、--start/--end、--real-cases-csv 或 --real-cases")
    if args.real_shp and (args.real_cases_csv or args.real_cases):
        ap.error("--real-shp 不能与 --real-cases-csv/--real-cases 同时使用")
    if args.real_shp and args.real_dir:
        ap.error("--real-shp 不能与 --real-dir 同时使用")
    if args.max_side < 1:
        ap.error("--max-side 必须是正整数")
    if args.window_margin_deg <= 0:
        ap.error("--window-margin-deg 必须大于0")
    float_parameters = {
        "--building-clearance-m": args.building_clearance_m,
        "--ridge-penalty-weight": args.ridge_penalty_weight,
        "--landcover-penalty-weight": args.landcover_penalty_weight,
        "--settlement-weight": args.settlement_weight,
        "--population-weight": args.population_weight,
        "--grade-penalty-weight": args.grade_penalty_weight,
        "--terrain-slope-weight": args.terrain_slope_weight,
        "--roughness-weight": args.roughness_weight,
        "--official-hazard-weight": args.official_hazard_weight,
        "--turn-penalty-weight": args.turn_penalty_weight,
        "--search-turn-penalty-weight": args.search_turn_penalty_weight,
        "--smoothing-turn-penalty-weight": args.smoothing_turn_penalty_weight,
        "--astar-heuristic-weight": args.astar_heuristic_weight,
        "--window-margin-deg": args.window_margin_deg,
        "--max-ground-grade": args.max_ground_grade,
        "--max-surface-slope-deg": args.max_surface_slope_deg,
        "--max-terrain-elevation-m": args.max_terrain_elevation_m,
        "--tower-min-span-m": args.tower_min_span_m,
        "--tower-max-span-m": args.tower_max_span_m,
        "--tower-site-pad-radius-m": args.tower_site_pad_radius_m,
        "--tower-attachment-height-m": args.tower_attachment_height_m,
        "--tower-max-midspan-sag-m": args.tower_max_midspan_sag_m,
        "--tower-min-clearance-m": args.tower_min_clearance_m,
    }
    for option, value in float_parameters.items():
        if value is not None and not math.isfinite(value):
            ap.error(f"{option} 必须是有限数值")
    if args.tower_site_pad_radius_m is not None:
        if args.tower_site_pad_radius_m <= 0:
            ap.error("--tower-site-pad-radius-m 必须大于0")
        if not args.experimental_tower_graph:
            ap.error("--tower-site-pad-radius-m 必须与 --experimental-tower-graph 同用")
    if (args.building_clearance_m < 0 or args.ridge_penalty_weight < 0
            or args.landcover_penalty_weight < 0
            or args.settlement_weight < 0
            or args.population_weight < 0
            or args.grade_penalty_weight < 0
            or args.terrain_slope_weight < 0
            or args.roughness_weight < 0
            or args.official_hazard_weight < 0
            or args.turn_penalty_weight < 0
            or args.search_turn_penalty_weight < 0
            or args.smoothing_turn_penalty_weight < 0):
        ap.error("避让距离与代价权重不能为负数")
    if args.astar_heuristic_weight < 1:
        ap.error("--astar-heuristic-weight 必须大于或等于 1.0")
    if (args.experimental_theta_star and args.experimental_theta_star_shared_objective):
        ap.error("--experimental-theta-star 与 --experimental-theta-star-shared-objective 不能同时指定")
    if ((args.experimental_astar_16_shared_objective
            or args.experimental_theta_star_shared_objective)
            and args.grade_penalty_weight != 0.0):
        ap.error("共享搜索空间对照要求 --grade-penalty-weight 0；地形坡度代价栅格和硬坡度限制仍生效")
    if ((args.experimental_astar_16_shared_objective
            or args.experimental_theta_star_shared_objective)
            and (args.search_turn_penalty_weight > 0
                 or args.smoothing_turn_penalty_weight > 0)):
        ap.error("共享目标实验暂不支持非零搜索或平滑转弯权重")
    if args.population_weight > 0 and not args.population_raster:
        ap.error("启用 --population-weight 时必须同时指定 --population-raster")
    if args.population_raster and not Path(args.population_raster).is_file():
        ap.error(f"--population-raster 文件不存在: {args.population_raster}")
    if args.population_weight > 0 and not HAS_RASTERIO:
        ap.error("启用人口软成本需要 rasterio")
    if args.building_clearance_m >= 5000:
        ap.error("--building-clearance-m 必须小于5000米（当前距离栅格上限）")
    if args.max_ground_grade is not None and args.max_ground_grade <= 0:
        ap.error("--max-ground-grade 必须大于0")
    tower_profile = (args.tower_attachment_height_m,
                     args.tower_max_midspan_sag_m,
                     args.tower_min_clearance_m)
    if args.experimental_tower_graph:
        if args.tower_min_span_m is None or args.tower_max_span_m is None:
            ap.error("--experimental-tower-graph 必须显式提供 --tower-min-span-m 和 --tower-max-span-m")
        if (args.tower_min_span_m < 0
                or args.tower_max_span_m <= args.tower_min_span_m):
            ap.error("塔位图档距必须满足 0 <= 最小档距 < 最大档距")
        if args.tower_candidate_stride < 1:
            ap.error("--tower-candidate-stride 必须是正整数")
        if any(value is not None for value in tower_profile) and any(
                value is None for value in tower_profile):
            ap.error("净空筛查必须同时提供挂点高度、最大跨中弧垂和最小净空")
    elif (args.tower_min_span_m is not None or args.tower_max_span_m is not None
          or any(value is not None for value in tower_profile)):
        ap.error("塔位图参数必须与 --experimental-tower-graph 同时使用")

    if args.rebuild_osm_cache:
        print("[OSM] 请求重建台湾 OSM 缓存 ...")
        try:
            from . import rebuild_osm_cache as _roc
            _roc.rebuild()
        except Exception as _e:
            print(f"[warn] OSM 缓存重建失败: {_e}")
        print("[OSM] 缓存重建完成, 继续后续流程。")

    OUT = Path(args.out_dir) if args.out_dir else OUTPUTS_DIR
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print("  复杂山区输电线路启发式路径规划 V5.0")
    print("=" * 70)

    dem_path = Path(args.dem) if args.dem else find_dem()
    if dem_path is None:
        sys.exit("[FATAL] 未找到 DEM，请用 --dem 指定。")
    print(f"[DEM] {dem_path}")

    station_task_by_route = {}
    if args.stations_csv:
        try:
            station_tasks = load_station_tasks(
                args.stations_csv, args.route_tasks_csv,
                allow_unverified=args.allow_unverified_endpoints)
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        if (args.select_experimental_tower_graph
                and any(task["endpoint_qualification"]
                        != "declared_verified_line_connection_points"
                        for task in station_tasks)):
            ap.error("选择塔位图方案要求所有端点声明为已核验线路接入点")
        cases = []
        for task in station_tasks:
            cases.append((task["route_id"], task["start"], task["end"]))
            station_task_by_route[task["route_id"]] = task
    elif args.start and args.end:
        try:
            start = parse_lat_lon(args.start, "--start")
            end = parse_lat_lon(args.end, "--end")
        except ValueError as exc:
            ap.error(str(exc))
        cases = [("Custom", start, end)]
    else:
        if args.real_cases_csv:
            if not Path(args.real_cases_csv).is_file():
                ap.error(f"--real-cases-csv 文件不存在: {args.real_cases_csv}")
            try:
                csv_cases = load_real_cases_csv(args.real_cases_csv)
            except (OSError, ValueError) as exc:
                ap.error(str(exc))
            # 默认从 CSV 同目录找参考线；显式 --real-dir 时使用指定目录。
            args.real_dir = args.real_dir or str(Path(args.real_cases_csv).parent)
            cases = [(n, s, e) for (n, s, e, _) in csv_cases]
        elif args.real_cases:
            cases = [(n, s, e) for (n, s, e, _) in REAL_CASES]
        else:
            cases = DEFAULT_CASES[:max(1, args.cases)]

    allow_global_reference_mapping = not any((
        args.real_dir, args.stations_csv, args.start, args.real_cases_csv))
    args.real_dir = args.real_dir or str(REAL_PATH_DIR)
    try:
        validate_cases_available(cases)
        validate_unique_case_names(
            cases,
            check_reference_names=not args.real_shp and bool(
                args.real_cases_csv or args.real_cases or args.with_real),
        )
        validate_real_shp_argument(args.real_shp, cases)
    except ValueError as exc:
        ap.error(str(exc))

    global_real_exists = args.real_shp is not None
    use_real = bool(args.with_real or args.real_cases or args.real_cases_csv or global_real_exists)
    if use_real:
        mode = "单任务目标线路" if global_real_exists else args.real_dir
        print(f"[真实对比] 已启用: 模式={mode}；真实线仅用于规划后的事后评估")

    run_provenance = capture_run_provenance(
        Path(__file__).resolve().parents[1], args, cases, dem_path,
        DOWNLOADED_DIR, model_weights=MODEL_WEIGHTS, has_torch=HAS_TORCH,
        effective_runtime={
            "random_seed": RANDOM_SEED,
            "has_torch": HAS_TORCH,
            "has_scipy": HAS_SCIPY,
            "has_rasterio": HAS_RASTERIO,
            "has_geopandas": HAS_GPD,
        })
    OUT.mkdir(parents=True, exist_ok=True)

    # CostUNet（全局加载一次）
    costunet = load_costunet() if args.use_costunet and HAS_TORCH else None
    if args.use_costunet and costunet is None:
        print("[warn] CostUNet不可用，继续使用规则代价面")

    all_results = []
    main_done = False

    failed_cases = []
    for ci, (cname, (slat, slon), (elat, elon)) in enumerate(cases):
        print(f"\n--- 用例 {ci + 1}/{len(cases)}: {cname} ---")
        try:
            margin = args.window_margin_deg
            minlon, maxlon = min(slon, elon) - margin, max(slon, elon) + margin
            minlat, maxlat = min(slat, elat) - margin, max(slat, elat) + margin
            dem, transform, crs, pm = load_dem_window(dem_path, minlon, minlat, maxlon, maxlat, args.max_side)
            H, W = dem.shape
            population_proxy = None
            population_valid_mask = None
            population_metadata = {
                "status": "disabled_weight_zero",
                "source_path": (str(Path(args.population_raster).resolve())
                                if args.population_raster else None),
                "weight": args.population_weight,
                "normalization_scale_people_per_km2": (
                    POPULATION_DENSITY_NORMALIZATION_PEOPLE_PER_KM2),
            }
            if args.population_weight > 0:
                dem_valid_mask = np.isfinite(dem)
                population_density, population_metadata = load_population_density_window(
                    args.population_raster, dem.shape, transform, crs,
                    required_mask=dem_valid_mask,
                )
                population_proxy, population_unknown_mask = population_cost_for_routing(
                    population_density, nodata_policy=args.population_nodata_policy,
                    valid_mask=dem_valid_mask)
                population_valid_mask = ~population_unknown_mask
                population_metadata["weight"] = args.population_weight
                population_metadata["source_value_semantics"] = (
                    "modelled people per source cell; not point locations")
                population_metadata["cost_semantics"] = (
                    "log1p(population_density_people_per_km2) / "
                    "log1p(normalization_scale), clipped to [0, 1]")
                population_metadata["nodata_cost_policy"] = {
                    "maximum": "fixed maximum normalized proxy penalty (1.0); not a population estimate",
                    "median": "median normalized proxy among valid population cells on valid DEM pixels in this planning window; imputation only",
                    "zero": "zero-cost fill sensitivity scenario; not an empty-population estimate",
                }[args.population_nodata_policy]
                nodata_policy_label = {
                    "maximum": "最大归一化惩罚",
                    "median": "DEM有效区域中位数填补（敏感性分析）",
                    "zero": "零代价填补（敏感性分析）",
                }[args.population_nodata_policy]
                print(
                    "  人口软成本已对齐："
                    f"coverage={population_metadata['coverage_ratio']:.3f}, "
                    f"scale={population_metadata['normalization_scale_people_per_km2']:.0f} people/km², "
                    f"NoData策略={nodata_policy_label}"
                )
            if args.search_turn_penalty_weight > 0:
                state_gib = H * W * (8 * 8 + 8) / (1024 ** 3)
                if state_gib >= 0.5:
                    print(f"  [warn] 转弯状态数组约需 {state_gib:.2f} GiB；还未包含其他栅格和搜索队列")

            sr, sc = geo_to_rowcol(slat, slon, transform)
            er, ec = geo_to_rowcol(elat, elon, transform)
            station_task = station_task_by_route.get(cname)
            shortcut_endpoints = (
                (float(station_task["from_station"]["lon"]),
                 float(station_task["from_station"]["lat"])),
                (float(station_task["to_station"]["lon"]),
                 float(station_task["to_station"]["lat"])),
            ) if station_task else ((float(slon), float(slat)), (float(elon), float(elat)))

            left = transform.c; top = transform.f
            right = left + transform.a * W; bottom = top + transform.e * H
            bbox = (left, bottom, right, top)
            needs_feature_stack = costunet is not None or (
                args.include_ai_diagnostics and HAS_TORCH and not args.no_vin)
            osm = compute_osm_bands(
                dem, transform, bbox, cache_dir=args.osm_cache_dir,
                include_model_features=needs_feature_stack,
                include_building_distance=args.building_clearance_m > 0,
            )
            osm_layer_metadata = getattr(osm, "metadata", {}).get("layers", {})
            reject_osm_planning_layer_read_errors(osm_layer_metadata)
            terrain = compute_terrain_factors(dem, pm, transform, crs)
            official_hazard_masks, official_hazard_metadata = load_official_hazard_masks(
                args.official_hazards_dir, bbox, transform, dem.shape)
            official_hazard_metadata["soft_weight_applied_in_run"] = args.official_hazard_weight
            if official_hazard_metadata.get("coverage_status") != "complete":
                print("  [warn] 官方灾害图层覆盖不完整；未知瓦片不按无风险处理。"
                      f" status={official_hazard_metadata.get('status')}, "
                      f"coverage={official_hazard_metadata.get('coverage_status')}")
            custom_constraints = (
                load_user_constraint_overlays(
                    args.constraint_geojson, transform, dem.shape,
                    require_verified_hard_constraints=args.select_experimental_tower_graph)
                if args.constraint_geojson else None
            )
            custom_constraint_metadata = custom_constraints["metadata"] if custom_constraints else None
            land_mask_layer = (load_land_mask_geojson(
                args.land_mask_geojson, transform, dem.shape)
                if args.land_mask_geojson else None)
            land_mask_metadata = land_mask_layer["metadata"] if land_mask_layer else None
            for layer_name in ("buildings", "landuse"):
                layer = osm_layer_metadata.get(layer_name, {})
                if layer.get("status") != "available":
                    print(f"  [warn] OSM {layer_name} 图层状态={layer.get('status', 'unknown')}；"
                          "不能据此认定该范围内没有居民点/建筑")
            stack = build_feature_stack(dem, terrain, osm) if needs_feature_stack else None

            model_cost = costunet.predict(stack) if costunet is not None else None
            cost = build_route_cost(
                terrain, osm, args.settlement_weight, model_cost,
                ridge_penalty_weight=args.ridge_penalty_weight,
                landcover_penalty_weight=args.landcover_penalty_weight,
                official_hazard_masks=official_hazard_masks,
                official_hazard_weight=args.official_hazard_weight,
                terrain_slope_weight=args.terrain_slope_weight,
                roughness_weight=args.roughness_weight,
                population_proxy=population_proxy,
                population_weight=args.population_weight,
            )

            real_geo = None
            if custom_constraints:
                soft_cost = custom_constraints["soft_cost"]
                cost = (cost + soft_cost).astype(np.float32)
            print(f"  代价面范围 [{np.nanmin(cost):.3f}, {np.nanmax(cost):.3f}]")

            slope = terrain["slope"]; elev = terrain["elevation"]
            building_mask = osm.get("building_mask")
            combined_hard_mask = (
                np.asarray(custom_constraints["hard_mask"], dtype=np.uint8).copy()
                if custom_constraints else np.zeros(dem.shape, dtype=np.uint8))
            if land_mask_layer:
                combined_hard_mask = np.maximum(
                    combined_hard_mask,
                    (land_mask_layer["land_mask"] == 0).astype(np.uint8))
            if args.max_surface_slope_deg is not None:
                if not 0 < args.max_surface_slope_deg <= 90:
                    raise ValueError("--max-surface-slope-deg 必须在(0,90]内")
            if args.tower_site_pad_radius_m is not None:
                if (not math.isclose(transform.b, 0.0, rel_tol=0.0, abs_tol=1e-12)
                        or not math.isclose(transform.d, 0.0, rel_tol=0.0, abs_tol=1e-12)):
                    raise ValueError("塔位平台筛查要求北向上、无旋转或错切的 DEM")
                lat_edges = (transform.f, transform.f + transform.e * H)
                edge_spacings = [pixel_spacing_meters(transform, lat, crs)
                                 for lat in lat_edges]
                route_pixel_spacing = tuple(
                    min(spacing[axis] for spacing in edge_spacings) for axis in range(2))
            else:
                lat_center = transform.f + transform.e * (H / 2.0)
                route_pixel_spacing = pixel_spacing_meters(transform, lat_center, crs)
            tower_span_mask, tower_site_mask = build_tower_masks(
                dem, slope,
                custom_hard_mask=combined_hard_mask,
                max_surface_slope_deg=args.max_surface_slope_deg,
                max_terrain_elevation_m=args.max_terrain_elevation_m,
                building_mask=building_mask,
                building_distance_m=osm.get("building_distance_m"),
                building_clearance_m=args.building_clearance_m,
                tower_site_pad_radius_m=args.tower_site_pad_radius_m,
                pixel_spacing_m=route_pixel_spacing,
            )
            endpoint_blockers = {"DEM NoData": ~np.isfinite(dem)}
            if custom_constraints:
                endpoint_blockers["用户硬约束"] = custom_constraints["hard_mask"] > 0
            if land_mask_layer:
                endpoint_blockers["陆域范围外"] = land_mask_layer["land_mask"] == 0
            if args.max_surface_slope_deg is not None:
                endpoint_blockers["地表坡度阈值"] = slope > args.max_surface_slope_deg
            if args.max_terrain_elevation_m is not None:
                endpoint_blockers["地形海拔阈值"] = elev > args.max_terrain_elevation_m
            if building_mask is not None:
                endpoint_blockers["OSM建筑足迹"] = np.asarray(building_mask) > 0
            if args.building_clearance_m > 0 and "building_distance_m" in osm:
                endpoint_blockers["建筑净距阈值"] = (
                    osm["building_distance_m"] <= args.building_clearance_m)
            building_geometries = getattr(osm, "geometries", {}).get("buildings", [])
            endpoint_building_exceptions = open_safe_building_endpoint_pixels(
                (("起点", (sr, sc), (slon, slat)),
                 ("终点", (er, ec), (elon, elat))),
                tower_site_mask, endpoint_blockers, building_geometries, transform,
                args.building_clearance_m,
            )
            for exception in endpoint_building_exceptions:
                tower_span_mask[exception["row"], exception["col"]] = 1
            endpoint_mask = (tower_span_mask if args.experimental_tower_graph
                             else tower_site_mask)
            validate_endpoint_cells((sr, sc), (er, ec), dem.shape, endpoint_mask,
                                    endpoint_blockers)
            same_grid_cell = (sr, sc) == (er, ec)
            cost_grid = (1.0 + 4.0 * cost).astype(np.float64)
            cost_grid = np.where(tower_span_mask > 0, cost_grid, np.inf)
            tower_span_cost_grid = (1.0 + 4.0 * cost).astype(np.float64)
            tower_span_cost_grid = np.where(
                tower_span_mask > 0, tower_span_cost_grid, np.inf)
            if args.include_ai_diagnostics:
                cost_finite = np.where(np.isfinite(cost), cost, 1.0).astype(np.float32)
                cost_vin = np.where(tower_span_mask > 0, cost_grid, 0.0).astype(np.float32)
            methods = {}
            grade_reports = {}
            V_main = None
            t_vin = 0.0
            t_ppo = 0.0
            t_theta = 0.0
            t_tower_graph = 0.0
            # AI methods are opt-in diagnostics; route selection below excludes them.
            if args.include_ai_diagnostics and HAS_TORCH and not args.no_vin and not same_grid_cell:
                t_v = time.time()
                extra = build_vin_extra(stack)
                goal_mask = np.zeros((H, W), np.float32); goal_mask[er, ec] = 1.0
                torch.manual_seed(RANDOM_SEED)
                V = train_vin_or_skip_on_oom(cost_vin, goal_mask, extra, tower_span_mask,
                                             route_pixel_spacing, n_steps=80)
                if V is None:
                    print("  [VIN-Grad] 本用例未生成 AI 候选")
                else:
                    V_main = np.nan_to_num(V, nan=1e6, posinf=1e6, neginf=1e6)
                    geo_vin = smooth_path_with_constraints(gradient_track_path(
                        V, cost_finite, tower_span_mask, (sr, sc), (er, ec), transform),
                        transform, tower_span_mask, cost_grid, dem,
                        0.0, pixel_spacing=route_pixel_spacing,
                        max_ground_grade=args.max_ground_grade, crs=crs,
                        shortcut_endpoints=shortcut_endpoints)
                    geo_vin, grade_reports["VIN-Grad"] = filter_path_by_ground_grade(
                        geo_vin, dem, transform, args.max_ground_grade, crs)
                    if geo_vin:
                        grade_reports["VIN-Grad"] = ground_grade_limit_report(
                            geo_vin, dem, transform, args.max_ground_grade, crs)
                    else:
                        grade_reports["VIN-Grad"] = ground_grade_limit_report(
                            None, dem, transform, args.max_ground_grade, crs)
                    # 保存真实 V/cost/hard/起终点, 便于 PPO 推理调试(无需重跑训练即可秒级重测)
                    try:
                        np.savez(str(OUT / "_last_V.npz"), V=V_main, cost=cost_finite,
                                 hard=tower_span_mask, tower_sites=tower_site_mask,
                                 start=np.array([sr, sc]), goal=np.array([er, ec]))
                    except Exception:
                        pass
                    methods["VIN-Grad"] = geo_vin
                    t_vin = time.time() - t_v
                    print(f"  [VIN-Grad] 路径 {len(geo_vin) if geo_vin else 0} 点, 耗时 {t_vin:.1f}s")
            else:
                reason = ("未启用 AI 诊断" if not args.include_ai_diagnostics else
                          "起终点位于同一栅格" if same_grid_cell else
                          "已禁用或 torch 不可用")
                print(f"  [skip] VIN-Grad {reason}")

            # --- 方法2: RL/PPO 策略梯度路径生成器 (每用例独立训练: 代价图与目标不同) ---
            if args.include_ai_diagnostics and HAS_TORCH and not args.no_ppp and not same_grid_cell:
                def build_ppo_candidate():
                    torch.manual_seed(RANDOM_SEED + 100)
                    print("  [PPO] 训练策略网络 ...")
                    t_p = time.time()
                    ppo_policy, _ = train_ppo(
                        cost_finite, tower_span_mask, (er, ec), (sr, sc), value_map=V_main,
                        n_updates=args.ppo_updates, episodes=args.ppo_episodes,
                        ppo_epochs=args.ppo_epochs, bc_steps=args.ppo_bc)
                    print(f"  [PPO] 训练完成 {time.time() - t_p:.1f}s")
                    ctx = build_ppo_context(np.clip(cost_finite, 0, 1), (er, ec), H, W, value_map=V_main)
                    ppo_path = ppo_rollout_path(ppo_policy, ctx, cost_finite, tower_span_mask, (er, ec), (sr, sc),
                                               n_tries=6, guide_v=V_main, guide_w=6.0, v_tol=0.35)
                    geo_ppo = smooth_path_with_constraints(
                        [rowcol_to_geo(r, c, transform) for r, c in ppo_path],
                        transform, tower_span_mask, cost_grid, dem,
                        0.0, pixel_spacing=route_pixel_spacing,
                        max_ground_grade=args.max_ground_grade, crs=crs,
                        shortcut_endpoints=shortcut_endpoints)
                    geo_ppo, grade_reports["PPO"] = filter_path_by_ground_grade(
                        geo_ppo, dem, transform, args.max_ground_grade, crs)
                    if geo_ppo:
                        grade_reports["PPO"] = ground_grade_limit_report(
                            geo_ppo, dem, transform, args.max_ground_grade, crs)
                    else:
                        grade_reports["PPO"] = ground_grade_limit_report(
                            None, dem, transform, args.max_ground_grade, crs)
                    nonlocal t_ppo
                    t_ppo = time.time() - t_p
                    return geo_ppo

                geo_ppo, ppo_oom = run_method_or_skip_on_oom("PPO", build_ppo_candidate)
                if ppo_oom is None:
                    methods["PPO"] = geo_ppo
                    print(f"  [PPO] 路径 {len(geo_ppo) if geo_ppo else 0} 点")
            else:
                reason = ("未启用 AI 诊断" if not args.include_ai_diagnostics else
                          "起终点位于同一栅格" if same_grid_cell else
                          "已禁用或 torch 不可用")
                print(f"  [skip] PPO {reason}")

            # Weight 1 is optimal A*. Larger weights are approximate; reopened
            # states do not provide a general optimality bound.
            astar_method = (
                "A*(turn-aware)" if args.search_turn_penalty_weight > 0 else
                "A*(baseline)" if args.astar_heuristic_weight == 1.0 else
                "A*(weighted)"
            )
            t_astar = time.time()
            a_path = astar_search(
                cost_grid, (sr, sc), (er, ec), elevation=dem, pixel_m=pm,
                grade_penalty_weight=args.grade_penalty_weight,
                max_ground_grade=args.max_ground_grade,
                pixel_spacing=route_pixel_spacing,
                heuristic_weight=args.astar_heuristic_weight,
                turn_penalty_weight=args.search_turn_penalty_weight,
            )
            raw_astar_geo = ([rowcol_to_geo(r, c, transform) for r, c in a_path]
                             if a_path else None)
            geo_astar = [rowcol_to_geo(r, c, transform) for r, c in a_path] if a_path else None
            if geo_astar:
                geo_astar = smooth_path_with_constraints(
                    geo_astar, transform, tower_span_mask, cost_grid, dem,
                    args.grade_penalty_weight, args.smoothing_turn_penalty_weight,
                    route_pixel_spacing, max_ground_grade=args.max_ground_grade,
                    crs=crs, shortcut_endpoints=shortcut_endpoints)
            geo_astar, grade_reports[astar_method] = filter_path_by_ground_grade(
                geo_astar, dem, transform, args.max_ground_grade, crs)
            methods[astar_method] = geo_astar
            t_astar = time.time() - t_astar
            print(f"  [{astar_method}] 路径 {len(geo_astar) if geo_astar else 0} 点, "
                  f"路径生成总耗时 {t_astar:.1f}s")

            sampled_pareto_by_method = {}
            if args.experimental_sampled_pareto:
                started = time.time()
                sampled_candidates = sampled_pareto_astar_routes(
                    cost_grid, tower_span_mask > 0, (sr, sc), (er, ec),
                    pixel_spacing=route_pixel_spacing, elevation=dem, pixel_m=pm,
                    grade_penalty_weight=args.grade_penalty_weight,
                    max_ground_grade=args.max_ground_grade,
                    turn_penalty_weight=args.search_turn_penalty_weight,
                )
                sample_elapsed = time.time() - started
                for index, candidate in enumerate(sampled_candidates, start=1):
                    method = f"Sampled Pareto {index:02d}"
                    geo = [rowcol_to_geo(row, col, transform)
                           for row, col in candidate["path"]]
                    if geo:
                        geo = smooth_path_with_constraints(
                            geo, transform, tower_span_mask, cost_grid, dem,
                            args.grade_penalty_weight, args.smoothing_turn_penalty_weight,
                            route_pixel_spacing,
                            max_ground_grade=args.max_ground_grade, crs=crs,
                            shortcut_endpoints=shortcut_endpoints,
                        )
                    geo, grade_reports[method] = filter_path_by_ground_grade(
                        geo, dem, transform, args.max_ground_grade, crs)
                    methods[method] = geo
                    sampled_pareto_by_method[method] = {
                        "sampled_environmental_weights": candidate[
                            "sampled_environmental_weights"],
                        "objective_vector_on_search_grid": candidate["objective_vector"],
                        "pareto_scope": candidate["pareto_scope"],
                        "search_hard_constraint_legal": candidate[
                            "hard_constraint_legal"],
                        "search_hard_violation_count": candidate[
                            "hard_violation_count"],
                        "candidate_search_batch_time_s": sample_elapsed,
                    }
                    print(f"  [{method}] 网格候选 {len(candidate['path'])} 点；"
                          "之后仍需端点、硬约束和建筑复核")

            experimental_neighbor_results = {}
            experimental_theta_results = {}
            neighbor_experiments = []
            if args.experimental_astar_16:
                neighbor_experiments.extend((
                    (ASTAR_8_EXACT_METHOD, NEIGHBORS_8, 0.0),
                    (ASTAR_16_METHOD, NEIGHBORS_16, 0.0),
                ))
            if args.experimental_astar_16_shared_objective:
                neighbor_experiments.extend((
                    (ASTAR_8_SHARED_METHOD, NEIGHBORS_8,
                     args.grade_penalty_weight),
                    (ASTAR_16_SHARED_METHOD, NEIGHBORS_16,
                     args.grade_penalty_weight),
                ))
            for method, neighbors, grade_weight in neighbor_experiments:
                started = time.time()
                raw_path = astar_exact_integral_path(
                    cost_grid, tower_span_mask, (sr, sc), (er, ec), neighbors,
                    pixel_spacing=route_pixel_spacing, elevation=dem,
                    max_ground_grade=args.max_ground_grade,
                    grade_penalty_weight=grade_weight,
                )
                geo_path = ([rowcol_to_geo(r, c, transform) for r, c in raw_path]
                            if raw_path else None)
                if geo_path:
                    geo_path = smooth_path_with_constraints(
                        geo_path, transform, tower_span_mask, cost_grid, dem,
                        grade_weight, 0.0, route_pixel_spacing,
                        max_ground_grade=args.max_ground_grade, crs=crs,
                        shortcut_endpoints=shortcut_endpoints,
                    )
                geo_path, grade_reports[method] = filter_path_by_ground_grade(
                    geo_path, dem, transform, args.max_ground_grade, crs)
                elapsed = time.time() - started
                search_raster_cost = integrate_path_cost(
                    raw_path, cost_grid, tower_span_mask,
                    pixel_spacing=route_pixel_spacing,
                    neighbor_offsets=neighbors,
                ) if raw_path else math.inf
                search_cost = (integrate_path_cost(
                    raw_path, cost_grid, tower_span_mask,
                    pixel_spacing=route_pixel_spacing,
                    neighbor_offsets=neighbors, elevation=dem,
                    grade_penalty_weight=grade_weight,
                ) if raw_path else math.inf)
                search_grade_penalty_cost = (
                    search_cost - search_raster_cost
                    if math.isfinite(search_cost) and math.isfinite(search_raster_cost)
                    else math.inf)
                methods[method] = geo_path
                experimental_neighbor_results[method] = {
                    "path": raw_path,
                    "neighbors": neighbors,
                    "search_cost": search_cost,
                    "search_raster_cost": search_raster_cost,
                    "search_grade_penalty_cost": search_grade_penalty_cost,
                    "grade_penalty_weight": grade_weight,
                    "objective_scope": (
                        "piecewise_constant_raster_cost_plus_per_search_edge_endpoint_grade"
                        if grade_weight > 0.0 else "piecewise_constant_raster_cost_only"),
                    "time_s": elapsed,
                }
                print(f"  [{method}] 路径 {len(geo_path) if geo_path else 0} 点, "
                      f"路径生成总耗时 {elapsed:.1f}s；精确积分，"
                      f"软地面坡度权重={grade_weight:g}，转弯权重=0")

            if (args.experimental_theta_star
                    or args.experimental_theta_star_shared_objective):
                theta_shared_objective = args.experimental_theta_star_shared_objective
                theta_method = (THETA_SHARED_METHOD if theta_shared_objective
                                else "Theta*(experimental)")
                theta_grade_weight = (args.grade_penalty_weight
                                      if theta_shared_objective else 0.0)
                t_theta = time.time()
                theta_path = theta_star_cost_path(
                    cost_grid, tower_span_mask, (sr, sc), (er, ec),
                    pixel_spacing=route_pixel_spacing, elevation=dem,
                    max_ground_grade=args.max_ground_grade,
                    grade_penalty_weight=theta_grade_weight)
                geo_theta = ([rowcol_to_geo(r, c, transform) for r, c in theta_path]
                             if theta_path else None)
                if geo_theta:
                    geo_theta = smooth_path_with_constraints(
                        geo_theta, transform, tower_span_mask, cost_grid, dem,
                        theta_grade_weight, 0.0, route_pixel_spacing,
                        max_ground_grade=args.max_ground_grade, crs=crs,
                        shortcut_endpoints=shortcut_endpoints)
                grade_reports[theta_method] = ground_grade_limit_report(
                    geo_theta, dem, transform, args.max_ground_grade, crs)
                t_theta = time.time() - t_theta
                methods[theta_method] = geo_theta
                print(f"  [{theta_method}] 路径 {len(geo_theta) if geo_theta else 0} 点, "
                      f"耗时 {t_theta:.1f}s；仅作对照候选")
                if theta_shared_objective:
                    theta_raster_cost = integrate_polyline_cost(
                        theta_path or [], cost_grid, tower_span_mask,
                        pixel_spacing=route_pixel_spacing)
                    theta_search_cost = integrate_polyline_cost(
                        theta_path or [], cost_grid, tower_span_mask,
                        pixel_spacing=route_pixel_spacing, elevation=dem,
                        grade_penalty_weight=theta_grade_weight)
                    theta_objective_scope = (
                        "piecewise_constant_raster_cost_plus_per_search_edge_endpoint_grade"
                        if theta_grade_weight > 0.0
                        else "piecewise_constant_raster_cost_only")
                    experimental_theta_results[theta_method] = {
                        "path": theta_path,
                        "search_cost": theta_search_cost,
                        "search_raster_cost": theta_raster_cost,
                        "grade_penalty_weight": theta_grade_weight,
                        "objective_scope": theta_objective_scope,
                        "time_s": t_theta,
                    }

            experimental_tower_graph_results = None
            if args.experimental_tower_graph:
                started = time.time()
                tower_profile_args = {}
                clearance_profile_status = "not_requested"
                clearance_profile_ready = True
                if all(value is not None for value in tower_profile):
                    endpoint_elevation_overrides = None
                    if station_task:
                        endpoint_elevation_overrides, elevation_status = (
                            station_clearance_elevation_overrides(
                                station_task, args.dem_vertical_datum))
                    else:
                        elevation_status = "dem_only"
                    if elevation_status in (
                            "dem_only",
                            "station_and_dem_vertical_datums_declared_matching"):
                        tower_profile_args = {
                            "dem_raster": dem,
                            "attachment_height_m": args.tower_attachment_height_m,
                            "max_midspan_sag_m": args.tower_max_midspan_sag_m,
                            "minimum_clearance_m": args.tower_min_clearance_m,
                            **(endpoint_elevation_overrides or {}),
                        }
                        clearance_profile_status = (
                            "dem_only_profile_screening" if elevation_status == "dem_only"
                            else "station_elevation_anchored_profile_screening")
                    else:
                        clearance_profile_status = f"unknown_{elevation_status}"
                        clearance_profile_ready = False
                start_col_corner, start_row_corner = ~transform * (slon, slat)
                goal_col_corner, goal_row_corner = ~transform * (elon, elat)
                start_tower_position_rc = (start_row_corner - 0.5,
                                           start_col_corner - 0.5)
                goal_tower_position_rc = (goal_row_corner - 0.5,
                                          goal_col_corner - 0.5)
                tower_path = None
                if clearance_profile_ready:
                    tower_path = tower_graph_route(
                        tower_span_cost_grid, tower_span_mask.astype(bool), tower_site_mask.astype(bool),
                        (sr, sc), (er, ec), args.tower_min_span_m,
                        args.tower_max_span_m, pixel_spacing=route_pixel_spacing,
                        candidate_stride=args.tower_candidate_stride,
                        turn_penalty_weight=args.search_turn_penalty_weight,
                        start_position_rc=start_tower_position_rc,
                        goal_position_rc=goal_tower_position_rc,
                        **tower_profile_args,
                    )
                tower_site_hazard_exposure = tower_site_hazard_exposure_diagnostic(
                    tower_path, official_hazard_masks, official_hazard_metadata,
                    route_pixel_spacing, args.tower_site_pad_radius_m,
                    grid_shape=dem.shape)
                tower_geo = ([rowcol_to_geo(row, col, transform)
                              for row, col in tower_path] if tower_path else None)
                span_clearance_profiles = []
                if tower_path and tower_profile_args:
                    for span_index, (first, second) in enumerate(
                            zip(tower_path, tower_path[1:])):
                        overrides = {}
                        for terminal, elevation_key in (
                                (start_tower_position_rc, "start_ground_elevation_m"),
                                (goal_tower_position_rc, "goal_ground_elevation_m")):
                            elevation = tower_profile_args.get(elevation_key)
                            if elevation is not None:
                                if first == terminal:
                                    overrides["start_ground_elevation_m"] = elevation
                                elif second == terminal:
                                    overrides["end_ground_elevation_m"] = elevation
                        span_clearance_profiles.append({
                            "span_index": span_index,
                            "start_rc": list(first),
                            "end_rc": list(second),
                            **span_clearance_status(
                                first, second, dem, route_pixel_spacing,
                                args.tower_attachment_height_m,
                                args.tower_max_midspan_sag_m,
                                args.tower_min_clearance_m,
                                **overrides),
                        })
                grade_reports[TOWER_GRAPH_METHOD] = overhead_span_grade_report()
                methods[TOWER_GRAPH_METHOD] = tower_geo
                t_tower_graph = time.time() - started
                experimental_tower_graph_results = {
                    "path": tower_path,
                    "time_s": t_tower_graph,
                    "span_min_m": args.tower_min_span_m,
                    "span_max_m": args.tower_max_span_m,
                    "candidate_stride": args.tower_candidate_stride,
                    "turn_penalty_weight": args.search_turn_penalty_weight,
                    "tower_site_max_surface_slope_deg": args.max_surface_slope_deg,
                    "tower_site_slope_status": (
                        "interior_candidates_screened" if args.max_surface_slope_deg is not None
                        else "not_evaluated"),
                    "tower_site_pad_radius_m": args.tower_site_pad_radius_m,
                    "tower_site_pad_status": (
                        "site_pad_footprint_screened"
                        if args.tower_site_pad_radius_m is not None else "not_configured"),
                    "tower_site_slope_scope": (
                        "intermediate_tower_site_pad_footprint_cells"
                        if args.tower_site_pad_radius_m is not None
                        else "intermediate_tower_candidate_pixels_only"),
                    "selected_intermediate_tower_site_count": (
                        max(0, len(tower_path) - 2) if tower_path else 0),
                    "tower_site_hazard_exposure": tower_site_hazard_exposure,
                    "station_terminal_slope_status": "not_evaluated_existing_station_endpoints",
                    "clearance_profile_screening": bool(tower_profile_args),
                    "clearance_profile_status": clearance_profile_status,
                    "clearance_profile_spans": span_clearance_profiles,
                    "dem_vertical_datum": args.dem_vertical_datum,
                    "site_cost_model": "zero; no validated tower-construction cost supplied",
                }
                print(f"  [{TOWER_GRAPH_METHOD}] 路径 {len(tower_geo) if tower_geo else 0} 点, "
                      f"耗时 {t_tower_graph:.1f}s；不参与最终选路")

            station_offsets_by_method = {}
            station_measurements = {}
            hard_violation_reports = {}
            tower_span_reports = {}
            if station_task:
                for method, geo in list(methods.items()):
                    if not geo:
                        continue
                    geo, offsets = attach_station_endpoints(geo, station_task)
                    if method == TOWER_GRAPH_METHOD:
                        geo, tower_span_reports[method] = validate_tower_span_lengths(
                            geo, args.tower_min_span_m, args.tower_max_span_m)
                        grade_report = grade_reports[method]
                    else:
                        geo, grade_report = filter_path_by_ground_grade(
                            geo, dem, transform, args.max_ground_grade, crs)
                    validation_mask = tower_span_mask
                    geo, hard_report = validate_route_hard_constraints(
                        geo, transform, validation_mask)
                    hard_report.update(tower_span_reports.get(method, {}))
                    if geo and osm_layer_metadata.get("buildings", {}).get("status") == "available":
                        geo, building_report = validate_route_against_buildings(
                            geo, building_geometries)
                        hard_report.update(building_report)
                        if building_report["building_geometry_status"] == "fail":
                            hard_report["hard_constraint_status"] = "fail"
                            hard_report["hard_violations"] += 1
                    annotate_route_validation_context(
                        hard_report, endpoint_building_exceptions,
                        osm_layer_metadata.get("buildings", {}))
                    methods[method] = geo
                    grade_reports[method] = grade_report
                    hard_violation_reports[method] = hard_report
                    station_offsets_by_method[method] = offsets
                    if geo:
                        print(f"  [{method}] 站点端点恢复后的硬约束复核通过")
                    elif hard_report["hard_constraint_status"] == "fail":
                        print(f"  [{method}] 站点端点恢复后穿越硬禁区，拒绝该候选")
            else:
                for method, geo in list(methods.items()):
                    geo = expand_single_cell_route(geo, (slat, slon), (elat, elon))
                    geo = attach_input_endpoints(geo, (slat, slon), (elat, elon))
                    if method == TOWER_GRAPH_METHOD:
                        geo, tower_span_reports[method] = validate_tower_span_lengths(
                            geo, args.tower_min_span_m, args.tower_max_span_m)
                    else:
                        geo, grade_reports[method] = filter_path_by_ground_grade(
                            geo, dem, transform, args.max_ground_grade, crs)
                    validation_mask = tower_span_mask
                    geo, hard_report = validate_route_hard_constraints(
                        geo, transform, validation_mask)
                    hard_report.update(tower_span_reports.get(method, {}))
                    if geo and osm_layer_metadata.get("buildings", {}).get("status") == "available":
                        geo, building_report = validate_route_against_buildings(
                            geo, building_geometries)
                        hard_report.update(building_report)
                        if building_report["building_geometry_status"] == "fail":
                            hard_report["hard_constraint_status"] = "fail"
                            hard_report["hard_violations"] += 1
                    annotate_route_validation_context(
                        hard_report, endpoint_building_exceptions,
                        osm_layer_metadata.get("buildings", {}))
                    hard_violation_reports[method] = hard_report
                    methods[method] = geo

            # 指标
            start_ll = (slat, slon); end_ll = (elat, elon)
            case_metrics = {}
            for m, geo in methods.items():
                if geo:
                    method_grade_weight = _metric_grade_penalty_weight(
                        m, experimental_neighbor_results, args.grade_penalty_weight,
                        experimental_theta_results)
                    method_turn_weight = (args.search_turn_penalty_weight
                                          if m in ("A*(turn-aware)", TOWER_GRAPH_METHOD)
                                          or m in sampled_pareto_by_method
                                          else 0.0)
                    metric_cost_grid = (tower_span_cost_grid
                                        if m == TOWER_GRAPH_METHOD else cost_grid)
                    metric_hard_mask = tower_span_mask
                    met = compute_metrics(
                        geo, dem, transform, metric_cost_grid, metric_hard_mask,
                        start_ll, end_ll,
                        grade_penalty_weight=method_grade_weight,
                        turn_penalty_weight=method_turn_weight,
                        pixel_spacing=route_pixel_spacing)
                    met["weight_independent_exposure"] = route_layer_exposure(
                        geo, transform, dem.shape, terrain, osm,
                        official_hazard_masks, osm_layer_metadata,
                        official_hazard_metadata.get("coverage_status", "unknown"),
                        official_hazard_metadata.get("layers", {}))
                    met["custom_constraint_route_exposure"] = route_custom_constraint_exposure(
                        geo, custom_constraints, transform, dem.shape)
                    met["population_data_coverage"] = route_population_coverage(
                        geo, population_valid_mask, transform, dem.shape)
                    met["search_window_margin_deg"] = args.window_margin_deg
                    met["search_window_edge_clearance_m"] = route_search_window_clearance_m(
                        geo, transform, dem.shape, route_pixel_spacing)
                    if m == astar_method and raw_astar_geo:
                        met["search_turn_penalty_weight"] = args.search_turn_penalty_weight
                        met["smoothing_turn_penalty_weight"] = args.smoothing_turn_penalty_weight
                        met["astar_search_path_objective"] = astar_path_objective(
                            a_path, cost_grid, dem, route_pixel_spacing,
                            args.grade_penalty_weight, args.search_turn_penalty_weight)
                        raw_at_search_weight = path_cost_integral(
                            raw_astar_geo, dem, transform, cost_grid,
                            args.grade_penalty_weight, args.search_turn_penalty_weight,
                            route_pixel_spacing)
                        post_at_search_weight = path_cost_integral(
                            geo, dem, transform, cost_grid,
                            args.grade_penalty_weight, args.search_turn_penalty_weight,
                            route_pixel_spacing)
                        raw_at_smoothing_weight = path_cost_integral(
                            raw_astar_geo, dem, transform, cost_grid,
                            args.grade_penalty_weight, args.smoothing_turn_penalty_weight,
                            route_pixel_spacing)
                        post_at_smoothing_weight = path_cost_integral(
                            geo, dem, transform, cost_grid,
                            args.grade_penalty_weight, args.smoothing_turn_penalty_weight,
                            route_pixel_spacing)
                        met["astar_raw_vs_post_cost_diagnostics"] = {
                            "raw_at_search_weight": raw_at_search_weight,
                            "post_at_search_weight": post_at_search_weight,
                            "raw_at_smoothing_weight": raw_at_smoothing_weight,
                            "post_at_smoothing_weight": post_at_smoothing_weight,
                            "post_minus_raw_at_search_weight_meter": (
                                post_at_search_weight["total_cost_meter"]
                                - raw_at_search_weight["total_cost_meter"]
                                if post_at_search_weight["status"] == "measured"
                                and raw_at_search_weight["status"] == "measured"
                                else None),
                            "post_minus_raw_at_smoothing_weight_meter": (
                                post_at_smoothing_weight["total_cost_meter"]
                                - raw_at_smoothing_weight["total_cost_meter"]
                                if post_at_smoothing_weight["status"] == "measured"
                                and raw_at_smoothing_weight["status"] == "measured"
                                else None),
                        }
                    if m in sampled_pareto_by_method:
                        met["candidate_role"] = "experimental_nonselectable_pareto_sample"
                        met["sampled_pareto"] = sampled_pareto_by_method[m]
                        met["search_turn_penalty_weight"] = args.search_turn_penalty_weight
                        met["smoothing_turn_penalty_weight"] = args.smoothing_turn_penalty_weight
                    if m in ("A*(baseline)", "A*(weighted)"):
                        met["astar_heuristic_weight"] = args.astar_heuristic_weight
                        met["ridge_penalty_weight"] = args.ridge_penalty_weight
                        met["landcover_penalty_weight"] = args.landcover_penalty_weight
                        met["official_hazard_weight"] = args.official_hazard_weight
                    if m == "A*(turn-aware)":
                        met["astar_heuristic_weight"] = args.astar_heuristic_weight
                        met["turn_penalty_weight"] = args.search_turn_penalty_weight
                        met["search_turn_penalty_weight"] = args.search_turn_penalty_weight
                        met["ridge_penalty_weight"] = args.ridge_penalty_weight
                        met["landcover_penalty_weight"] = args.landcover_penalty_weight
                        met["official_hazard_weight"] = args.official_hazard_weight
                    met["time_s"] = round({
                        "VIN-Grad": t_vin,
                        "PPO": t_ppo,
                        "Theta*(experimental)": t_theta,
                        THETA_SHARED_METHOD: experimental_theta_results.get(
                            THETA_SHARED_METHOD, {}).get("time_s", 0.0),
                        "A*(baseline)": t_astar,
                        "A*(weighted)": t_astar,
                        "A*(turn-aware)": t_astar,
                        TOWER_GRAPH_METHOD: t_tower_graph,
                    }.get(m, experimental_neighbor_results.get(m, {}).get("time_s", 0.0)), 2)
                    if m == TOWER_GRAPH_METHOD and experimental_tower_graph_results:
                        met["search_objective_model"] = "candidate_tower_graph_screening"
                        met["search_path_objective"] = {
                            "status": ("measured" if geo else
                                       "rejected_after_search" if experimental_tower_graph_results["path"]
                                       else "no_path"),
                            "unit": "cost_meter_plus_turn_penalty",
                            "tower_graph_parameters": {
                                key: value for key, value in experimental_tower_graph_results.items()
                                if key not in ("path", "time_s")
                            },
                            "objective_scope": (
                                "screening only; zero tower-site cost; optional parabolic DEM clearance "
                                "is not conductor or load-case certification"),
                        }
                        met["candidate_validation_status"] = _candidate_validation_status(
                            experimental_tower_graph_results["path"], geo)
                    if not station_task:
                        start_center = rowcol_to_geo(sr, sc, transform)
                        end_center = rowcol_to_geo(er, ec, transform)
                        met["start_input_to_grid_center_offset_m"] = round(
                            haversine_km(slon, slat, *start_center) * 1000.0, 2)
                        met["end_input_to_grid_center_offset_m"] = round(
                            haversine_km(elon, elat, *end_center) * 1000.0, 2)
                elif m in grade_reports:
                    met = {}
                else:
                    continue
                if m in grade_reports:
                    met.update(grade_reports[m])
                met.update(hard_violation_reports.get(m, {}))
                if m == TOWER_GRAPH_METHOD and experimental_tower_graph_results:
                    clearance_status = experimental_tower_graph_results[
                        "clearance_profile_status"]
                    met.update({
                        "candidate_validation_status": tower_candidate_validation_status(
                            geo, experimental_tower_graph_results["path"], clearance_status,
                            met.get("building_geometry_status"),
                            experimental_tower_graph_results["tower_site_slope_status"],
                            experimental_tower_graph_results["tower_site_pad_status"],
                            experimental_tower_graph_results["tower_site_pad_radius_m"]),
                        "clearance_profile_screening": experimental_tower_graph_results[
                            "clearance_profile_screening"],
                        "clearance_profile_status": clearance_status,
                        "clearance_profile_spans": experimental_tower_graph_results[
                            "clearance_profile_spans"],
                        "tower_site_max_surface_slope_deg": experimental_tower_graph_results[
                            "tower_site_max_surface_slope_deg"],
                        "tower_site_slope_status": experimental_tower_graph_results[
                            "tower_site_slope_status"],
                        "tower_site_slope_scope": experimental_tower_graph_results[
                            "tower_site_slope_scope"],
                        "tower_site_pad_radius_m": experimental_tower_graph_results[
                            "tower_site_pad_radius_m"],
                        "tower_site_pad_status": experimental_tower_graph_results[
                            "tower_site_pad_status"],
                        "selected_intermediate_tower_site_count": experimental_tower_graph_results[
                            "selected_intermediate_tower_site_count"],
                        "station_terminal_slope_status": experimental_tower_graph_results[
                            "station_terminal_slope_status"],
                        "tower_site_hazard_exposure": experimental_tower_graph_results[
                            "tower_site_hazard_exposure"],
                        "dem_vertical_datum": experimental_tower_graph_results[
                            "dem_vertical_datum"],
                    })
                if m in experimental_neighbor_results:
                    experiment = experimental_neighbor_results[m]
                    search_cost = experiment["search_cost"]
                    met["time_s"] = round(experiment["time_s"], 2)
                    met["search_objective_model"] = (
                        f"{len(experiment['neighbors'])}_neighbor_exact_raster_integral"
                        + ("_plus_per_search_edge_endpoint_grade_penalty"
                           if experiment["grade_penalty_weight"] > 0.0 else ""))
                    met["search_path_objective"] = {
                        "status": ("measured" if math.isfinite(search_cost)
                                   else "invalid_path" if experiment["path"] else "no_path"),
                        "unit": "cost_meter",
                        "search_objective_cost_meter": search_cost if math.isfinite(search_cost) else None,
                        "raster_cost": (experiment["search_raster_cost"]
                                        if math.isfinite(experiment["search_raster_cost"])
                                        else None),
                        "soft_grade_penalty_cost_meter": (
                            experiment["search_grade_penalty_cost"]
                            if math.isfinite(experiment["search_grade_penalty_cost"])
                            else None),
                        "soft_grade_penalty_weight": experiment[
                            "grade_penalty_weight"],
                        "neighbor_count": len(experiment["neighbors"]),
                        "turn_penalty": 0.0,
                        "hard_ground_grade_limit": args.max_ground_grade,
                        "objective_scope": experiment["objective_scope"],
                    }
                    met["candidate_validation_status"] = _candidate_validation_status(
                        experiment["path"], geo)
                    if m in SHARED_SEARCH_ROUTE_TAGS:
                        met.update(_shared_route_artifact_paths(
                            cname, m, None, None))
                if m in experimental_theta_results:
                    experiment = experimental_theta_results[m]
                    search_cost = experiment["search_cost"]
                    met["time_s"] = round(experiment["time_s"], 2)
                    met["search_objective_model"] = (
                        "theta_star_exact_raster_integral"
                        + ("_plus_per_search_edge_endpoint_grade_penalty"
                           if experiment["grade_penalty_weight"] > 0.0 else ""))
                    met["search_path_objective"] = {
                        "status": ("measured" if math.isfinite(search_cost)
                                   else "invalid_path" if experiment["path"] else "no_path"),
                        "unit": "cost_meter",
                        "search_objective_cost_meter": (
                            search_cost if math.isfinite(search_cost) else None),
                        "raster_cost": (
                            experiment["search_raster_cost"]
                            if math.isfinite(experiment["search_raster_cost"])
                            else None),
                        "soft_grade_penalty_cost_meter": (
                            search_cost - experiment["search_raster_cost"]
                            if math.isfinite(search_cost)
                            and math.isfinite(experiment["search_raster_cost"])
                            else None),
                        "soft_grade_penalty_weight": experiment[
                            "grade_penalty_weight"],
                        "turn_penalty": 0.0,
                        "hard_ground_grade_limit": args.max_ground_grade,
                        "objective_scope": experiment["objective_scope"],
                    }
                    met["candidate_validation_status"] = _candidate_validation_status(
                        experiment["path"], geo)
                    met.update(_shared_route_artifact_paths(
                        cname, m, None, None))
                if geo and "hard_constraint_status" not in met:
                    met["hard_constraint_status"] = (
                        "pass" if met.get("hard_violations", 0) == 0 else "fail")
                met.update(station_offsets_by_method.get(m, {}))
                case_metrics[m] = met

            if (experimental_tower_graph_results
                    and TOWER_GRAPH_METHOD not in case_metrics):
                case_metrics[TOWER_GRAPH_METHOD] = {
                    "candidate_validation_status": tower_candidate_validation_status(
                        None, experimental_tower_graph_results["path"],
                        experimental_tower_graph_results["clearance_profile_status"],
                        None, experimental_tower_graph_results["tower_site_slope_status"],
                        experimental_tower_graph_results["tower_site_pad_status"],
                        experimental_tower_graph_results["tower_site_pad_radius_m"]),
                    "clearance_profile_screening": experimental_tower_graph_results[
                        "clearance_profile_screening"],
                    "clearance_profile_status": experimental_tower_graph_results[
                        "clearance_profile_status"],
                    "clearance_profile_spans": experimental_tower_graph_results[
                        "clearance_profile_spans"],
                    "tower_site_max_surface_slope_deg": experimental_tower_graph_results[
                        "tower_site_max_surface_slope_deg"],
                    "tower_site_slope_status": experimental_tower_graph_results[
                        "tower_site_slope_status"],
                    "tower_site_slope_scope": experimental_tower_graph_results[
                        "tower_site_slope_scope"],
                    "tower_site_pad_radius_m": experimental_tower_graph_results[
                        "tower_site_pad_radius_m"],
                    "tower_site_pad_status": experimental_tower_graph_results[
                        "tower_site_pad_status"],
                    "selected_intermediate_tower_site_count": experimental_tower_graph_results[
                        "selected_intermediate_tower_site_count"],
                    "station_terminal_slope_status": experimental_tower_graph_results[
                        "station_terminal_slope_status"],
                    "tower_site_hazard_exposure": experimental_tower_graph_results[
                        "tower_site_hazard_exposure"],
                    "dem_vertical_datum": experimental_tower_graph_results[
                        "dem_vertical_datum"],
                }

            # 按固定优先规则筛选通过已配置筛查的候选；覆盖完整性另行标注。
            # 真实线仍只在规划选路完成后读取，用于事后评分。
            best_method, _, selection_rule = _select_final_route(
                methods, case_metrics,
                select_tower_graph=args.select_experimental_tower_graph)
            final_geo = methods.get(best_method) if best_method else None
            final_is_ai = False if best_method else None
            for method, candidate in sampled_pareto_by_method.items():
                geo = methods.get(method)
                canonical_geo = (json.dumps(
                    [[float(lon), float(lat)] for lon, lat in geo],
                    ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                    if geo else None)
                candidate["post_validation_path_sha256"] = (
                    hashlib.sha256(canonical_geo.encode("utf-8")).hexdigest()
                    if canonical_geo else None)
                candidate["candidate_frozen_before_reference_read"] = True
            sampled_pareto_freeze_sha256 = None
            if args.experimental_sampled_pareto:
                final_dir = OUT / route_output_name(cname)
                final_dir.mkdir(parents=True, exist_ok=True)
                freeze_manifest = {
                    "scope": "finite_scalarization_samples_only",
                    "objective_vector_scope": "pre_smoothing_search_grid_paths",
                    "reference_geometry_used_for_planning": False,
                    "frozen_before_reference_read": True,
                    "candidates": [
                        {
                            "candidate_id": method,
                            "path_sha256": candidate[
                                "post_validation_path_sha256"],
                            "sampled_environmental_weights": candidate[
                                "sampled_environmental_weights"],
                            "objective_vector_on_search_grid": candidate[
                                "objective_vector_on_search_grid"],
                            "post_validation_status": hard_violation_reports.get(
                                method, {}).get("hard_constraint_status", "no_path"),
                        }
                        for method, candidate in sampled_pareto_by_method.items()
                    ],
                }
                freeze_bytes = json.dumps(
                    freeze_manifest, ensure_ascii=False, indent=2,
                    allow_nan=False).encode("utf-8")
                (final_dir / "sampled_pareto_freeze.json").write_bytes(freeze_bytes)
                sampled_pareto_freeze_sha256 = hashlib.sha256(freeze_bytes).hexdigest()
            osm_data_quality, missing_osm_layers = screening_data_quality(osm_layer_metadata)
            hazard_coverage_complete = (
                official_hazard_metadata.get("coverage_status") == "complete")
            missing_hazard_layers = (
                [] if args.official_hazard_weight == 0 or hazard_coverage_complete
                else ["official_hazards"])
            missing_screening_layers = missing_osm_layers + missing_hazard_layers
            if (args.population_weight > 0
                    and population_metadata.get("status") != "complete"):
                missing_screening_layers.append("population_raster_coverage")
            screening_quality = (
                "incomplete_screening_layers" if missing_screening_layers
                else osm_data_quality)
            hazard_route_exposure = route_hazard_exposure(
                final_geo, official_hazard_masks, transform, dem.shape)
            selected_metrics = case_metrics.get(best_method, {}) if best_method else {}

            real_evaluation_status = "not_requested"
            real_evaluation_methods = []
            if use_real:
                real_evaluation_status = "reference_missing"
                if global_real_exists:
                    real_geo, real_evaluation_status = load_case_reference(
                        args.real_shp, require_single_line=True)
                else:
                    try:
                        rp = resolve_real_path(
                            cname, args.real_dir,
                            include_global_mapping=allow_global_reference_mapping)
                    except AmbiguousReferenceError as exc:
                        print(f"  [warn] 目标线路参考文件匹配不唯一: {exc}")
                        real_geo, real_evaluation_status = None, "reference_ambiguous"
                    else:
                        real_geo, real_evaluation_status = load_case_reference(rp)
                if real_geo:
                    print(f"  [POST-EVAL] 加载真实线路: {len(real_geo)}条折线")
                    grid_corners = [transform * point for point in (
                        (0, 0), (W, 0), (W, H), (0, H))]
                    planning_bounds = (
                        min(point[0] for point in grid_corners),
                        min(point[1] for point in grid_corners),
                        max(point[0] for point in grid_corners),
                        max(point[1] for point in grid_corners),
                    )
                    for method, geo in methods.items():
                        if not geo or method not in case_metrics:
                            continue
                        dev = compute_deviation(
                            geo, real_geo, planning_bounds_wgs84=planning_bounds,
                            chain_profile_bins=10)
                        if not dev:
                            continue
                        metrics = case_metrics[method]
                        err = eval_planned_route_error(geo, dev)
                        if err:
                            metrics.update(err)
                        real_evaluation_methods.append(method)
                        if err:
                            similarity = err["route_similarity"]
                            symmetric = similarity["symmetric"]
                            coverage_100m = symmetric["coverage_f1_within_m"]["100"]
                            coverage_1km = symmetric["coverage_f1_within_m"]["1000"]
                            print(f"  [{method}] 较差方向P95={symmetric['p95_m']:.0f}m "
                                  f"采样最大偏移={symmetric['sampled_hausdorff_m']:.0f}m "
                                  f"100m覆盖F1={coverage_100m:.1%} "
                                  f"1km覆盖F1={coverage_1km:.1%} "
                                  f"长度差={similarity['length']['absolute_difference_km']:.2f}km")
                    real_evaluation_status = (
                        "comparison_completed" if real_evaluation_methods
                        else "reference_loaded_no_comparable_route")

            selected_metrics = case_metrics.get(best_method, {}) if best_method else {}
            planning_status = route_planning_status(
                final_geo, station_task, missing_screening_layers,
                osm_coverage_unverified=(
                    osm_data_quality != "osm_proxy_coverage_complete"))
            case_metrics[FINAL_ROUTE_LABEL] = {
                "method": best_method,
                "selection_rule": selection_rule,
                "route_similarity_version": selected_metrics.get("route_similarity_version"),
                "route_similarity": selected_metrics.get("route_similarity"),
                "is_ai_method": final_is_ai,
                "real_evaluation_status": real_evaluation_status,
                "real_evaluation_methods": real_evaluation_methods,
                "planning_status": planning_status,
                "engineering_feasibility": "not_evaluated",
                **station_endpoint_properties(station_task),
                "hard_constraint_status": selected_metrics.get("hard_constraint_status", "no_path"),
                "hard_violations": selected_metrics.get("hard_violations"),
                "osm_data_quality": osm_data_quality,
                "screening_data_quality": screening_quality,
                "missing_osm_screening_layers": missing_osm_layers,
                "missing_screening_layers": missing_screening_layers,
                "official_hazard_metadata": official_hazard_metadata,
                "official_hazard_route_exposure": hazard_route_exposure,
                "constraint_layer_metadata": custom_constraint_metadata,
                "land_mask_metadata": land_mask_metadata,
                "length_km": round(path_length_km(final_geo), 3) if final_geo else None}
            case_metrics[FINAL_ROUTE_LABEL].update(selected_metrics)
            if final_geo:
                case_metrics[FINAL_ROUTE_LABEL].update(route_profile_metrics(final_geo, dem, transform))
            if best_method in grade_reports:
                case_metrics[FINAL_ROUTE_LABEL].update(grade_reports[best_method])
            case_metrics[FINAL_ROUTE_LABEL].update(station_offsets_by_method.get(best_method, {}))
            case_metrics[FINAL_ROUTE_LABEL].update({
                "settlement_weight": args.settlement_weight,
                "official_hazard_weight": args.official_hazard_weight,
                "turn_penalty_weight": args.turn_penalty_weight,
                "search_turn_penalty_weight": args.search_turn_penalty_weight,
                "smoothing_turn_penalty_weight": args.smoothing_turn_penalty_weight,
                "ridge_penalty_weight": args.ridge_penalty_weight,
                "landcover_penalty_weight": args.landcover_penalty_weight,
                "building_clearance_m": args.building_clearance_m,
                "grade_penalty_weight": args.grade_penalty_weight,
                "terrain_slope_weight": args.terrain_slope_weight,
                "roughness_weight": args.roughness_weight,
                "population_weight": args.population_weight,
                "population_layer": population_metadata,
                "population_data_coverage": route_population_coverage(
                    final_geo, population_valid_mask, transform, dem.shape),
                "max_ground_grade_ratio": args.max_ground_grade,
                "max_surface_slope_deg": args.max_surface_slope_deg,
                "max_surface_slope_scope": "candidate_tower_sites_only",
                "max_terrain_elevation_m": args.max_terrain_elevation_m,
                "osm_layer_metadata": osm_layer_metadata,
                "voltage_rules_applied": False,
                "route_constraints_limitations": (
                    "Population raster, when enabled, is a modelled exposure proxy; no sensitive-target inventory, official protected-area boundaries, "
                    "conductor clearance, tower siting or span feasibility checked; administrative land polygons, when enabled, are a coarse land-area screen, not a surveyed shoreline"),
            })
            if station_task:
                start_station = station_task["from_station"]
                end_station = station_task["to_station"]
                station_measurements = station_elevation_metrics(station_task)
                for metrics in case_metrics.values():
                    if isinstance(metrics, dict):
                        metrics.update({
                            "route_id": cname,
                            "from_station_id": start_station["station_id"],
                            "from_station_name": start_station["name"],
                            "from_station_type": start_station["station_type"] or None,
                            "from_station_source": start_station["source"] or None,
                            "from_station_verified": start_station["verified"] or None,
                            "to_station_id": end_station["station_id"],
                            "to_station_name": end_station["name"],
                            "to_station_type": end_station["station_type"] or None,
                            "to_station_source": end_station["source"] or None,
                            "to_station_verified": end_station["verified"] or None,
                            **station_endpoint_properties(station_task),
                            "voltage_kv": station_task["voltage_kv"] or None,
                            **station_coordinate_crs_properties(station_task),
                            **station_measurements,
                            "settlement_weight": args.settlement_weight,
                            "ridge_penalty_weight": args.ridge_penalty_weight,
                            "landcover_penalty_weight": args.landcover_penalty_weight,
                            "building_clearance_m": args.building_clearance_m,
                            "grade_penalty_weight": args.grade_penalty_weight,
                            "terrain_slope_weight": args.terrain_slope_weight,
                            "roughness_weight": args.roughness_weight,
                            "population_weight": args.population_weight,
                            "population_layer": population_metadata,
                            "max_ground_grade_ratio": args.max_ground_grade,
                            "max_surface_slope_deg": args.max_surface_slope_deg,
                            "max_surface_slope_scope": "candidate_tower_sites_only",
                            "max_terrain_elevation_m": args.max_terrain_elevation_m,
                            "route_constraints_limitations": "Population raster, when enabled, is a modelled exposure proxy; no sensitive-target inventory, official protected-area layer, conductor clearance, tower siting or span feasibility checked",
                        })
            print(f"  [集成] 最终路线采用 {best_method} ({selection_rule})；线路相似度仅作事后指标")
            # 导出最终路线矢量 (GeoJSON)
            safe_now = route_output_name(cname)
            final_dir = OUT / safe_now
            final_dir.mkdir(parents=True, exist_ok=True)
            try:
                if final_geo:
                    _fg = [{"type": "Feature",
                            "geometry": {"type": "LineString",
                                         "coordinates": geojson_coordinates(final_geo)},
                            "properties": {"case": cname, "method": best_method,
                                            "route_similarity_version": selected_metrics.get("route_similarity_version"),
                                            "route_similarity": selected_metrics.get("route_similarity"),
                                           "selection_rule": selection_rule,
                                           "planning_status": planning_status,
                                           "engineering_feasibility": "not_evaluated",
                                           **station_endpoint_properties(station_task),
                                           "hard_constraint_status": selected_metrics.get("hard_constraint_status", "no_path"),
                                           "hard_violations": selected_metrics.get("hard_violations"),
                                           "hard_constraint_validation_scope": "loaded_raster_masks_only",
                                           "hard_constraint_completeness_status": "not_independently_verified",
                                           "osm_data_quality": osm_data_quality,
                                           "screening_data_quality": screening_quality,
                                           "missing_osm_screening_layers": missing_osm_layers,
                                           "missing_screening_layers": missing_screening_layers,
                                           "official_hazard_metadata": official_hazard_metadata,
                                           "official_hazard_route_exposure": hazard_route_exposure,
                                           "constraint_layer_metadata": custom_constraint_metadata,
                                           "length_km": selected_metrics.get("length_km"),
                                           "start_input_to_grid_center_offset_m": selected_metrics.get("start_input_to_grid_center_offset_m"),
                                           "end_input_to_grid_center_offset_m": selected_metrics.get("end_input_to_grid_center_offset_m"),
                                           "mean_cost": selected_metrics.get("mean_cost"),
                                           "path_cost_integral": selected_metrics.get("path_cost_integral"),
                                           "astar_search_path_objective": selected_metrics.get("astar_search_path_objective"),
                                           "sinuosity": selected_metrics.get("sinuosity"),
                                           "cum_relief_m": selected_metrics.get("cum_relief_m"),
                                           "net_relief_m": selected_metrics.get("net_relief_m"),
                                           "from_station_id": station_task["from_station"]["station_id"] if station_task else None,
                                           "to_station_id": station_task["to_station"]["station_id"] if station_task else None,
                                           "from_station_name": station_task["from_station"]["name"] if station_task else None,
                                           "to_station_name": station_task["to_station"]["name"] if station_task else None,
                                           **station_coordinate_crs_properties(station_task),
                                           "from_station_source": station_task["from_station"]["source"] or None if station_task else None,
                                           "from_station_verified": station_task["from_station"]["verified"] or None if station_task else None,
                                           "to_station_source": station_task["to_station"]["source"] or None if station_task else None,
                                           "to_station_verified": station_task["to_station"]["verified"] or None if station_task else None,
                                           **station_endpoint_properties(station_task),
                                           "voltage_kv": station_task["voltage_kv"] or None if station_task else None,
                                           **station_measurements,
                                           "voltage_rules_applied": False,
                                           "settlement_weight": args.settlement_weight,
                                           "official_hazard_weight": args.official_hazard_weight,
                                           "turn_penalty_weight": args.turn_penalty_weight,
                                           "search_turn_penalty_weight": args.search_turn_penalty_weight,
                                           "smoothing_turn_penalty_weight": args.smoothing_turn_penalty_weight,
                                           "ridge_penalty_weight": args.ridge_penalty_weight,
                                           "landcover_penalty_weight": args.landcover_penalty_weight,
                                           "building_clearance_m": args.building_clearance_m,
                                           "grade_penalty_weight": args.grade_penalty_weight,
                                           "terrain_slope_weight": args.terrain_slope_weight,
                                           "roughness_weight": args.roughness_weight,
                                            "population_weight": args.population_weight,
                                            "population_layer": population_metadata,
                                            "population_data_coverage": route_population_coverage(
                                                final_geo, population_valid_mask, transform, dem.shape),
                                           "max_ground_grade_ratio": args.max_ground_grade,
                                           "max_surface_slope_deg": args.max_surface_slope_deg,
                                           "max_surface_slope_scope": "candidate_tower_sites_only",
                                           "max_terrain_elevation_m": args.max_terrain_elevation_m,
                                           "osm_layer_metadata": osm_layer_metadata,
                                           "route_constraints_limitations": (
                                                "Population raster, when enabled, is a modelled exposure proxy; no sensitive-target inventory, official protected-area boundaries, "
                                               "conductor clearance, tower siting or span feasibility checked"),
                                           **grade_reports.get(best_method, {}),
                                           **station_offsets_by_method.get(best_method, {}),
                                           **(route_profile_metrics(final_geo, dem, transform) if final_geo else {})}}]
                    write_geojson_verified(
                        final_dir / "final_route.geojson",
                        {"type": "FeatureCollection", "features": _fg})
                    if final_geo:
                        write_elevation_profile_csv(final_geo, dem, transform, final_dir / "elevation_profile.csv")
                if args.experimental_sampled_pareto:
                    pareto_manifest = {
                        "scope": "finite_scalarization_samples_only",
                        "objective_vector_scope": "pre_smoothing_search_grid_paths",
                        "reference_geometry_used_for_planning": False,
                        "freeze_manifest": "sampled_pareto_freeze.json",
                        "freeze_manifest_sha256": sampled_pareto_freeze_sha256,
                        "candidates": [],
                    }
                    for method, candidate in sampled_pareto_by_method.items():
                        metrics = case_metrics.get(method, {})
                        suffix = method.rsplit(" ", 1)[-1]
                        pareto_manifest["candidates"].append({
                            "candidate_id": method,
                            "geojson": f"planned_route_sampled_pareto_{suffix}.geojson"
                            if methods.get(method) else None,
                            **candidate,
                            "post_validation_status": metrics.get(
                                "hard_constraint_status", "no_path"),
                            "hard_violations": metrics.get("hard_violations"),
                            "post_smoothing_path_cost_integral": metrics.get(
                                "path_cost_integral"),
                            "post_validation_path_sha256": candidate.get(
                                "post_validation_path_sha256"),
                            "candidate_frozen_before_reference_read": candidate.get(
                                "candidate_frozen_before_reference_read"),
                            "route_similarity_version": metrics.get(
                                "route_similarity_version"),
                            "route_similarity": metrics.get("route_similarity"),
                            "length_km": metrics.get("length_km"),
                        })
                    with open(final_dir / "sampled_pareto_candidates.json", "w",
                              encoding="utf-8") as _pf:
                        json.dump(pareto_manifest, _pf, ensure_ascii=False, indent=2)
            except Exception as exc:
                raise RuntimeError(f"最终路线必需文件写出失败: {exc}") from exc
            # 逐走廊绘制规划路线与参考线对比图并导出矢量。
            safe = route_output_name(cname)
            cdir = OUT / safe
            cdir.mkdir(parents=True, exist_ok=True)
            plot_comparison(dem, transform, methods, start_ll, end_ll,
                            cdir / f"path_comparison_{safe}.png",
                            title=f"规划路线与参考线路对比 - {cname}", real_ll=real_geo,
                            hazard_masks=official_hazard_masks)
            if args.experimental_sampled_pareto:
                sampled_methods = {
                    name: methods[name] for name in sampled_pareto_by_method
                    if methods.get(name)
                }
                if sampled_methods:
                    plot_comparison(
                        dem, transform, sampled_methods, start_ll, end_ll,
                        cdir / f"sampled_pareto_comparison_{safe}.png",
                        title=f"启发式多目标候选（非工程定线）- {cname}",
                        real_ll=real_geo, hazard_masks=official_hazard_masks,
                    )
            # 逐用例导出矢量与坐标(便于复用/二次绘图)
            for m, geo in methods.items():
                if m.startswith("Sampled Pareto "):
                    tag = f"sampled_pareto_{m.rsplit(' ', 1)[-1]}"
                else:
                    tag = _route_method_export_tag(m)
                export_route(geo, crs, cdir / f"planned_route_{tag}.shp",
                             cdir / f"planned_route_{tag}.geojson")
            legacy_astar_geo = methods.get(astar_method)
            if legacy_astar_geo:
                export_route(
                    legacy_astar_geo, crs,
                    cdir / "planned_route_aastar.shp",
                    cdir / "planned_route_aastar.geojson",
                )
            if raw_astar_geo:
                export_route(
                    raw_astar_geo, crs,
                    cdir / "astar_raw_route.shp",
                    cdir / "astar_raw_route.geojson",
                )
            for method, tag, search_geo in _shared_search_route_exports(
                    experimental_neighbor_results, experimental_theta_results,
                    transform):
                search_geojson_path = (
                    cdir / f"planned_route_{tag}_search_raw.geojson")
                validated_geo = methods.get(method)
                validated_geojson_path = (
                    cdir / f"planned_route_{tag}_validated.geojson"
                    if validated_geo and len(validated_geo) >= 2 else None)
                artifact_stems = [search_geojson_path.stem]
                if validated_geojson_path is not None:
                    artifact_stems.append(validated_geojson_path.stem)
                _assert_route_artifacts_absent(cdir, artifact_stems)
                export_route(
                    search_geo, crs,
                    cdir / f"planned_route_{tag}_search_raw.shp",
                    search_geojson_path,
                )
                if not search_geojson_path.is_file():
                    raise RuntimeError(
                        f"raw search route export missing: {search_geojson_path}")
                if validated_geojson_path is not None:
                    export_route(
                        validated_geo, crs,
                        cdir / f"planned_route_{tag}_validated.shp",
                        validated_geojson_path,
                    )
                    if not validated_geojson_path.is_file():
                        raise RuntimeError(
                            f"validated route export missing: {validated_geojson_path}")
                case_metrics[method].update(_shared_route_artifact_paths(
                    cname, method, search_geo,
                    validated_geo if validated_geojson_path else None))
            # 导出真实/参考线路(便于下游复用与二次对比)
            if real_geo:
                export_route(real_geo, crs, cdir / "real_route.shp", cdir / "real_route.geojson")
            paths_json = {m: [[round(lon, 6), round(lat, 6)] for lon, lat in geo]
                          for m, geo in methods.items() if geo}
            with open(cdir / "paths.json", "w", encoding="utf-8") as f:
                json.dump(paths_json, f, ensure_ascii=False)

            # 仅首个用例额外交互兼容产物(单图 + 值图 + 主方法指标)
            OUT.mkdir(parents=True, exist_ok=True)
            if not main_done:
                # 立即置位, 防止任一子步骤异常被外层 per-corridor except 捕获后 main_done 未设置,
                # 导致后续走廊重新进入本块并错误覆盖根目录产物。
                main_done = True
                try:
                    plot_comparison(dem, transform, methods, start_ll, end_ll,
                                    OUT / "path_comparison.png", title=f"规划路线与参考线路对比 - {cname}",
                                    real_ll=real_geo, hazard_masks=official_hazard_masks)
                except Exception as _pe:
                    print(f"[WARN] 主对比图绘制失败(跳过, 不影响本走廊其余产出): {_pe}")
                if "VIN-Grad" in methods and HAS_TORCH and V_main is not None:
                    # 复用主流程已计算的 V（避免重复 80 步 Bellman 训练）
                    try:
                        plot_value_map(V_main, transform, methods["VIN-Grad"], start_ll, end_ll, OUT / "value_map.png")
                    except Exception as _ve:
                        print(f"[WARN] 值图绘制失败(跳过): {_ve}")
                if "VIN-Grad" in case_metrics:
                    try:
                        with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
                            json.dump(case_metrics["VIN-Grad"], f, ensure_ascii=False, indent=2)
                    except Exception as _me:
                        print(f"[WARN] metrics.json 写出失败(跳过): {_me}")
            all_results.append({"case": cname, "metrics": case_metrics})
            # 只在本走廊必需图件、矢量与路径文件写出后，增量提交其成功记录。
            try:
                _real_out = []
                for _rec in all_results:
                    _entry = {"case": _rec["case"]}
                    for _m, _met in _rec["metrics"].items():
                        if "route_similarity" in _met:
                            _entry[_m] = {
                                "method": _met.get("method", _m),
                                "length_km": _met.get("length_km"),
                                "route_similarity_version": _met.get("route_similarity_version"),
                                "route_similarity": _met["route_similarity"],
                            }
                    if _entry:
                        _real_out.append(_entry)
                with open(OUT / "real_comparison_metrics.json", "w", encoding="utf-8") as _f:
                    json.dump(_real_out, _f, ensure_ascii=False, indent=2)
            except Exception:
                pass
        except Exception as _exc:
            import traceback as _tb
            _tb.print_exc()
            print(f"[WARN] 用例 {cname} 处理失败, 跳过此用例: {_exc}")
            failed_cases.append({"case": cname, "error": format_case_error(_exc)})
        finally:
            # Drop per-case raster/vector references before loading the next window.
            dem = population_density = population_proxy = population_valid_mask = None
            population_unknown_mask = dem_valid_mask = None
            osm = terrain = stack = model_cost = cost = None
            slope = elev = building_mask = building_geometries = None
            custom_constraints = endpoint_blockers = endpoint_building_exceptions = None
            hard_mask = tower_span_mask = cost_grid = tower_span_cost_grid = None
            official_hazard_masks = methods = experimental_neighbor_results = None
            experimental_tower_graph_results = sampled_pareto_by_method = None
            real_geo = None
            import gc as _gc
            _gc.collect()
            if HAS_TORCH:
                try:
                    import torch as _torch
                    _torch.cuda.empty_cache()
                except Exception:
                    pass

    run_summary = {
        "scope": "per_case_planning_and_required_route_artifacts",
        **summarize_run_completion(len(cases), all_results, failed_cases),
        "real_evaluation_by_case": [
            {"case": record["case"],
             "status": record["metrics"].get(FINAL_ROUTE_LABEL, {}).get("real_evaluation_status", "not_recorded"),
             "methods": record["metrics"].get(FINAL_ROUTE_LABEL, {}).get("real_evaluation_methods", [])}
            for record in all_results
        ],
        "dem": str(dem_path),
        "max_side": args.max_side,
        "argv": sys.argv,
    }
    try:
        with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
            json.dump(run_summary, f, ensure_ascii=False, indent=2)
    except Exception as _exc:
        print(f"[WARN] 批次运行摘要写出失败: {_exc}")
    print(f"[批次状态] 应测 {len(cases)} 条，完成 {run_summary['completed_cases']} 条，"
          f"无可行路线 {len(run_summary['no_route_cases'])} 条，失败 {len(failed_cases)} 条")

    # 聚合对比图 + 指标表
    _write_comparison_outputs(all_results, OUT)
    # CSV
    rows = []
    for rec in all_results:
        for m, met in rec["metrics"].items():
            row = {"case": rec["case"], "method": m, **met}
            if "route_similarity" in row:
                row["route_similarity_json"] = json.dumps(
                    row.pop("route_similarity"), ensure_ascii=False)
            rows.append(row)
    if rows:
        import csv
        # 用所有行的键并集作为表头, 避免不同方法(A*/VIN/PPO)或有无真实对比导致字段不一致
        all_keys = []
        for r in rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)
        with open(OUT / "comparison_metrics.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader(); w.writerows(rows)

    # 参考线路空间相似度 JSON + 控制台摘要
    if use_real:
        real_out = []
        for rec in all_results:
            entry = {"case": rec["case"]}
            for m, met in rec["metrics"].items():
                if "route_similarity" in met:
                    entry[m] = {
                        "method": met.get("method", m),
                        "length_km": met.get("length_km"),
                        "route_similarity_version": met.get("route_similarity_version"),
                        "route_similarity": met["route_similarity"],
                    }
            if entry:
                real_out.append(entry)
        try:
            with open(OUT / "real_comparison_metrics.json", "w", encoding="utf-8") as f:
                json.dump(real_out, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print("\n" + "=" * 70)
        print("  参考线路空间相似度 (双向米制偏移 + 线路覆盖；仅事后评估)")
        print("=" * 70)
        print(f"  {'走廊':16s} {'方法':16s} {'较差方向P95(m)':>14s} {'采样最大(m)':>12s} "
              f"{'F1@100m':>9s} {'F1@1km':>9s} {'长度差(km)':>11s}")
        candidate_methods = ["A*(baseline)", "A*(weighted)", "A*(turn-aware)",
                             TOWER_GRAPH_METHOD,
                             ASTAR_8_EXACT_METHOD, ASTAR_16_METHOD,
                             ASTAR_8_SHARED_METHOD, ASTAR_16_SHARED_METHOD,
            "Theta*(experimental)", THETA_SHARED_METHOD, "VIN-Grad", "PPO"]
        failed_case_names = {item["case"] for item in failed_cases}
        for rec in all_results:
            if rec["case"] in failed_case_names or rec["metrics"].get(FINAL_ROUTE_LABEL, {}).get("planning_status") == "no_hard_constraint_compliant_route":
                continue
            final_route = rec["metrics"].get(FINAL_ROUTE_LABEL, {})
            endpoint_group = (
                "声明独立端点" if (
                    final_route.get("station_endpoint_qualification")
                    == "declared_verified_line_connection_points"
                    and final_route.get("station_endpoint_reference_independence")
                    == "declared_independent_of_reference"
                ) else "探索诊断/独立性未证"
            )
            for m in candidate_methods:
                met = rec["metrics"].get(m, {})
                similarity = met.get("route_similarity")
                if similarity:
                    symmetric = similarity["symmetric"]
                    print(f"  {rec['case']:16s} {m:16s} [{endpoint_group}] "
                          f"{symmetric['p95_m']:11.0f} "
                          f"{symmetric['sampled_hausdorff_m']:12.0f} "
                          f"{symmetric['coverage_f1_within_m']['100']:8.1%} "
                          f"{symmetric['coverage_f1_within_m']['1000']:8.1%} "
                          f"{similarity['length']['absolute_difference_km']:11.2f}")
        print("  说明：误差与覆盖率只作事后诊断，不参与选路；独立性分组读取站点表声明，"
              "程序不认证声明真实性或其相对参考线的独立性。")
        print("-" * 70)
        for m in candidate_methods:
            summary = method_visual_similarity_summary(
                all_results, m, expected_cases=len(cases),
                failed_case_names=failed_case_names,
            )
            independent = summary["declared_independent_endpoints"]
            exploratory = summary["exploratory_or_independence_unproven"]
            independent_p95 = independent["median_symmetric_p95_m"]
            exploratory_p95 = exploratory["median_symmetric_p95_m"]
            independent_p95_text = f"{independent_p95:.1f}m" if independent_p95 is not None else "NA"
            exploratory_p95_text = f"{exploratory_p95:.1f}m" if exploratory_p95 is not None else "NA"
            print(f"  {m:16s} 声明独立端点={independent['measured']}/{independent['endpoint_cases']} "
                  f"中位P95={independent_p95_text}；探索/独立性未证={exploratory['measured']}/"
                  f"{exploratory['endpoint_cases']} 中位P95={exploratory_p95_text} "
                  f"未分类={summary['unclassified_cases']}")
        heuristic_selected = [r["case"] for r in all_results
                          if r["case"] not in failed_case_names
                          and r["metrics"].get(FINAL_ROUTE_LABEL, {}).get("planning_status")
                          != "no_hard_constraint_compliant_route"
                          if r["metrics"].get(FINAL_ROUTE_LABEL, {}).get("method")
                          in ("A*(baseline)", "A*(weighted)", "A*(turn-aware)")]
        if heuristic_selected:
            print(f"  启发式搜索生成最终路线的走廊: {heuristic_selected}")
        print("=" * 70)

    # 控制台摘要
    print("\n" + "=" * 70)
    print("  结果摘要（各走廊均值）")
    print("=" * 70)
    for m in ["A*(baseline)", "A*(weighted)", "A*(turn-aware)",
              ASTAR_8_EXACT_METHOD, ASTAR_16_METHOD,
              ASTAR_8_SHARED_METHOD, ASTAR_16_SHARED_METHOD,
              "Theta*(experimental)", THETA_SHARED_METHOD, "VIN-Grad", "PPO"]:
        method_metrics = _method_route_metrics(all_results, m)
        lens = [metrics["length_km"] for metrics in method_metrics]
        mc = [metrics["mean_cost"] for metrics in method_metrics]
        if lens:
            print(f"  {m:14s} 平均长度 {np.mean(lens):7.2f}km  平均代价 {np.mean(mc):.3f}")
    print(f"  总耗时: {time.time() - t0:.1f}s")
    print(f"  输出目录: {OUT}")
    print("=" * 70)
    try:
        provenance_path = write_run_provenance(
            run_provenance, OUT,
            (args.real_dir, Path(args.real_shp).parent if args.real_shp else None))
        print(f"[运行溯源] {provenance_path}")
    except Exception as exc:
        print(f"[WARN] 运行溯源写出失败: {exc}")


if __name__ == "__main__":
    main()
