#!/usr/bin/env python3
"""Post-hoc OSM infrastructure exposure audit for frozen Taiwan route candidates.

Historical line geometries are read only after every planned route artifact has
been matched to the pre-scoring candidate freeze. This script never plans routes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, shape
from shapely.ops import linemerge, transform as transform_geometry, unary_union
from shapely.strtree import STRtree

from transmission_path_planner.metrics import load_real_path
from transmission_path_planner.osm import load_osm_geoms_with_status


LAYERS = {
    "roads": ("taiwan_roads.pkl", "highway"),
    "water": ("taiwan_water.pkl", "waterway"),
    "railways": ("taiwan_railways.pkl", "railway"),
}
THRESHOLDS_M = (25, 50, 100, 250, 500, 1000)
CASE_IDS = ("TW59", "TW345", "TW102")
COLORS = {"A*(8-neighbor shared objective experimental)": "#277da1",
          "A*(16-neighbor shared objective experimental)": "#43aa8b",
          "Theta*(shared objective experimental)": "#f8961e"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def line_parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry] if geometry.length > 0 else []
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        return [part for child in geometry.geoms for part in line_parts(child)]
    raise ValueError(f"expected line geometry, got {geometry.geom_type}")


def merge_lines(lines):
    merged = unary_union(lines)
    return linemerge(merged) if merged.geom_type in {"MultiLineString", "GeometryCollection"} else merged


def line_from_geojson(path: Path):
    document = read_json(path)
    features = document.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError(f"no GeoJSON features in {path}")
    lines = []
    for feature in features:
        geometry = feature.get("geometry")
        if geometry:
            lines.extend(line_parts(shape(geometry)))
    if not lines:
        raise ValueError(f"no line geometries in {path}")
    return merge_lines(lines)


def projected(geometry, transformer):
    return transform_geometry(transformer.transform, geometry)


def length_weighted_samples(lines, step_m=50.0):
    points = []
    weights = []
    for line in lines:
        if line.length <= 0:
            continue
        count = max(1, int(math.ceil(line.length / step_m)))
        weight = line.length / count
        for index in range(count):
            points.append(line.interpolate((index + 0.5) * weight))
            weights.append(weight)
    return points, np.asarray(weights, dtype=np.float64)


def weighted_percentile(values, weights, percentile):
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    mids = (np.cumsum(sorted_weights) - 0.5 * sorted_weights) / sorted_weights.sum()
    return float(np.interp(percentile / 100.0, mids, sorted_values,
                           left=sorted_values[0], right=sorted_values[-1]))


def nearest_exposure(lines, feature_geometries):
    route_length = float(sum(line.length for line in lines))
    if not feature_geometries:
        return {"status": "no_features_observed", "route_length_m": route_length,
                "sample_step_m": 50.0, "sample_count": 0,
                "p50_m": None, "p90_m": None, "p95_m": None,
                "mean_m": None, "within_m": {str(x): None for x in THRESHOLDS_M}}
    tree = STRtree(feature_geometries)
    points, weights = length_weighted_samples(lines)
    distances = np.asarray([
        point.distance(feature_geometries[int(tree.nearest(point))])
        for point in points
    ], dtype=np.float64)
    return {
        "status": "measured_on_observed_features_only",
        "route_length_m": route_length,
        "sample_step_m": 50.0,
        "sample_count": int(len(distances)),
        "p50_m": weighted_percentile(distances, weights, 50),
        "p90_m": weighted_percentile(distances, weights, 90),
        "p95_m": weighted_percentile(distances, weights, 95),
        "mean_m": float(np.average(distances, weights=weights)),
        "within_m": {str(limit): float(np.average(distances <= limit, weights=weights))
                      for limit in THRESHOLDS_M},
    }


def intersection_exposure(lines, feature_geometries):
    if not feature_geometries:
        return {"status": "no_features_observed", "crossing_count": None,
                "overlap_length_m": None}
    tree = STRtree(feature_geometries)
    crossing_positions = []
    overlap_length = 0.0

    def collect_points(geometry, line):
        nonlocal overlap_length
        if geometry.is_empty:
            return
        if geometry.geom_type == "Point":
            crossing_positions.append(float(line.project(geometry)))
        elif geometry.geom_type == "MultiPoint":
            for point in geometry.geoms:
                crossing_positions.append(float(line.project(point)))
        elif geometry.geom_type == "LineString":
            overlap_length += float(geometry.length)
        elif geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
            for part in geometry.geoms:
                collect_points(part, line)

    for route in lines:
        indexes = tree.query(route, predicate="intersects")
        for index in indexes:
            collect_points(route.intersection(feature_geometries[int(index)]), route)
    crossing_positions.sort()
    unique_crossings = []
    for position in crossing_positions:
        if not unique_crossings or position - unique_crossings[-1] > 15.0:
            unique_crossings.append(position)
    return {"status": "measured_on_observed_features_only",
            "crossing_count": len(unique_crossings),
            "overlap_length_m": overlap_length}


def feature_types(features, tag_name):
    counts = Counter(str(tags.get(tag_name)) for _, tags in features
                     if isinstance(tags, dict) and tags.get(tag_name) is not None)
    return dict(counts.most_common(12))


def cache_details(path: Path, layer_meta: dict, tag_name: str, raw) -> dict:
    details = {
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "mtime_utc": (datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                      if path.is_file() else None),
        "sha256": sha256(path) if path.is_file() else None,
        "loader_metadata": layer_meta,
        "element_type_counts": {},
        "generator": raw.get("generator") if isinstance(raw, dict) else None,
        "osm3s": raw.get("osm3s") if isinstance(raw, dict) else None,
        "snapshot_time_status": "unknown_not_recorded_in_cache",
    }
    return details


def line_bounds(lines):
    bounds = [line.bounds for line in lines]
    return (min(x[0] for x in bounds), min(x[1] for x in bounds),
            max(x[2] for x in bounds), max(x[3] for x in bounds))


def save_case_map(case_id, reference, route_records, osm_for_plot, out_dir):
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    style = {"roads": ("#9aa0a6", 0.45, 0.22),
             "water": ("#277da1", 0.8, 0.48),
             "railways": ("#b5179e", 1.0, 0.7)}
    for name, features in osm_for_plot.items():
        color, width, alpha = style[name]
        for geometry, _ in features:
            if geometry.geom_type == "LineString":
                x, y = geometry.xy
                ax.plot(x, y, color=color, linewidth=width, alpha=alpha,
                        label=name if not any(line.get_label() == name for line in ax.lines) else "_nolegend_")
    for record in route_records:
        color = COLORS.get(record["method"], "#577590")
        linestyle = "--" if record["arm"] == "median" else "-"
        for line in record["wgs84_lines"]:
            x, y = line.xy
            ax.plot(x, y, color=color, linewidth=1.5, alpha=0.72,
                    linestyle=linestyle, label=record["method"] if not any(
                        item.get_label() == record["method"] for item in ax.lines) else "_nolegend_")
    for line in line_parts(reference):
        x, y = line.xy
        ax.plot(x, y, color="#d62828", linewidth=2.3, label="reference" if not any(
            item.get_label() == "reference" for item in ax.lines) else "_nolegend_")
    all_lines = [reference] + [line for record in route_records for line in record["wgs84_lines"]]
    min_x, min_y, max_x, max_y = line_bounds([part for item in all_lines for part in line_parts(item)])
    pad_x = max((max_x - min_x) * 0.08, 0.005)
    pad_y = max((max_y - min_y) * 0.08, 0.005)
    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)
    ax.set_aspect(1 / max(math.cos(math.radians((min_y + max_y) / 2)), 0.01))
    ax.set_xlabel("Longitude (WGS84)")
    ax.set_ylabel("Latitude (WGS84)")
    ax.set_title(f"{case_id}: frozen candidates, reference and cached OSM features\n"
                 "OSM snapshot date unknown; infrastructure exposure is diagnostic only")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.savefig(out_dir / f"{case_id}_osm_exposure.png", dpi=160)
    plt.close(fig)


def validate_candidates(results_dir: Path):
    summary_path = results_dir / "run_summary.json"
    freeze_path = results_dir / "candidate_freeze.json"
    summary = read_json(summary_path)
    freeze = read_json(freeze_path)
    if summary.get("status") != "complete":
        raise ValueError("source run is not complete")
    if summary.get("reference_geometry_read") is not False:
        raise ValueError("source run does not confirm reference isolation")
    if freeze.get("reference_geometry_read") is not False:
        raise ValueError("candidate freeze does not confirm reference isolation")
    if sha256(freeze_path) != summary.get("candidate_freeze_sha256"):
        raise ValueError("candidate freeze hash mismatch")
    if freeze.get("plan_sha256") != summary.get("plan_sha256"):
        raise ValueError("plan hash mismatch between run and candidate freeze")
    if any(run.get("returncode") != 0 or run.get("launch_error")
           for run in summary.get("runs", [])):
        raise ValueError("one or more source runs failed")
    for run in summary.get("runs", []):
        if any(str(arg).lower().startswith(("--real", "--reference"))
               for arg in run.get("argv", [])):
            raise ValueError("reference argument found in planning command")

    candidates = [item for item in freeze.get("candidates", [])
                  if item.get("stage") == "validated" and item.get("status") == "present"]
    if len(candidates) != 18:
        raise ValueError(f"expected 18 validated routes, found {len(candidates)}")
    for item in candidates:
        artifact = (results_dir / item["artifact_path_from_results_root"]).resolve()
        if not artifact.is_relative_to(results_dir.resolve()):
            raise ValueError("candidate path escapes results directory")
        if sha256(artifact) != item.get("artifact_sha256"):
            raise ValueError(f"candidate artifact hash mismatch: {artifact}")
        item["_verified_path"] = str(artifact)
    return summary, freeze, candidates


def read_references(reference_dir: Path, score_csv: Path, cases):
    scored_reference_km = {}
    with score_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            case_id = row.get("case_id") or row.get("\ufeffcase_id")
            if case_id and row.get("reference_km"):
                scored_reference_km[case_id] = float(row["reference_km"])
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
    output = {}
    for case_id in cases:
        shp = reference_dir / f"{case_id}_real_path.shp"
        if not shp.is_file():
            raise FileNotFoundError(shp)
        component_hashes = {}
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            part = shp.with_suffix(suffix)
            if part.is_file():
                component_hashes[part.name] = sha256(part)
        coordinates = load_real_path(str(shp), require_single_line=True, raise_errors=True)
        geometry = merge_lines([LineString(coords) for coords in coordinates])
        lines = line_parts(geometry)
        length_km = sum(projected(line, transformer).length for line in lines) / 1000.0
        expected = scored_reference_km.get(case_id)
        if expected is None or abs(length_km - expected) > 0.02:
            raise ValueError(f"reference length does not match frozen posthoc scores: {case_id} {length_km} vs {expected}")
        output[case_id] = {"geometry": geometry, "lines": lines,
                           "source": str(shp), "component_sha256": component_hashes,
                           "length_km": length_km,
                           "expected_scored_length_km": expected}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--score-csv", type=Path, required=True)
    parser.add_argument("--osm-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {out_dir}")

    summary, freeze, candidates = validate_candidates(args.results_dir.resolve())
    print(f"candidate_freeze_verified=true validated_routes={len(candidates)} reference_geometry_read=false")

    # Reference data is deliberately loaded only after every planned artifact is verified.
    references = read_references(args.reference_dir.resolve(), args.score_csv.resolve(), CASE_IDS)
    out_dir.mkdir(parents=True, exist_ok=False)
    cache_dir = args.osm_cache_dir.resolve()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
    rows = []
    case_results = {}

    for case_id in CASE_IDS:
        ref = references[case_id]
        planned = []
        for candidate in candidates:
            if candidate["case_id"] != case_id:
                continue
            geometry = line_from_geojson(Path(candidate["_verified_path"]))
            planned.append({**candidate, "geometry": geometry,
                            "wgs84_lines": line_parts(geometry),
                            "projected_lines": [projected(line, transformer)
                                                for line in line_parts(geometry)]})
        all_wgs84_lines = [line for record in planned for line in record["wgs84_lines"]] + ref["lines"]
        min_lon, min_lat, max_lon, max_lat = line_bounds(all_wgs84_lines)
        bbox = (min_lon - 0.02, min_lat - 0.02, max_lon + 0.02, max_lat + 0.02)
        layer_features = {}
        layer_manifest = {}
        for layer, (filename, tag_name) in LAYERS.items():
            cache_path = cache_dir / filename
            features, metadata = load_osm_geoms_with_status(cache_path, "line", bbox)
            osm_module = __import__("transmission_path_planner.osm", fromlist=["_PKL_CACHE"])
            raw = getattr(osm_module, "_PKL_CACHE", {}).get(cache_path)
            details = cache_details(cache_path, metadata, tag_name, raw)
            details["element_type_counts"] = feature_types(features, tag_name)
            details["coverage_claim"] = "observed feature extent/count only; completeness unknown"
            layer_manifest[layer] = details
            layer_features[layer] = features

        projected_features = {
            layer: [projected(geometry, transformer) for geometry, _ in features]
            for layer, features in layer_features.items()
        }
        for record in planned:
            row = {"case_id": case_id, "line_type": "planned",
                   "route_id": record["artifact_path_from_results_root"],
                   "method": record["method"], "arm": record["arm"], "stage": record["stage"],
                   "candidate_sha256": record["artifact_sha256"],
                   "length_km": sum(line.length for line in record["projected_lines"]) / 1000.0}
            for layer in LAYERS:
                exposure = nearest_exposure(record["projected_lines"], projected_features[layer])
                crossings = intersection_exposure(record["projected_lines"], projected_features[layer])
                row[f"{layer}_distance_p50_m"] = exposure["p50_m"]
                row[f"{layer}_distance_p95_m"] = exposure["p95_m"]
                row[f"{layer}_within_100m"] = exposure["within_m"]["100"]
                row[f"{layer}_within_500m"] = exposure["within_m"]["500"]
                row[f"{layer}_crossings"] = crossings["crossing_count"]
                row[f"{layer}_overlap_length_m"] = crossings["overlap_length_m"]
            rows.append(row)

        ref_lines = [projected(line, transformer) for line in ref["lines"]]
        row = {"case_id": case_id, "line_type": "reference", "route_id": case_id,
               "method": "posthoc_reference", "arm": "", "stage": "reference",
               "candidate_sha256": None,
               "length_km": sum(line.length for line in ref_lines) / 1000.0}
        for layer in LAYERS:
            exposure = nearest_exposure(ref_lines, projected_features[layer])
            crossings = intersection_exposure(ref_lines, projected_features[layer])
            row[f"{layer}_distance_p50_m"] = exposure["p50_m"]
            row[f"{layer}_distance_p95_m"] = exposure["p95_m"]
            row[f"{layer}_within_100m"] = exposure["within_m"]["100"]
            row[f"{layer}_within_500m"] = exposure["within_m"]["500"]
            row[f"{layer}_crossings"] = crossings["crossing_count"]
            row[f"{layer}_overlap_length_m"] = crossings["overlap_length_m"]
        rows.append(row)

        case_results[case_id] = {
            "request_bbox_wgs84": bbox,
            "reference": {key: value for key, value in ref.items()
                          if key not in {"geometry", "lines"}},
            "osm_layers": layer_manifest,
        }
        save_case_map(case_id, ref["geometry"], planned, layer_features, out_dir)
        print(f"case={case_id} planned={len(planned)} OSM_counts=" +
              ",".join(f"{key}:{layer_manifest[key]['loader_metadata']['geometry_count_in_bbox']}"
                       for key in LAYERS))

    fields = list(rows[0])
    with (out_dir / "route_exposure.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_json = {
        "experiment_id": out_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_status": summary.get("status"),
        "source_run_summary_sha256": sha256(args.results_dir.resolve() / "run_summary.json"),
        "candidate_freeze_sha256": sha256(args.results_dir.resolve() / "candidate_freeze.json"),
        "reference_geometry_read_after_all_candidate_hashes_verified": True,
        "validated_candidate_count": len(candidates),
        "candidate_hash_failures": 0,
        "reference_sources": {case_id: references[case_id]["component_sha256"]
                              for case_id in CASE_IDS},
        "osm_cache_sources": case_results,
        "metrics": {
            "sample_step_m": 50,
            "distance_percentiles": [50, 90, 95],
            "near_feature_thresholds_m": list(THRESHOLDS_M),
            "intersection_dedup_chainage_tolerance_m": 15,
            "interpretation": "Exposures describe observed OSM cache features only; unknown cache date and incomplete OSM coverage prevent treating missing features as absence on the ground.",
        },
        "route_exposure_csv_sha256": None,
    }
    csv_path = out_dir / "route_exposure.csv"
    summary_json["route_exposure_csv_sha256"] = sha256(csv_path)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={out_dir} csv_sha256={summary_json['route_exposure_csv_sha256']}")


if __name__ == "__main__":
    main()
