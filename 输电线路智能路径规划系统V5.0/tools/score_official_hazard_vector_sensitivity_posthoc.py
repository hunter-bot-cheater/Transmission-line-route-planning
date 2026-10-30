#!/usr/bin/env python3
"""Hash-gated post-hoc comparison for the official SHP/MVT hazard experiment.

Planning is complete before this tool reads any reference geometry. It verifies
both candidate freezes and all candidate files, hashes the reference source set,
records those hashes, then reads the shapefiles and calculates 5 m metrics.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import inspect
import json
import math
import os
import re
import statistics
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


CASE_IDS = tuple(f"B12-{index:02d}" for index in range(1, 13))
CASE_RE = re.compile(r"^B12-\d{2}$")
ARMS = ("mvt_b12_v2", "shp_b12_v2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def canonical_geometry_hash(geometry: dict) -> str:
    kind = geometry.get("type")
    raw = geometry.get("coordinates")
    if kind == "LineString":
        lines = [raw]
    elif kind == "MultiLineString":
        lines = raw
    else:
        raise ValueError(f"unsupported candidate geometry type: {kind!r}")
    normalized = []
    for line in lines:
        points = [[round(float(value), 9) for value in point]
                  for point in line]
        if len(points) < 2:
            raise ValueError("candidate line has fewer than two coordinates")
        forward = json.dumps(points, separators=(",", ":"), allow_nan=False)
        reverse = json.dumps(list(reversed(points)), separators=(",", ":"),
                             allow_nan=False)
        normalized.append(min(forward, reverse))
    normalized.sort()
    payload = {"type": kind, "lines": normalized}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_runs(experiment_root: Path) -> tuple[dict, dict]:
    verified = {}
    for arm in ARMS:
        summary_path = experiment_root / f"results_{arm}" / "run_summary.json"
        provenance_path = experiment_root / f"results_{arm}" / "run_provenance.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (summary.get("complete") is not True
                or summary.get("expected_cases") != 12
                or summary.get("completed_cases") != 12
                or summary.get("failed_cases")
                or summary.get("no_route_cases")):
            raise ValueError(f"{arm} run is incomplete")
        evaluations = summary.get("real_evaluation_by_case", [])
        if len(evaluations) != 12 or any(
                item.get("status") != "not_requested" for item in evaluations):
            raise ValueError(f"{arm} run requested reference evaluation")
        argv = summary.get("argv", [])
        if any(isinstance(item, str) and item.startswith(("--real", "--reference"))
               for item in argv):
            raise ValueError(f"{arm} planning argv contains reference flags")
        if provenance.get("configuration", {}).get("real_shp") is not None:
            raise ValueError(f"{arm} provenance has a real SHP configured")
        if provenance.get("reference_geometry_policy", {}).get(
                "hashed_by_provenance") is not False:
            raise ValueError(f"{arm} provenance does not confirm reference isolation")
        if (experiment_root / f"results_{arm}" / "exit_status.txt").read_text(
                encoding="utf-8").strip() != "0":
            raise ValueError(f"{arm} run did not exit with status 0")
        verified[arm] = {
            "summary": summary,
            "summary_path": summary_path,
            "provenance_path": provenance_path,
            "summary_sha256": sha256(summary_path),
            "provenance_sha256": sha256(provenance_path),
        }
    if len(verified[ARMS[0]]["summary"].get("argv", [])) == 0:
        raise ValueError("planning argv is missing")
    return verified[ARMS[0]], verified[ARMS[1]]


def verify_candidate_freezes(experiment_root: Path) -> tuple[dict, dict]:
    freezes = {}
    for arm in ARMS:
        path = experiment_root / f"candidate_freeze_{arm}_20261030.json"
        freeze = json.loads(path.read_text(encoding="utf-8"))
        if (freeze.get("candidate_count") != 12
                or freeze.get("candidate_routes_frozen_before_reference_geometry_read") is not True
                or freeze.get("reference_geometry_read") is not False
                or freeze.get("run_complete") is not True):
            raise ValueError(f"{arm} candidate freeze is incomplete or not isolated")
        if len(freeze.get("candidates", [])) != 12:
            raise ValueError(f"{arm} candidate manifest does not have 12 rows")
        seen = set()
        for item in freeze["candidates"]:
            case = item.get("case_id")
            if not isinstance(case, str) or not CASE_RE.match(case) or case in seen:
                raise ValueError(f"invalid or duplicate candidate case in {arm}: {case!r}")
            seen.add(case)
            route_path = experiment_root / item["path"]
            if not route_path.is_file() or sha256(route_path) != item["file_sha256"]:
                raise ValueError(f"frozen candidate file hash mismatch: {route_path}")
            document = json.loads(route_path.read_text(encoding="utf-8"))
            features = document.get("features", [])
            if len(features) != 1 or not features[0].get("geometry"):
                raise ValueError(f"expected one candidate line in {route_path}")
            if (not route_path.parent.name.startswith(case + "-")
                    or features[0].get("properties", {}).get("case") != case):
                raise ValueError(f"candidate case ID mismatch in {route_path}")
            if canonical_geometry_hash(features[0]["geometry"]) != item[
                    "canonical_geometry_sha256_9dp_direction_invariant"]:
                raise ValueError(f"frozen candidate geometry hash mismatch: {route_path}")
        if seen != set(CASE_IDS):
            raise ValueError(f"{arm} candidate IDs differ from B12-01..B12-12")
        supporting = freeze.get("supporting_file_sha256", {})
        for relative, expected in supporting.items():
            source = experiment_root / relative
            if not source.is_file() or sha256(source) != expected:
                raise ValueError(f"frozen supporting input changed: {source}")
        freezes[arm] = {"path": path, "document": freeze, "sha256": sha256(path)}
    case_sets = [
        {item["case_id"] for item in freezes[arm]["document"]["candidates"]}
        for arm in ARMS
    ]
    if case_sets[0] != case_sets[1]:
        raise ValueError("MVT and SHP candidate case sets differ")
    return freezes[ARMS[0]], freezes[ARMS[1]]


def normalized_name(value: str) -> str:
    return "".join(char.casefold() for char in unicodedata.normalize("NFKC", value)
                   if char.isalnum())


def haversine_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    delta_lon, delta_lat = lon2 - lon1, lat2 - lat1
    value = (math.sin(delta_lat / 2) ** 2
             + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2)
    return 6_371_008.8 * 2 * math.asin(min(1.0, math.sqrt(value)))


def reference_map(experiment_root: Path, reference_dir: Path) -> tuple[dict, list[dict]]:
    input_dir = experiment_root / "inputs_benchmark12_v2"
    with (input_dir / "stations.csv").open(encoding="utf-8-sig", newline="") as stream:
        stations = {row["station_id"]: row for row in csv.DictReader(stream)}
    with (input_dir / "route_tasks.csv").open(encoding="utf-8-sig", newline="") as stream:
        tasks = list(csv.DictReader(stream))
    with (reference_dir / "real_cases.csv").open(encoding="utf-8-sig", newline="") as stream:
        references = list(csv.DictReader(stream))
    if {row["route_id"] for row in tasks} != set(CASE_IDS):
        raise ValueError("route_tasks.csv case IDs are not exactly B12-01..B12-12")
    shapefiles = list(reference_dir.glob("*_real_path.shp"))
    shapefiles_by_name = {}
    for path in shapefiles:
        stem = path.stem.removesuffix("_real_path")
        key = normalized_name(stem)
        shapefiles_by_name.setdefault(key, []).append(path)
    mapping = {}
    source_records = []
    for task in tasks:
        case = task["route_id"]
        start_row, end_row = stations[task["from_station_id"]], stations[task["to_station_id"]]
        start = (float(start_row["lon"]), float(start_row["lat"]))
        end = (float(end_row["lon"]), float(end_row["lat"]))
        matches = []
        for reference in references:
            ref_start = (float(reference["start_lon"]), float(reference["start_lat"]))
            ref_end = (float(reference["end_lon"]), float(reference["end_lat"]))
            forward = (haversine_m(start, ref_start), haversine_m(end, ref_end))
            reverse = (haversine_m(start, ref_end), haversine_m(end, ref_start))
            chosen = forward if max(forward) <= max(reverse) else reverse
            if max(chosen) <= 5.0:
                matches.append((reference, chosen))
        if len(matches) != 1:
            raise ValueError(f"cannot uniquely match {case} endpoints to real_cases.csv")
        reference, endpoint_deltas = matches[0]
        paths = shapefiles_by_name.get(normalized_name(reference["name"]), [])
        if len(paths) != 1:
            raise ValueError(f"cannot uniquely resolve reference SHP for {case}: {reference['name']}")
        mapping[case] = paths[0]
        source_records.append({
            "case_id": case,
            "reference_name": reference["name"],
            "reference_shapefile": str(paths[0].resolve()),
            "endpoint_match_max_delta_m": max(endpoint_deltas),
        })
    if len(mapping) != 12 or len(set(mapping.values())) != 12:
        raise ValueError("reference mapping is incomplete or reuses a shapefile")
    return mapping, source_records


def reference_file_hashes(mapping: dict[str, Path]) -> dict[str, str]:
    hashes = {}
    for path in sorted(set(mapping.values())):
        for suffix in (".shp", ".shx", ".dbf", ".prj"):
            component = path.with_suffix(suffix)
            if not component.is_file():
                raise ValueError(f"reference shapefile component is missing: {component}")
            hashes[str(component.resolve())] = sha256(component)
        component = path.with_suffix(".cpg")
        if component.is_file():
            hashes[str(component.resolve())] = sha256(component)
    return hashes


def geometry_lines(geometry) -> list[list[tuple[float, float]]]:
    if geometry.geom_type == "LineString":
        parts = [geometry]
    elif geometry.geom_type == "MultiLineString":
        parts = list(geometry.geoms)
    else:
        raise ValueError(f"expected line geometry, got {geometry.geom_type}")
    return [[(float(point[0]), float(point[1])) for point in part.coords]
            for part in parts if part.length > 0]


def read_candidate(path: Path):
    from shapely.geometry import shape

    document = json.loads(path.read_text(encoding="utf-8"))
    feature = document["features"][0]
    return shape(feature["geometry"])


def read_reference(path: Path) -> tuple[object, list[list[tuple[float, float]]]]:
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import unary_union

    frame = gpd.read_file(path)
    if frame.empty or frame.crs is None:
        raise ValueError(f"reference SHP is empty or missing CRS: {path}")
    if not frame.crs.equals("EPSG:4326"):
        frame = frame.to_crs("EPSG:4326")
    parts = []
    for geometry in frame.geometry:
        if geometry is None:
            continue
        if geometry.geom_type == "LineString":
            parts.append(geometry)
        elif geometry.geom_type == "MultiLineString":
            parts.extend(geometry.geoms)
        else:
            raise ValueError(f"reference SHP contains {geometry.geom_type}: {path}")
    parts = [part for part in parts if part.length > 0]
    if not parts:
        raise ValueError(f"reference SHP contains no line geometry: {path}")
    merged = unary_union(parts)
    if merged.geom_type == "LineString":
        lines = [merged]
    elif merged.geom_type == "MultiLineString":
        lines = list(merged.geoms)
    else:
        raise ValueError(f"reference union has unexpected geometry: {merged.geom_type}")
    return merged, geometry_lines(MultiLineString(lines) if len(lines) > 1 else lines[0])


def metric_row(case: str, arm: str, similarity: dict) -> dict:
    planned = similarity["planned_to_reference"]
    reference = similarity["reference_to_planned"]
    symmetric = similarity["symmetric"]
    length = similarity["length"]
    endpoints = similarity.get("endpoint_offsets") or {}
    frechet = similarity.get("discrete_frechet_approx") or {}
    return {
        "case_id": case,
        "arm": arm,
        "sample_interval_m": similarity["sampling_interval_m"],
        "planned_coverage_100m_pct": 100 * planned["coverage_within_m"]["100"],
        "reference_coverage_100m_pct": 100 * reference["coverage_within_m"]["100"],
        "coverage_f1_100m_pct": 100 * symmetric["coverage_f1_within_m"]["100"],
        "planned_p95_m": planned["p95_m"],
        "reference_p95_m": reference["p95_m"],
        "symmetric_p95_m": symmetric["p95_m"],
        "planned_p99_m": planned["p99_m"],
        "reference_p99_m": reference["p99_m"],
        "sampled_hausdorff_m": symmetric["sampled_hausdorff_m"],
        "length_delta_pct": length["absolute_difference_pct"],
        "planned_km": length["planned_km"],
        "reference_km": length["reference_km"],
        "start_endpoint_offset_m": endpoints.get("planned_start_to_reference_endpoint_m"),
        "end_endpoint_offset_m": endpoints.get("planned_end_to_reference_endpoint_m"),
        "discrete_frechet_m": frechet.get("distance_m"),
    }


def project_geometry(geometry, crs: str):
    from pyproj import Transformer
    from shapely.ops import transform

    project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
    return transform(project, geometry)


def planning_argument_comparison(mvt_argv: list[str], shp_argv: list[str]) -> dict:
    def normalize(argv: list[str]) -> list[str]:
        output = []
        index = 0
        while index < len(argv):
            item = argv[index]
            if item == "--out-dir":
                output.extend((item, "<arm-output-directory>"))
                index += 2
            elif item in ("--official-hazards-dir", "--official-hazard-vector-dir"):
                output.extend(("<official-hazard-source-flag>", "<official-hazard-source-path>"))
                index += 2
            else:
                output.append(item)
                index += 1
        return output

    mvt_normalized, shp_normalized = normalize(mvt_argv), normalize(shp_argv)
    return {
        "non_source_arguments_match": mvt_normalized == shp_normalized,
        "normalized_unified_diff": list(difflib.unified_diff(
            mvt_normalized, shp_normalized, fromfile="MVT", tofile="SHP", lineterm="")),
        "hazard_source_args": {
            "mvt": [item for item in mvt_argv
                    if item in ("--official-hazards-dir", "--official-hazard-vector-dir")],
            "shp": [item for item in shp_argv
                    if item in ("--official-hazards-dir", "--official-hazard-vector-dir")],
        },
    }


def draw_route(ax, geometry, **kwargs) -> None:
    parts = [geometry] if geometry.geom_type == "LineString" else list(geometry.geoms)
    for part in parts:
        coords = list(part.coords)
        ax.plot([point[0] / 1000 for point in coords],
                [point[1] / 1000 for point in coords], **kwargs)


def draw_colored_error(ax, source, target, interval_m: float, norm):
    import numpy as np
    from matplotlib.collections import LineCollection
    from shapely.geometry import Point

    parts = [source] if source.geom_type == "LineString" else list(source.geoms)
    segments, distances = [], []
    target_parts = [target] if target.geom_type == "LineString" else list(target.geoms)
    for part in parts:
        count = max(1, int(math.ceil(part.length / interval_m)))
        measures = np.linspace(0.0, part.length, count + 1)
        coords = [part.interpolate(float(distance)).coords[0] for distance in measures]
        segments.extend([[(a[0] / 1000, a[1] / 1000),
                          (b[0] / 1000, b[1] / 1000)]
                         for a, b in zip(coords, coords[1:])])
        distances.extend(target.distance(
            Point((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)) / 1000
            for a, b in zip(coords, coords[1:]))
    collection = LineCollection(segments, cmap="turbo", norm=norm, linewidth=2.1)
    collection.set_array(np.asarray(distances))
    ax.add_collection(collection)
    for part in target_parts:
        coords = list(part.coords)
        ax.plot([point[0] / 1000 for point in coords],
                [point[1] / 1000 for point in coords],
                color="#30343b", linestyle="--", linewidth=1.15)
    return collection


def set_equal_limits(ax, geometries) -> None:
    bounds = [geometry.bounds for geometry in geometries]
    min_x = min(item[0] for item in bounds)
    min_y = min(item[1] for item in bounds)
    max_x = max(item[2] for item in bounds)
    max_y = max(item[3] for item in bounds)
    padding = max(max_x - min_x, max_y - min_y, 1000) * 0.04
    ax.set_xlim((min_x - padding) / 1000, (max_x + padding) / 1000)
    ax.set_ylim((min_y - padding) / 1000, (max_y + padding) / 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xlabel("Local east (km)")
    ax.set_ylabel("Local north (km)")


def render_maps(output_dir: Path, scored: dict,
                color_interval_m: float) -> list[Path]:
    os.environ["MPLCONFIGDIR"] = str(output_dir / "matplotlib-config")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    cases = sorted(scored[ARMS[0]])
    p95_values = [scored[arm][case]["row"]["symmetric_p95_m"] / 1000
                  for arm in ARMS for case in cases]
    vmax = max(1.0, min(float(np.percentile(p95_values, 90)), 50.0))
    norm = Normalize(vmin=0, vmax=vmax)
    output_files = []

    fig, axes = plt.subplots(4, 3, figsize=(15, 16), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        record = scored[ARMS[0]][case]
        crs = record["similarity"]["comparison_crs"]
        reference = project_geometry(record["reference_geometry"], crs)
        mvt = project_geometry(record["planned_geometry"], crs)
        shp = project_geometry(scored[ARMS[1]][case]["planned_geometry"], crs)
        draw_route(ax, reference, color="#30343b", linestyle="--", linewidth=1.35)
        draw_route(ax, mvt, color="#2677b8", linewidth=1.35)
        draw_route(ax, shp, color="#e0782f", linewidth=1.35)
        set_equal_limits(ax, [reference, mvt, shp])
        rows = {arm: scored[arm][case]["row"] for arm in ARMS}
        ax.set_title(
            f"{case} | MVT F1={rows[ARMS[0]]['coverage_f1_100m_pct']:.1f}% "
            f"P95={rows[ARMS[0]]['symmetric_p95_m']/1000:.1f}km\n"
            f"SHP F1={rows[ARMS[1]]['coverage_f1_100m_pct']:.1f}% "
            f"P95={rows[ARMS[1]]['symmetric_p95_m']/1000:.1f}km")
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], color="#30343b", linestyle="--", label="Reference line"),
        Line2D([0], [0], color="#2677b8", label="MVT hazard route"),
        Line2D([0], [0], color="#e0782f", label="SHP hazard route"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3)
    fig.suptitle("12-case route geometry overlay | MVT vs SHP hazard representation",
                 fontsize=15)
    overview = output_dir / "12_route_overlay_mvt_vs_shp.png"
    fig.savefig(overview, dpi=125)
    plt.close(fig)
    output_files.append(overview)

    for case in cases:
        record = scored[ARMS[0]][case]
        crs = record["similarity"]["comparison_crs"]
        reference = project_geometry(record["reference_geometry"], crs)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        panels = ((axes[0, 0], ARMS[0], False), (axes[0, 1], ARMS[0], True),
                  (axes[1, 0], ARMS[1], False), (axes[1, 1], ARMS[1], True))
        for ax, arm, reverse in panels:
            planned = project_geometry(scored[arm][case]["planned_geometry"], crs)
            source, target = (reference, planned) if reverse else (planned, reference)
            draw_colored_error(ax, source, target, color_interval_m, norm)
            set_equal_limits(ax, [planned, reference])
            row = scored[arm][case]["row"]
            direction = "reference→planned" if reverse else "planned→reference"
            coverage = (row["reference_coverage_100m_pct"] if reverse
                        else row["planned_coverage_100m_pct"])
            p95 = row["reference_p95_m"] if reverse else row["planned_p95_m"]
            ax.set_title(
                f"{arm.split('_')[0].upper()} | {direction}: "
                f"within 100m {coverage:.1f}% · P95 {p95/1000:.2f}km")
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="turbo"), ax=axes,
                     extend="max",
                     label=(f"Distance to target line (km); samples every {color_interval_m:g} m; "
                            f"values above {norm.vmax:.1f} km saturated"))
        fig.suptitle(
            f"{case} | distance-colored in both directions; 100m coverage is descriptive")
        output = output_dir / f"{case}_error_map.png"
        fig.savefig(output, dpi=145)
        plt.close(fig)
        output_files.append(output)
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--planner-root", type=Path, required=True,
                        help="frozen execution bundle containing transmission_path_planner")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-interval-m", type=float, default=5.0)
    parser.add_argument("--map-color-interval-m", type=float, default=100.0)
    args = parser.parse_args()
    experiment_root = args.experiment_root.resolve()
    reference_dir = args.reference_dir.resolve()
    planner_root = args.planner_root.resolve()
    output_dir = args.out_dir.resolve()
    allowed_root = Path("/data1/B24041626/pp").resolve()
    for label, path in (("experiment", experiment_root), ("reference", reference_dir),
                        ("planner bundle", planner_root), ("output", output_dir)):
        try:
            path.relative_to(allowed_root)
        except ValueError:
            raise ValueError(f"{label} path must stay under {allowed_root}: {path}") from None
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output: {output_dir}; use a new output path after any failed partial run")
    if not math.isfinite(args.sample_interval_m) or not 1 <= args.sample_interval_m <= 1000:
        raise ValueError("sample interval must be in [1, 1000] meters")
    if not math.isfinite(args.map_color_interval_m) or args.map_color_interval_m <= 0:
        raise ValueError("map color interval must be positive")

    mvt_run, shp_run = verify_runs(experiment_root)
    mvt_freeze, shp_freeze = verify_candidate_freezes(experiment_root)
    freezes = {ARMS[0]: mvt_freeze, ARMS[1]: shp_freeze}
    reference_paths, reference_records = reference_map(experiment_root, reference_dir)
    source_hashes = reference_file_hashes(reference_paths)
    source_case_csv = reference_dir / "real_cases.csv"
    input_manifest_path = experiment_root / "inputs_benchmark12_v2" / "input_manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    source_csv_hash = sha256(source_case_csv)
    stations_hash = sha256(experiment_root / "inputs_benchmark12_v2" / "stations.csv")
    tasks_hash = sha256(experiment_root / "inputs_benchmark12_v2" / "route_tasks.csv")
    if (source_csv_hash != input_manifest.get("source_csv_sha256")
            or stations_hash != input_manifest.get("stations_csv_sha256")
            or tasks_hash != input_manifest.get("route_tasks_csv_sha256")):
        raise ValueError("benchmark endpoints/tasks differ from the frozen input manifest")
    argument_comparison = planning_argument_comparison(
        mvt_run["summary"].get("argv", []), shp_run["summary"].get("argv", []))
    pre_read_manifest = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_geometry_loaded": False,
        "candidate_freeze_sha256": {
            arm: freezes[arm]["sha256"] for arm in ARMS
        },
        "candidate_count_per_arm": {
            arm: freezes[arm]["document"]["candidate_count"] for arm in ARMS
        },
        "run_summary_sha256": {
            ARMS[0]: mvt_run["summary_sha256"],
            ARMS[1]: shp_run["summary_sha256"],
        },
        "run_provenance_sha256": {
            ARMS[0]: mvt_run["provenance_sha256"],
            ARMS[1]: shp_run["provenance_sha256"],
        },
        "planning_argv": {
            ARMS[0]: mvt_run["summary"].get("argv", []),
            ARMS[1]: shp_run["summary"].get("argv", []),
        },
        "planning_argument_comparison": argument_comparison,
        "reference_case_mapping": reference_records,
        "reference_source_file_sha256": source_hashes,
        "reference_case_index_sha256": source_csv_hash,
        "frozen_endpoint_input_sha256": {
            "input_manifest.json": sha256(input_manifest_path),
            "stations.csv": stations_hash,
            "route_tasks.csv": tasks_hash,
            "real_cases.csv": source_csv_hash,
        },
        "planner_metrics_source_sha256": sha256(
            planner_root / "transmission_path_planner" / "metrics.py"),
        "posthoc_script_sha256": sha256(Path(__file__).resolve()),
        "sample_interval_m": args.sample_interval_m,
        "claim_scope": "post-hoc geometry similarity only; endpoints are historical and unverified",
    }
    output_dir.mkdir(parents=False, exist_ok=False)
    write_json_exclusive(output_dir / "reference_freeze_manifest.json", pre_read_manifest)

    for path_string, expected in source_hashes.items():
        path = Path(path_string)
        if sha256(path) != expected:
            raise ValueError(f"reference source changed after freeze: {path}")
    sys_path = str(planner_root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, sys_path)
    import importlib.metadata
    import transmission_path_planner.metrics as metrics_module
    supports_interval_argument = "sample_interval_m" in inspect.signature(
        metrics_module.compute_deviation).parameters
    if not supports_interval_argument:
        metrics_module.ROUTE_SAMPLE_STEP_M = args.sample_interval_m

    def compute_deviation(planned, reference, **options):
        options.setdefault("chain_profile_bins", 40)
        if supports_interval_argument:
            options["sample_interval_m"] = args.sample_interval_m
        result = metrics_module.compute_deviation(planned, reference, **options)
        actual = (result or {}).get("route_similarity", {}).get("sampling_interval_m")
        if actual != args.sample_interval_m:
            raise ValueError(f"metric sampler used {actual}, expected {args.sample_interval_m}")
        return result

    references = {}
    for case, path in reference_paths.items():
        geometry, lines = read_reference(path)
        references[case] = {"geometry": geometry, "lines": lines, "path": path}

    candidates_by_arm = {}
    planned_geometries = {arm: {} for arm in ARMS}
    for arm in ARMS:
        candidates_by_arm[arm] = {
            item["case_id"]: item for item in freezes[arm]["document"]["candidates"]
        }
        for case, item in candidates_by_arm[arm].items():
            geometry = read_candidate(experiment_root / item["path"])
            planned_geometries[arm][case] = geometry

    scored = {arm: {} for arm in ARMS}
    metric_rows = []
    for case in sorted(references):
        reference_lines = references[case]["lines"]
        reference_arg = reference_lines[0] if len(reference_lines) == 1 else reference_lines
        for arm in ARMS:
            geometry = planned_geometries[arm][case]
            planned_lines = geometry_lines(geometry)
            planned_arg = planned_lines[0] if len(planned_lines) == 1 else planned_lines
            result = compute_deviation(planned_arg, reference_arg)
            if not result or not result.get("route_similarity"):
                raise ValueError(f"metric computation failed: {arm}/{case}")
            similarity = result["route_similarity"]
            if similarity["sampling_interval_m"] != args.sample_interval_m:
                raise ValueError("metric sampler did not use the requested interval")
            row = metric_row(case, arm, similarity)
            scored[arm][case] = {
                "row": row,
                "similarity": similarity,
                "planned_geometry": geometry,
                "reference_geometry": references[case]["geometry"],
            }
            metric_rows.append({
                **row,
                "route_similarity_version": result["route_similarity_version"],
                "route_similarity": similarity,
                "reference_source": str(references[case]["path"]),
                "reference_name": next(item["reference_name"] for item in reference_records
                                        if item["case_id"] == case),
            })

    pairwise_rows = []
    for case in sorted(references):
        mvt_lines = geometry_lines(planned_geometries[ARMS[0]][case])
        shp_lines = geometry_lines(planned_geometries[ARMS[1]][case])
        mvt_arg = mvt_lines[0] if len(mvt_lines) == 1 else mvt_lines
        shp_arg = shp_lines[0] if len(shp_lines) == 1 else shp_lines
        pair = compute_deviation(mvt_arg, shp_arg)
        if not pair or not pair.get("route_similarity"):
            raise ValueError(f"candidate dispersion metric failed: {case}")
        pairwise_rows.append({
            "case_id": case,
            "sampling_interval_m": pair["route_similarity"]["sampling_interval_m"],
            "mvt_to_shp_coverage_100m_pct": 100 * pair["route_similarity"][
                "planned_to_reference"]["coverage_within_m"]["100"],
            "shp_to_mvt_coverage_100m_pct": 100 * pair["route_similarity"][
                "reference_to_planned"]["coverage_within_m"]["100"],
            "symmetric_p95_m": pair["route_similarity"]["symmetric"]["p95_m"],
            "symmetric_sampled_hausdorff_m": pair["route_similarity"][
                "symmetric"]["sampled_hausdorff_m"],
        })

    for arm in ARMS:
        path = experiment_root / f"candidate_freeze_{arm}_20261030.json"
        if sha256(path) != freezes[arm]["sha256"]:
            raise ValueError(f"candidate freeze changed during scoring: {arm}")
        for item in freezes[arm]["document"]["candidates"]:
            candidate = experiment_root / item["path"]
            if sha256(candidate) != item["file_sha256"]:
                raise ValueError(f"candidate changed during scoring: {candidate}")
    for path_string, expected in source_hashes.items():
        if sha256(Path(path_string)) != expected:
            raise ValueError(f"reference source changed during scoring: {path_string}")

    render_paths = render_maps(output_dir, scored, args.map_color_interval_m)
    overview_summary = {
        "candidate_count_per_arm": {arm: len(scored[arm]) for arm in ARMS},
        "median_f1_100m_pct": {
            arm: statistics.median(
                scored[arm][case]["row"]["coverage_f1_100m_pct"]
                for case in scored[arm])
            for arm in ARMS
        },
        "median_planned_coverage_100m_pct": {
            arm: statistics.median(
                scored[arm][case]["row"]["planned_coverage_100m_pct"]
                for case in scored[arm])
            for arm in ARMS
        },
        "median_reference_coverage_100m_pct": {
            arm: statistics.median(
                scored[arm][case]["row"]["reference_coverage_100m_pct"]
                for case in scored[arm])
            for arm in ARMS
        },
        "median_symmetric_p95_m": {
            arm: statistics.median(
                scored[arm][case]["row"]["symmetric_p95_m"]
                for case in scored[arm])
            for arm in ARMS
        },
        "median_symmetric_p99_m": {
            arm: statistics.median(
                max(scored[arm][case]["row"]["planned_p99_m"],
                    scored[arm][case]["row"]["reference_p99_m"])
                for case in scored[arm])
            for arm in ARMS
        },
        "median_mvt_shp_pairwise_symmetric_p95_m": statistics.median(
            row["symmetric_p95_m"] for row in pairwise_rows),
        "engineering_feasibility": "not_evaluated for all 24 candidate routes",
        "endpoint_qualification": "historical benchmark coordinates; unverified and shared with reference endpoints",
    }
    write_json_exclusive(output_dir / "route_similarity_posthoc.json", {
        "schema_version": 1,
        "metric_version": max(row["route_similarity_version"] for row in metric_rows),
        "sample_interval_m": args.sample_interval_m,
        "summary": overview_summary,
        "routes": metric_rows,
        "mvt_shp_pairwise_dispersion": pairwise_rows,
    })
    metric_columns = [key for key in metric_rows[0] if key != "route_similarity"]
    with (output_dir / "route_similarity_posthoc.csv").open(
            "x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in metric_columns}
                          for row in metric_rows])
    pair_columns = list(pairwise_rows[0])
    with (output_dir / "mvt_shp_pairwise_dispersion.csv").open(
            "x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_columns)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    source_root = planner_root / "transmission_path_planner"
    metric_source = source_root / "metrics.py"
    posthoc_manifest = {
        "schema_version": 1,
        "status": "complete",
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_geometry_read_after_candidate_hash_verification": True,
        "candidate_freeze_sha256": {
            arm: freezes[arm]["sha256"] for arm in ARMS
        },
        "run_summary_sha256": {
            ARMS[0]: mvt_run["summary_sha256"],
            ARMS[1]: shp_run["summary_sha256"],
        },
        "planning_argument_comparison": argument_comparison,
        "reference_freeze_manifest_sha256": sha256(
            output_dir / "reference_freeze_manifest.json"),
        "reference_source_file_sha256": source_hashes,
        "planner_metrics_source_sha256": sha256(metric_source),
        "route_similarity_metric_version": max(
            row["route_similarity_version"] for row in metric_rows),
        "metric_sampling_interval_m": args.sample_interval_m,
        "metric_sampling_control": (
            "explicit sample_interval_m argument" if supports_interval_argument
            else "server bundle ROUTE_SAMPLE_STEP_M constant overridden before scoring"),
        "map_color_interval_m": args.map_color_interval_m,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "shapely", "pyproj", "geopandas", "matplotlib")
        },
        "output_sha256": {
            path.name: sha256(path) for path in [
                *render_paths,
                output_dir / "route_similarity_posthoc.json",
                output_dir / "route_similarity_posthoc.csv",
                output_dir / "mvt_shp_pairwise_dispersion.csv",
            ]
        },
        "metrics_sha256": sha256(output_dir / "route_similarity_posthoc.json"),
        "pairwise_csv_sha256": sha256(output_dir / "mvt_shp_pairwise_dispersion.csv"),
        "engineering_feasibility": "not evaluated; planning run reports not_evaluated for all cases",
        "limitations": [
            "Benchmark route endpoints are historical and unverified, and were shared with the reference benchmark.",
            "The 100 m coverage values are sampled approximations at 5 m arc-length spacing.",
            "Loaded hazard masks and planner gates do not establish complete constraint coverage or engineering approval.",
            "The SHP/MVT comparison isolates source representation only if the recorded planning argv differs only in that source argument.",
        ],
    }
    summary_lines = [
        "# Official hazard SHP/MVT route sensitivity: post-hoc geometry comparison",
        "",
        f"- Candidates: 12 per arm, {len(metric_rows)} route/reference comparisons.",
        f"- Metric version: {posthoc_manifest['route_similarity_metric_version']}; "
        f"arc-length sampling: {args.sample_interval_m:g} m.",
        "- The two directional 100 m coverage values and their harmonic-mean F1 are "
        "descriptive spatial metrics; this report applies no coverage pass/fail gate.",
        f"- Both arms' non-source planning arguments match: "
        f"{argument_comparison['non_source_arguments_match']}.",
        "",
        "## Median route geometry results",
        "",
        "| Arm | planned→reference within 100 m | reference→planned within 100 m | "
        "F1 within 100 m | symmetric P95 | symmetric P99 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, label in ((ARMS[0], "MVT"), (ARMS[1], "SHP")):
        summary_lines.append(
            f"| {label} | {overview_summary['median_planned_coverage_100m_pct'][arm]:.2f}% "
            f"| {overview_summary['median_reference_coverage_100m_pct'][arm]:.2f}% "
            f"| {overview_summary['median_f1_100m_pct'][arm]:.2f}% "
            f"| {overview_summary['median_symmetric_p95_m'][arm]/1000:.2f} km "
            f"| {overview_summary['median_symmetric_p99_m'][arm]/1000:.2f} km |"
        )
    summary_lines.extend([
        "",
        f"- Median MVT↔SHP pairwise symmetric P95: "
        f"{overview_summary['median_mvt_shp_pairwise_symmetric_p95_m']/1000:.2f} km.",
        "- All route generation occurred on the server. Reference geometries were "
        "opened only after both candidate freezes and every route hash were verified.",
        "- The two arms use the same fixed, historical benchmark endpoints; those "
        "endpoints are unverified and share their source with the references.",
        "- All 24 routes are exploratory terrain-screening centerlines. Engineering "
        "feasibility is not evaluated; no tower-span, sag, clearance, or permit review "
        "is implied.",
        "- SHP/MVT is a hazard-source representation sensitivity test. It can show "
        "whether encoding/rasterization changes the chosen geometry; it does not by "
        "itself show an accuracy improvement.",
        "- The 5 m measurements are dense sampled approximations, not exact continuous "
        "Hausdorff bounds. Fréchet and endpoint metrics are unavailable for some "
        "disconnected geometries.",
        "",
        "## Files",
        "",
        "- `12_route_overlay_mvt_vs_shp.png`: all cases, reference with both planned routes.",
        "- `B12-*_error_map.png`: two-way spatial error maps for each case.",
        "- `route_similarity_posthoc.json` / `.csv`: full per-route metrics.",
        "- `mvt_shp_pairwise_dispersion.csv`: route-to-route changes between arms.",
        "- `reference_freeze_manifest.json` and `posthoc_manifest.json`: input hashes and run provenance.",
    ])
    with (output_dir / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(summary_lines) + "\n")
    posthoc_manifest["output_sha256"]["README.md"] = sha256(output_dir / "README.md")
    write_json_exclusive(output_dir / "posthoc_manifest.json", posthoc_manifest)

    print(json.dumps({"status": "complete", "output_dir": str(output_dir),
                      "routes_scored": len(metric_rows),
                      "candidate_route_pairs": len(pairwise_rows),
                      "figures": [str(path) for path in render_paths],
                      "summary": overview_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
