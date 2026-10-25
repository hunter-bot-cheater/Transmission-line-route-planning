"""Compare paired constraint-layer runs without using references for planning."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform as transform_geometry, unary_union

from transmission_path_planner.cli import route_custom_constraint_exposure
from transmission_path_planner.data import load_dem_window
from transmission_path_planner.osm import load_user_constraint_overlays


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "run_provenance.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    metrics_rows = json.loads((run_dir / "comparison_metrics.json").read_text(encoding="utf-8"))
    routes = {}
    for path in run_dir.rglob("final_route.geojson"):
        document = json.loads(path.read_text(encoding="utf-8"))
        for feature in document.get("features", []):
            case = (feature.get("properties") or {}).get("case")
            if case and feature.get("geometry"):
                routes[case] = shape(feature["geometry"])
    cases = {item["case"]: item for item in manifest["cases"]}
    metrics_by_case = {}
    for item in metrics_rows:
        case = item["case"]
        metrics_by_case[case] = item["metrics"].get("最终路线", {})
    failed_cases = {item["case"]: item.get("error", "")
                    for item in summary.get("failed_cases", [])}
    no_route_cases = set(summary.get("no_route_cases", []))
    completed_cases = {item["case"] for item in
                       summary.get("real_evaluation_by_case", [])}
    stale_routes = set(routes) & (set(failed_cases) | no_route_cases)
    missing_routes = completed_cases - set(routes)
    if stale_routes or missing_routes:
        raise ValueError(
            f"run summary and route artifacts disagree: "
            f"stale={sorted(stale_routes)}, missing={sorted(missing_routes)}")
    rows = {}
    for case in set(cases) | set(metrics_by_case) | set(routes):
        route = routes.get(case)
        if route is not None:
            status, reason = "route", ""
        elif case in failed_cases:
            status, reason = "failed", failed_cases[case]
        elif case in no_route_cases:
            status, reason = "no_route", "planner returned no route"
        else:
            status, reason = "no_artifact", "case has no route artifact"
        rows[case] = {"metrics": metrics_by_case.get(case, {}),
                      "route": route, "case_input": cases.get(case),
                      "status": status, "failure_reason": reason}
    return manifest, summary, rows


def _metric_similarity(metrics):
    similarity = metrics.get("route_similarity") or {}
    symmetric = similarity.get("symmetric") or {}
    f1 = symmetric.get("coverage_f1_within_m") or {}
    return {"symmetric_p95_m": symmetric.get("p95_m"),
            "f1_100m": f1.get("100"),
            "f1_1000m": f1.get("1000")}


def _projected_constraint_polygons(constraint_path, project):
    document = json.loads(Path(constraint_path).read_text(encoding="utf-8"))
    groups = {}
    for feature in document["features"]:
        properties = feature.get("properties") or {}
        if properties.get("mode") != "soft":
            continue
        category = str(properties.get("category") or "uncategorized")
        geometry = transform_geometry(project, shape(feature["geometry"]))
        groups.setdefault(category, []).append(geometry)
    return document, {name: unary_union(geometries)
                      for name, geometries in groups.items()}


def _plot_geometry(ax, geometry, facecolor, edgecolor):
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for polygon in polygons:
        if polygon.is_empty:
            continue
        x, y = polygon.exterior.xy
        ax.fill(x, y, color=facecolor, alpha=0.18, linewidth=0)
        ax.plot(x, y, color=edgecolor, linewidth=0.45, alpha=0.8)
        for ring in polygon.interiors:
            x, y = ring.xy
            ax.fill(x, y, color="white", linewidth=0)


def compare(without_dir, with_dir, dem_path, constraint_path, output_dir):
    without_dir = Path(without_dir)
    with_dir = Path(with_dir)
    without_manifest, without_summary, without_rows = _load_run(without_dir)
    with_manifest, with_summary, with_rows = _load_run(with_dir)
    base_config = without_manifest["configuration"]
    treatment_config = with_manifest["configuration"]
    config_keys = set(base_config) | set(treatment_config)
    changed_config = {key for key in config_keys
                      if base_config.get(key) != treatment_config.get(key)}
    if changed_config != {"out_dir", "constraint_geojson"}:
        raise ValueError(
            f"paired runs differ in unexpected configuration fields: "
            f"{sorted(changed_config)}")
    if base_config.get("constraint_geojson") or not treatment_config.get("constraint_geojson"):
        raise ValueError("expected an unconfigured baseline and a constraint-enabled treatment")
    if without_manifest.get("cases") != with_manifest.get("cases"):
        raise ValueError("paired runs use different case endpoints")
    if (without_manifest.get("source") != with_manifest.get("source")
            or without_manifest.get("effective_runtime")
            != with_manifest.get("effective_runtime")):
        raise ValueError("paired runs use different code or runtime provenance")

    def input_signatures(manifest, include_constraints=False):
        rows = [item for item in manifest.get("planning_inputs", [])
                if include_constraints or item.get("role") != "user_constraints"]
        if any(item.get("status") != "hashed" or not item.get("sha256")
               for item in rows):
            raise ValueError("paired input manifest contains unhashed inputs")
        return sorted((item.get("role"), item.get("path"),
                       item.get("size_bytes"), item.get("sha256"))
                      for item in rows)

    if input_signatures(without_manifest) != input_signatures(with_manifest):
        raise ValueError("paired runs differ in non-treatment input files")
    treatment_constraints = [
        item for item in with_manifest.get("planning_inputs", [])
        if item.get("role") == "user_constraints"]
    if (len(treatment_constraints) != 1
            or treatment_constraints[0].get("sha256") != _sha256(constraint_path)):
        raise ValueError("treatment constraint file does not match its provenance hash")
    if any(item.get("role") == "user_constraints"
           for item in without_manifest.get("planning_inputs", [])):
        raise ValueError("baseline unexpectedly contains user constraints")

    forward = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True).transform
    constraint_doc, category_polygons = _projected_constraint_polygons(
        constraint_path, forward)
    feature_counts = {}
    for feature in constraint_doc["features"]:
        props = feature.get("properties") or {}
        if props.get("mode") == "soft":
            category = str(props.get("category") or "uncategorized")
            feature_counts[category] = feature_counts.get(category, 0) + 1
    all_constraints = unary_union(list(category_polygons.values()))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    maps = []
    cases = sorted(set(without_rows) | set(with_rows))
    for case in cases:
        base = without_rows.get(case, {})
        treatment = with_rows.get(case, {})
        base_metrics = base.get("metrics", {})
        treatment_metrics = treatment.get("metrics", {})
        base_route = base.get("route")
        treatment_route = treatment.get("route")
        case_row = base.get("case_input") or treatment.get("case_input") or {}
        record = {
            "case": case,
            "status_without": base.get("status", "missing"),
            "status_with": treatment.get("status", "missing"),
            "failure_reason_without": base.get("failure_reason", ""),
            "failure_reason_with": treatment.get("failure_reason", ""),
            "length_without_km": base_metrics.get("length_km"),
            "length_with_km": treatment_metrics.get("length_km"),
        }
        if base_route and treatment_route:
            slat, slon = case_row["start_latlon"]
            elat, elon = case_row["end_latlon"]
            margin = float(base_config.get("window_margin_deg", 0.30))
            dem, transform, _, pixel_m = load_dem_window(
                dem_path, min(slon, elon) - margin, min(slat, elat) - margin,
                max(slon, elon) + margin, max(slat, elat) + margin,
                int(base_config["max_side"]))
            raster_constraints = load_user_constraint_overlays(
                constraint_path, transform, dem.shape)
            base_exposure = route_custom_constraint_exposure(
                [(x, y) for x, y in base_route.coords], raster_constraints,
                transform, dem.shape)
            with_exposure = route_custom_constraint_exposure(
                [(x, y) for x, y in treatment_route.coords], raster_constraints,
                transform, dem.shape)
            del dem, raster_constraints

            base_projected = transform_geometry(forward, base_route)
            treatment_projected = transform_geometry(forward, treatment_route)
            total_polygon_overlap_base = base_projected.intersection(all_constraints).length
            total_polygon_overlap_with = treatment_projected.intersection(all_constraints).length
            record.update({
                "length_delta_km": treatment_metrics.get("length_km", 0.0)
                - base_metrics.get("length_km", 0.0),
                "raster_soft_cells_without": base_exposure[
                    "soft_affected_route_cell_count"],
                "raster_soft_cells_with": with_exposure[
                    "soft_affected_route_cell_count"],
                "raster_soft_weight_sum_without": base_exposure[
                    "summed_soft_weight_cells"],
                "raster_soft_weight_sum_with": with_exposure[
                    "summed_soft_weight_cells"],
                "vector_overlap_km_without": total_polygon_overlap_base / 1000.0,
                "vector_overlap_km_with": total_polygon_overlap_with / 1000.0,
                "vector_overlap_fraction_without": (
                    total_polygon_overlap_base / base_projected.length
                    if base_projected.length else None),
                "vector_overlap_fraction_with": (
                    total_polygon_overlap_with / treatment_projected.length
                    if treatment_projected.length else None),
                **{f"{key}_without": value for key, value in
                   _metric_similarity(base_metrics).items()},
                **{f"{key}_with": value for key, value in
                   _metric_similarity(treatment_metrics).items()},
                "pixel_size_m": float(pixel_m),
            })
            for category, polygon in category_polygons.items():
                record[f"{category}_overlap_km_without"] = (
                    base_projected.intersection(polygon).length / 1000.0)
                record[f"{category}_overlap_km_with"] = (
                    treatment_projected.intersection(polygon).length / 1000.0)
            maps.append((case, base_projected, treatment_projected,
                         base_metrics.get("length_km"),
                         treatment_metrics.get("length_km")))
        else:
            record["failure_reason"] = (
                (base_metrics if not base_route else treatment_metrics)
                .get("planning_status", "not_available"))
        records.append(record)

    _write_map(maps, category_polygons, output_dir / "constraint_route_ab.png")
    (output_dir / "constraint_route_ab.json").write_text(
        json.dumps({
            "constraint_file_sha256": _sha256(constraint_path),
            "dem_file_sha256": _sha256(dem_path),
            "constraint_feature_counts_by_category": feature_counts,
            "soft_feature_count": sum(feature_counts.values()),
            "max_side": base_config["max_side"],
            "window_margin_deg": base_config["window_margin_deg"],
            "paired_runs": {
                "without_constraints": _run_provenance_record(
                    without_dir, without_manifest, without_summary),
                "with_constraints": _run_provenance_record(
                    with_dir, with_manifest, with_summary),
            },
            "cases": records,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(records, output_dir / "constraint_route_ab.csv")
    _write_report(records, output_dir / "constraint_route_ab.md",
                  feature_counts, base_config, dem_path, constraint_path)
    return records


def _run_provenance_record(run_dir, manifest, summary):
    return {
        "run_provenance_sha256": _sha256(Path(run_dir) / "run_provenance.json"),
        "run_summary_sha256": _sha256(Path(run_dir) / "run_summary.json"),
        "command": manifest.get("command"),
        "configuration": manifest.get("configuration"),
        "source": manifest.get("source"),
        "effective_runtime": manifest.get("effective_runtime"),
        "planning_inputs": manifest.get("planning_inputs"),
        "reference_geometry_policy": manifest.get("reference_geometry_policy"),
        "summary": {
            "expected_cases": summary.get("expected_cases"),
            "completed_cases": summary.get("completed_cases"),
            "failed_cases": summary.get("failed_cases"),
            "no_route_cases": summary.get("no_route_cases"),
        },
    }


def _write_map(maps, category_polygons, output_path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 20), constrained_layout=True)
    axes = list(axes.flat)
    colors = {"drinking_water_protection": ("#69a9d1", "#2376a5"),
              "marine_wildlife_habitat": ("#73c6a3", "#14865d")}
    for ax, (case, base, treatment, base_km, treatment_km) in zip(axes, maps):
        all_bounds = unary_union([base, treatment]).bounds
        minx, miny, maxx, maxy = all_bounds
        padx, pady = max((maxx - minx) * 0.08, 1000), max((maxy - miny) * 0.08, 1000)
        view = box(minx - padx, miny - pady, maxx + padx, maxy + pady)
        for category, polygon in category_polygons.items():
            clipped = polygon.intersection(view)
            if clipped.is_empty:
                continue
            fill, edge = colors.get(category, ("#cccccc", "#777777"))
            _plot_geometry(ax, clipped, fill, edge)
        ax.plot(*base.xy, color="#c0392b", linewidth=1.4, label="without soft constraints")
        ax.plot(*treatment.xy, color="#e0a800", linewidth=1.4, label="with soft constraints")
        delta = treatment_km - base_km
        ax.set_title(f"{case} | length delta {delta:+.2f} km")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        legend_handles = [
            Line2D([0], [0], color="#c0392b", linewidth=1.4,
                   label="without soft constraints"),
            Line2D([0], [0], color="#e0a800", linewidth=1.4,
                   label="with soft constraints"),
        ]
        for category, (fill, edge) in colors.items():
            if category in category_polygons:
                legend_handles.append(Patch(
                    facecolor=fill, edgecolor=edge, alpha=0.35,
                    label=category.replace("_", " ")))
        ax.legend(handles=legend_handles, loc="best", fontsize=7)
    for ax in axes[len(maps):]:
        ax.axis("off")
    fig.suptitle("Soft constraint A/B routes and screening polygons", fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_csv(records, path):
    columns = sorted({key for record in records for key in record})
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _write_report(records, path, feature_counts, config, dem_path, constraint_path):
    paired = [row for row in records if row.get("status_without") == "route"
              and row.get("status_with") == "route"]
    unsuccessful = [row for row in records if row.get("status_without") != "route"
                    or row.get("status_with") != "route"]
    deltas = [row["length_delta_km"] for row in paired]
    if deltas:
        median_delta = float(np.median(deltas))
        overlap_without = float(sum(
            [row["vector_overlap_km_without"] for row in paired]))
        overlap_with = float(sum(
            [row["vector_overlap_km_with"] for row in paired]))
        median_p95_without = float(np.median(
            [row["symmetric_p95_m_without"] for row in paired]))
        median_p95_with = float(np.median(
            [row["symmetric_p95_m_with"] for row in paired]))
        median_f1_without = float(np.median(
            [row["f1_100m_without"] for row in paired]))
        median_f1_with = float(np.median(
            [row["f1_100m_with"] for row in paired]))
    else:
        median_delta = overlap_without = overlap_with = float("nan")
        median_p95_without = median_p95_with = float("nan")
        median_f1_without = median_f1_with = float("nan")
    lines = [
        "# Soft constraint paired route screening",
        "",
        f"- Cases: {len(records)} total; {len(paired)} paired routes; "
        f"{len(unsuccessful)} with at least one unsuccessful result.",
        f"- Raster cap: `{config['max_side']}`; search margin: `{config['window_margin_deg']}` degrees.",
        f"- Median paired length change: `{median_delta:+.3f} km` (treatment minus baseline).",
        f"- Median symmetric P95 distance: `{median_p95_without/1000:.3f} km` to `{median_p95_with/1000:.3f} km`.",
        f"- Median F1@100m: `{100*median_f1_without:.3f}%` to `{100*median_f1_with:.3f}%`.",
        f"- Sum of per-route exact vector overlap with soft polygons: `{overlap_without:.3f} km` to `{overlap_with:.3f} km`.",
        f"- Soft features: `{sum(feature_counts.values())}`; by category: `{json.dumps(feature_counts, ensure_ascii=False)}`; each feature weight is `1.0`.",
        f"- DEM SHA-256: `{_sha256(dem_path)}`",
        f"- Constraint GeoJSON SHA-256: `{_sha256(constraint_path)}`",
        "",
        "| Case | Status base / treatment | Length base / treatment (km) | Δ length (km) | Raster soft cells base / treatment | Exact vector overlap base / treatment (km) | Symmetric P95 base / treatment (km) | F1@100m base / treatment | Failure detail |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in records:
        if row.get("status_without") != "route" or row.get("status_with") != "route":
            lines.append(
                f"| {row['case']} | {row['status_without']} / {row['status_with']} | — | — | — | — | — | — | "
                f"{row.get('failure_reason_without') or row.get('failure_reason_with') or '—'} |")
            continue
        lines.append(
            f"| {row['case']} | {row['status_without']} / {row['status_with']} | "
            f"{row['length_without_km']:.2f} / {row['length_with_km']:.2f} | "
            f"{row['length_delta_km']:+.2f} | "
            f"{row['raster_soft_cells_without']} / {row['raster_soft_cells_with']} | "
            f"{row['vector_overlap_km_without']:.2f} / {row['vector_overlap_km_with']:.2f} | "
            f"{row['symmetric_p95_m_without']/1000:.2f} / {row['symmetric_p95_m_with']/1000:.2f} | "
            f"{100*row['f1_100m_without']:.2f}% / {100*row['f1_100m_with']:.2f}% | — |")
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "This is a sensitivity comparison, not a weight calibration. The reference line geometries were used only by the planner's post-hoc scorer; they did not enter route search or candidate selection. The CSV endpoints are still historical-line endpoints, not independently verified station terminals.",
        "The 1024-cell cap produced roughly 102–134 m cells across these windows, so this comparison cannot support high-resolution engineering claims. OSM building and land-use coverage is observed-feature coverage, not proof of completeness. The official protection polygons are screening layers; overlap or avoidance does not establish legal status, risk reduction, or route feasibility.",
        "The soft layer removed about 10.1 km of summed route-polygon overlap across the seven successful pairs, while median symmetric P95 distance worsened. That is an avoidance-versus-reference-fit tradeoff, not an overall accuracy gain; the weight is uncalibrated.",
        "Vector overlap is measured after projecting the route and polygons to EPSG:3826. Raster exposure is all-touched cell contact on the exact planning grid; the two measures need not agree at coarse resolution.",
        "",
        "## Reproduction",
        "",
        "The paired commands used identical DEM, OSM cache, endpoints, A* settings, official hazard MVT, and `--max-side 1024`. The treatment additionally supplied `--constraint-geojson`; the baseline omitted it. Outputs remain under the ignored `outputs/constraints_soft_ab_8cases_1024_20261025/` directory.",
        "The adjacent `constraint_route_ab.json` preserves each run's exact command and working directory, planner source hashes, runtime flags, planning-input hashes, and run-summary hash. The run provenance reported a dirty source tree; compare its recorded source-file hashes before attempting to reproduce it from Git alone.",
    ])
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-dir", required=True)
    parser.add_argument("--with-dir", required=True)
    parser.add_argument("--dem", required=True)
    parser.add_argument("--constraints", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    compare(args.without_dir, args.with_dir, args.dem,
            args.constraints, args.out_dir)


if __name__ == "__main__":
    main()
