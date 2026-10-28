#!/usr/bin/env python3
"""Run the frozen posthoc scorer with the recorded DEM window bounds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SERVER_ROOT = Path("/data1/B24041626/pp")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def planning_bounds(dem_path: Path, argv: list[str], argv_option):
    import rasterio
    from rasterio.windows import Window, from_bounds

    start = argv_option(argv, "--start")
    end = argv_option(argv, "--end")
    margin = argv_option(argv, "--window-margin-deg")
    max_side = argv_option(argv, "--max-side")
    if any(value is None for value in (start, end, margin, max_side)):
        raise ValueError("recorded run is missing a DEM-window argument")

    start_lat, start_lon = map(float, start.split(","))
    end_lat, end_lon = map(float, end.split(","))
    margin = float(margin)
    if int(max_side) < 1:
        raise ValueError("--max-side must be positive")

    with rasterio.open(dem_path) as src:
        if src.crs is None or src.crs != rasterio.crs.CRS.from_epsg(4326):
            raise ValueError("frozen planner DEM is not EPSG:4326")
        dem_left, dem_bottom, dem_right, dem_top = src.bounds
        min_lon = max(min(start_lon, end_lon) - margin, dem_left)
        max_lon = min(max(start_lon, end_lon) + margin, dem_right)
        min_lat = max(min(start_lat, end_lat) - margin, dem_bottom)
        max_lat = min(max(start_lat, end_lat) + margin, dem_top)
        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError("planning window does not intersect the frozen DEM")

        window = from_bounds(min_lon, min_lat, max_lon, max_lat, src.transform)
        window = window.round_offsets().round_lengths()
        window = Window(
            col_off=max(0, window.col_off),
            row_off=max(0, window.row_off),
            width=min(window.width, src.width - window.col_off),
            height=min(window.height, src.height - window.row_off),
        )
        if window.width <= 0 or window.height <= 0:
            raise ValueError("rounded planning window is empty")
        transform = src.window_transform(window)
        left, top = transform.c, transform.f
        right = left + transform.a * window.width
        bottom = top + transform.e * window.height

    # Resampling in load_dem_window preserves the cropped window extent.
    return [float(left), float(bottom), float(right), float(top)]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    cli, _ = parser.parse_known_args()

    source_root = cli.source_root.resolve(strict=True)
    results_dir = cli.results_dir.resolve(strict=True)
    if source_root != (results_dir.parent / "source").resolve(strict=True):
        raise ValueError("source root must be the frozen experiment source directory")
    source_root.relative_to(SERVER_ROOT.resolve(strict=True))
    sys.path.insert(0, str(source_root))
    tools_dir = source_root / "tools"
    evidence_dir = source_root / "docs/evidence/theta_shared_objective_20261028"
    sys.path.insert(0, str(tools_dir))

    scorer = load_module(
        "frozen_window_coverage_scorer",
        tools_dir / "score_shared_objective_routes_posthoc.py",
    )
    adapter_manifest_path = results_dir.parent / "posthoc_scoring_adapter_manifest.json"
    adapter_manifest = json.loads(adapter_manifest_path.read_text(encoding="utf-8"))
    adapter_path = evidence_dir / "score_null_reference_fields_adapter.py"
    expected_adapter_sha = adapter_manifest["artifacts"]["adapter"]["sha256"]
    if sha256(adapter_path) != expected_adapter_sha:
        raise ValueError("frozen null-reference adapter hash mismatch")
    adapter = load_module("frozen_null_reference_adapter", adapter_path)
    scorer.contains_forbidden_reference_metric = adapter.has_non_null_reference_metric

    state = {}
    original_load = scorer.load_and_verify_metadata

    def load_with_bounds(args):
        metadata = original_load(args)
        plan, pre_summary = metadata[5], metadata[8]
        dem_path = Path(plan["inputs"]["dem"]["path"])
        bounds = {}
        for run in pre_summary["runs"]:
            value = planning_bounds(dem_path, run["argv"], scorer.argv_option)
            if run["case_id"] in bounds and bounds[run["case_id"]] != value:
                raise ValueError(f"DEM window differs between arms: {run['case_id']}")
            bounds[run["case_id"]] = value
        if set(bounds) != {case["case_id"] for case in plan["cases"]}:
            raise ValueError("recorded DEM windows do not match frozen plan cases")
        state["bounds"] = bounds
        state["output_dir"] = metadata[-1]
        return metadata

    scorer.load_and_verify_metadata = load_with_bounds
    original_score = scorer.score_candidates

    def score_with_bounds(plan, verified, references, score_source_root):
        from transmission_path_planner import metrics

        previous = metrics.compute_deviation
        bounds_by_reference = {
            id(references[case_id]["lines"]): state["bounds"][case_id]
            for case_id in references
        }

        def compute_with_bounds(planned, real, planning_bounds_wgs84=None):
            bounds = planning_bounds_wgs84 or bounds_by_reference.get(id(real))
            if bounds is None:
                raise ValueError("no frozen planning bounds for reference geometry")
            return previous(planned, real, planning_bounds_wgs84=bounds)

        metrics.compute_deviation = compute_with_bounds
        try:
            return original_score(plan, verified, references, score_source_root)
        finally:
            metrics.compute_deviation = previous

    scorer.score_candidates = score_with_bounds
    scorer.main()

    output_dir = state["output_dir"]
    report_path = output_dir / "route_similarity_posthoc.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["metrics"]
    coverage_count = sum(
        isinstance((row.get("route_similarity") or {}).get("planning_window_coverage"), dict)
        for row in rows
    )
    present_count = report["present_candidate_count"]
    if coverage_count != present_count:
        raise ValueError(f"coverage count {coverage_count} differs from present count {present_count}")

    report["planning_bounds_wgs84_by_case"] = state["bounds"]
    report["coverage_measured_candidate_count"] = coverage_count
    report["provenance"]["window_coverage_audit_adapter"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256(Path(__file__).resolve()),
        "method": "pass the frozen DEM window extent to the pinned compute_deviation metric",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "planning_window_coverage_audit.csv"
    fields = [
        "case_id", "arm", "method", "stage", "status", "planning_bounds_wgs84",
        "reference_length_coverage_ratio", "reference_length_within_window_m",
        "reference_length_outside_window_m", "symmetric_p95_m",
        "f1_100m", "f1_500m", "f1_1000m",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            similarity = row.get("route_similarity") or {}
            coverage = similarity.get("planning_window_coverage") or {}
            symmetric = similarity.get("symmetric") or {}
            f1 = symmetric.get("coverage_f1_within_m") or {}
            writer.writerow({
                "case_id": row["case_id"],
                "arm": row["arm"],
                "method": row["method"],
                "stage": row["stage"],
                "status": row["status"],
                "planning_bounds_wgs84": json.dumps(state["bounds"][row["case_id"]]),
                "reference_length_coverage_ratio": coverage.get("reference_length_coverage_ratio"),
                "reference_length_within_window_m": coverage.get("reference_length_within_window_m"),
                "reference_length_outside_window_m": coverage.get("reference_length_outside_window_m"),
                "symmetric_p95_m": symmetric.get("p95_m"),
                "f1_100m": f1.get("100"),
                "f1_500m": f1.get("500"),
                "f1_1000m": f1.get("1000"),
            })
    print(f"Planning-window coverage: {coverage_count}/{present_count} candidates")


if __name__ == "__main__":
    main()
