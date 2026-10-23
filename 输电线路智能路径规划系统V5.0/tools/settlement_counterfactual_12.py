"""Freeze and post-hoc score the 12-case mapped-settlement counterfactual.

Run ``python tools/settlement_counterfactual_12.py freeze`` before opening any
reference geometry, then ``... score`` once the candidate manifest is verified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "settlement_counterfactual_12_20261025"
OUT = DEFAULT_OUT
PLAN = OUT / "plan.json"
MANIFEST = OUT / "candidate_manifest.json"
POSTHOC = OUT / "posthoc_settlement"
REAL_DIR = Path(r"D:\大创\data\real_cases_tw")
OSM_DIR = Path(r"D:\大创\V5_final\data\downloaded")
HAZARD_DIR = ROOT / "data" / "official_sources" / "ardswc_115_mvt_z10_20261023"
VARIANTS = ("settlement_weight_2", "settlement_weight_0")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, category: str) -> dict:
    return {"category": category, "path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_records(records: list[dict]) -> None:
    for item in records:
        path = Path(item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise SystemExit(f"frozen file missing/changed: {path}")
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"frozen SHA-256 mismatch: {path}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def current_git_state() -> dict:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).strip()
    if status:
        raise SystemExit("freeze requires a clean worktree")
    return {"head": head, "worktree_clean": True}


def package_versions() -> dict:
    packages = {}
    for package in ("numpy", "shapely", "pyproj", "rasterio", "scipy", "matplotlib"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        import rasterio
        rasterio_version = rasterio.__version__
        gdal_version = rasterio.__gdal_version__
    except ImportError:
        rasterio_version = gdal_version = None
    return {"python": sys.version, "platform": platform.platform(),
            "packages": packages, "gdal": gdal_version,
            "rasterio": rasterio_version}


def load_proxy_scores(cases: list[str]) -> list[dict]:
    proxy_scores = []
    for variant in VARIANTS:
        rows = read_json(OUT / variant / "comparison_metrics.json")
        row_cases = [row.get("case") for row in rows]
        if len(row_cases) != len(set(row_cases)) or set(row_cases) != set(cases):
            raise SystemExit(f"comparison metrics case rows are incomplete/duplicated: {variant}")
        by_case = {row["case"]: row for row in rows}
        for case in cases:
            method = by_case[case].get("metrics", {}).get("A*(baseline)", {})
            if not method:
                raise SystemExit(f"baseline A* metrics missing: {variant}/{case}")
            exposure = method.get("weight_independent_exposure", {})
            proxy = exposure.get("mapped_settlement_proxy_mean")
            if proxy is not None:
                import math
                proxy = float(proxy)
                if not math.isfinite(proxy):
                    proxy = None
            buildings = exposure.get("mapped_osm_layer_status", {}).get("buildings")
            landuse = exposure.get("mapped_osm_layer_status", {}).get("landuse")
            if proxy is not None and (buildings != "available" or landuse != "available"):
                raise SystemExit(f"settlement proxy present while required OSM layer unavailable: {variant}/{case}")
            proxy_scores.append({
                "case": case, "variant": variant,
                "mapped_settlement_proxy_mean": proxy,
                "building_layer_status": buildings,
                "landuse_layer_status": landuse,
                "coverage_status": exposure.get("mapped_osm_coverage_status"),
            })
    return proxy_scores


def normalized_argv(summary: dict) -> list[str]:
    argv = summary.get("argv", [])
    kept = []
    skip_next = False
    for value in argv:
        if skip_next:
            skip_next = False
            continue
        if value in ("--settlement-weight", "--out-dir"):
            skip_next = True
            continue
        kept.append(str(value))
    if skip_next:
        raise SystemExit("malformed run argv")
    return kept


def parsed_cli_args(summary: dict) -> dict[str, str | bool]:
    argv = [str(value) for value in summary.get("argv", [])[1:]]
    value_flags = {
        "--dem", "--osm-cache-dir", "--real-cases-csv", "--real-dir", "--out-dir",
        "--settlement-weight", "--max-side", "--official-hazards-dir",
        "--official-hazard-weight", "--terrain-slope-weight", "--roughness-weight",
        "--ridge-penalty-weight", "--landcover-penalty-weight", "--grade-penalty-weight",
        "--turn-penalty-weight", "--astar-heuristic-weight", "--building-clearance-m",
    }
    switches = {"--no-vin", "--no-ppp"}
    parsed = {}
    index = 0
    while index < len(argv):
        flag = argv[index]
        if flag in switches:
            if flag in parsed:
                raise SystemExit(f"duplicate CLI switch: {flag}")
            parsed[flag] = True
            index += 1
            continue
        if flag not in value_flags or index + 1 >= len(argv):
            raise SystemExit(f"unexpected or malformed CLI argument: {flag}")
        if flag in parsed:
            raise SystemExit(f"duplicate CLI argument: {flag}")
        parsed[flag] = argv[index + 1]
        index += 2
    return parsed


def freeze() -> None:
    if MANIFEST.exists():
        raise SystemExit(f"candidate manifest already exists: {MANIFEST}")
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    git_state = current_git_state()
    if git_state["head"] != plan.get("git_head"):
        raise SystemExit("Git HEAD changed since candidate generation")
    source_diff = subprocess.run(
        ["git", "diff", "--quiet", plan["git_head"], "--",
         "transmission_path_planner"], cwd=ROOT, check=False)
    if source_diff.returncode != 0:
        raise SystemExit("planner source changed since candidate generation")
    summaries = {name: read_json(OUT / name / "run_summary.json")
                 for name in VARIANTS}
    expected_cases = int(plan["cases"])
    cases_by_variant = {}
    for name, summary in summaries.items():
        if (summary.get("complete") is not True
                or summary.get("completed_cases") != expected_cases
                or summary.get("recorded_cases") != expected_cases
                or summary.get("failed_cases") or summary.get("no_route_cases")):
            raise SystemExit(f"run is incomplete or failed: {name}")
        rows = summary.get("real_evaluation_by_case", [])
        if len(rows) != expected_cases or any(
                item.get("status") != "reference_missing" for item in rows):
            raise SystemExit(f"reference isolation check failed: {name}")
        cases_by_variant[name] = sorted(item["case"] for item in rows)
    if cases_by_variant[VARIANTS[0]] != cases_by_variant[VARIANTS[1]]:
        raise SystemExit("variant case IDs do not match")
    if [item["name"] for item in plan.get("variants", [])] != list(VARIANTS):
        raise SystemExit("plan variants do not match the freeze tool")
    weights = {item["name"]: float(item["weight"]) for item in plan["variants"]}
    expected_weights = {VARIANTS[0]: 2.0, VARIANTS[1]: 0.0}
    if weights != expected_weights:
        raise SystemExit(f"unexpected settlement weights: {weights}")
    quarantine = OUT / "reference_quarantine"
    if not quarantine.is_dir() or any(quarantine.rglob("*")):
        raise SystemExit("reference quarantine must exist and be empty")
    expected_csv = Path(plan["endpoint_csv"]).resolve()
    expected_dem = (ROOT / plan["dem"]).resolve()
    if sha256(expected_csv) != plan.get("endpoint_csv_sha256"):
        raise SystemExit("endpoint CSV changed since candidate generation")
    if sha256(expected_dem) != plan.get("dem_sha256"):
        raise SystemExit("DEM changed since candidate generation")
    with expected_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        csv_cases = [row["name"].strip() for row in csv.DictReader(stream)]
    if (csv_cases != plan.get("case_ids") or len(csv_cases) != expected_cases
            or len(set(csv_cases)) != expected_cases):
        raise SystemExit("pre-registered case IDs do not match endpoint CSV")
    if cases_by_variant[VARIANTS[0]] != sorted(csv_cases):
        raise SystemExit("planner case IDs do not match endpoint CSV")
    if plan.get("generation_worktree_clean") is not True:
        raise SystemExit("plan does not attest a clean generation worktree")
    for key, path in (("runner_script", ROOT / "tools" / "run_settlement_counterfactual_12.py"),
                      ("freeze_score_script", Path(__file__).resolve())):
        expected_hash = plan.get(key, {}).get("sha256")
        if not expected_hash or sha256(path) != expected_hash:
            raise SystemExit(f"experiment tool changed since pre-registration: {path}")
    if package_versions() != plan.get("environment"):
        raise SystemExit("runtime environment differs from pre-registered generation environment")
    if plan.get("min_proxy_complete_cases_for_overall_summary") != expected_cases:
        raise SystemExit("proxy summary completeness rule is missing or changed")
    planned_args = plan.get("planner_cli_args_by_variant", {})
    for name in VARIANTS:
        actual_argv = [str(value) for value in summaries[name].get("argv", [])[1:]]
        if actual_argv != planned_args.get(name):
            raise SystemExit(f"actual CLI argv differs from pre-registered arguments: {name}")
    expected_common = {
        "--dem": str(expected_dem),
        "--osm-cache-dir": str(OSM_DIR.resolve()),
        "--real-cases-csv": str(expected_csv),
        "--real-dir": str(quarantine.resolve()),
        "--max-side": str(plan["max_side"]),
        "--official-hazards-dir": str(HAZARD_DIR.resolve()),
        "--official-hazard-weight": "1.0",
        "--terrain-slope-weight": "0.75",
        "--roughness-weight": "0.25",
        "--ridge-penalty-weight": "0.1",
        "--landcover-penalty-weight": "0.0",
        "--grade-penalty-weight": "1.0",
        "--turn-penalty-weight": "0.0",
        "--astar-heuristic-weight": "1.0",
        "--building-clearance-m": "0.0",
        "--no-vin": True,
        "--no-ppp": True,
    }
    args_by_variant = {name: parsed_cli_args(summaries[name]) for name in VARIANTS}
    for name in VARIANTS:
        expected = {**expected_common,
                    "--out-dir": str((OUT / name).resolve()),
                    "--settlement-weight": str(weights[name])}
        if args_by_variant[name] != expected:
            raise SystemExit(f"actual CLI arguments differ from frozen plan for {name}")
    if normalized_argv(summaries[VARIANTS[0]]) != normalized_argv(summaries[VARIANTS[1]]):
        raise SystemExit("run argv differs beyond settlement weight/output directory")

    records = [file_record(PLAN, "experiment_plan")]
    endpoint_csv = Path(plan["endpoint_csv"])
    records.append(file_record(endpoint_csv, "endpoint_csv"))
    dem = ROOT / plan["dem"]
    records.append(file_record(dem, "dem"))
    if not plan.get("hazard_files"):
        raise SystemExit("official hazard files were not pre-hashed before candidate generation")
    actual_hazards = sorted(path for path in HAZARD_DIR.rglob("*") if path.is_file())
    planned_hazards = {item["relative_path"]: item for item in plan["hazard_files"]}
    if {path.relative_to(HAZARD_DIR).as_posix() for path in actual_hazards} != set(planned_hazards):
        raise SystemExit("official hazard file set changed since planning")
    for path in actual_hazards:
        item = planned_hazards[path.relative_to(HAZARD_DIR).as_posix()]
        if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise SystemExit(f"official hazard source changed since planning: {path}")
    osm_items = plan.get("osm_cache_files", [])
    actual_osm_files = sorted(path for path in OSM_DIR.iterdir() if path.is_file())
    planned_osm_paths = {item["relative_path"] for item in osm_items}
    if planned_osm_paths != {path.name for path in actual_osm_files}:
        raise SystemExit("OSM cache file set changed since planning")
    for item in osm_items:
        path = OSM_DIR / item["relative_path"]
        record = file_record(path, "osm_cache")
        if record["sha256"] != item["sha256"] or record["bytes"] != item["bytes"]:
            raise SystemExit(f"OSM cache changed since planning: {path}")
        records.append(record)
    records.extend(file_record(path, "official_hazard") for path in actual_hazards)
    source_files = sorted((ROOT / "transmission_path_planner").rglob("*.py"))
    planned_sources = {item["relative_path"]: item for item in plan.get("planner_source_files", [])}
    if {path.relative_to(ROOT).as_posix() for path in source_files} != set(planned_sources):
        raise SystemExit("planner source file set changed since planning")
    for path in source_files:
        item = planned_sources[path.relative_to(ROOT).as_posix()]
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise SystemExit(f"planner source changed since planning: {path}")
    records.extend(file_record(path, "planner_source") for path in source_files)
    for name in VARIANTS:
        files = sorted(path for path in (OUT / name).rglob("*") if path.is_file())
        if not files:
            raise SystemExit(f"candidate outputs are empty: {name}")
        records.extend(file_record(path, f"candidate:{name}") for path in files)
        records.append(file_record(OUT / f"{name}_console.log", "run_log"))
    records.append(file_record(Path(__file__).resolve(), "experiment_tool"))
    runner_path = ROOT / "tools" / "run_settlement_counterfactual_12.py"
    records.append(file_record(runner_path, "experiment_runner"))
    verify_records(records)
    proxy_scores = load_proxy_scores(cases_by_variant[VARIANTS[0]])
    verify_records(records)
    if PLAN.read_bytes() != plan_bytes:
        raise SystemExit("pre-registration plan changed during candidate freeze")
    manifest = {
        "experiment": plan["experiment"],
        "generation_git_head": plan["git_head"],
        "freeze_git_state": git_state,
        "expected_cases": expected_cases,
        "cases": cases_by_variant[VARIANTS[0]],
        "variants": list(VARIANTS),
        "settlement_weights": {
            item["name"]: item["weight"] for item in plan["variants"]
        },
        "max_side_by_variant": {
            name: summaries[name].get("max_side") for name in VARIANTS
        },
        "argv_common_after_removing_weight_and_output": normalized_argv(summaries[VARIANTS[0]]),
        "effective_parameters": {
            "official_hazard_weight": 1.0,
            "terrain_slope_weight": 0.75,
            "roughness_weight": 0.25,
            "ridge_penalty_weight": 0.1,
            "landcover_penalty_weight": 0.0,
            "grade_penalty_weight": 1.0,
            "turn_penalty_weight": 0.0,
            "astar_heuristic_weight": 1.0,
            "max_ground_grade": None,
            "max_surface_slope_deg": None,
            "building_clearance_m": 0.0,
            "official_hazards_dir": str(HAZARD_DIR.resolve()),
        },
        "reference_geometry_in_candidate_inputs": False,
        "reference_quarantine_empty": True,
        "reference_status_per_case": "reference_missing",
        "primary_settlement_metric": (
            "mapped_settlement_proxy_mean over all-touched route cells; "
            "mapped-feature proxy, not population exposure, not length weighted"),
        "candidate_and_input_files": records,
        "environment": package_versions(),
        "frozen_mapped_settlement_proxy": proxy_scores,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"frozen {len(cases_by_variant[VARIANTS[0]])} paired cases; "
          f"{len(records)} file hashes in {MANIFEST}")


def candidate_geometry(variant: str, case: str):
    from shapely.geometry import shape

    matches = []
    for path in (OUT / variant).glob("*/final_route.geojson"):
        document = read_json(path)
        for feature in document.get("features", []):
            if feature.get("properties", {}).get("case") == case:
                geometry = shape(feature["geometry"])
                if geometry.geom_type != "LineString":
                    raise ValueError(f"expected LineString in {path}")
                matches.append((path, geometry))
    if len(matches) != 1:
        raise ValueError(f"expected one candidate for {variant}/{case}, found {len(matches)}")
    return matches[0]


def score() -> None:
    if POSTHOC.exists():
        raise SystemExit(f"post-hoc output already exists: {POSTHOC}")
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("reference_geometry_in_candidate_inputs") is not False:
        raise SystemExit("manifest does not confirm reference isolation")
    verify_records(manifest["candidate_and_input_files"])
    own_record = next(item for item in manifest["candidate_and_input_files"]
                      if item["category"] == "experiment_tool")
    if sha256(Path(__file__).resolve()) != own_record["sha256"]:
        raise SystemExit("scoring tool changed after candidate freeze")

    from pyproj import Transformer
    from shapely.geometry import LineString
    from shapely.ops import transform
    import matplotlib.pyplot as plt
    from io import BytesIO
    from transmission_path_planner.metrics import (
        compute_deviation, eval_planned_route_error, load_real_path)
    from transmission_path_planner.io_export import resolve_real_path

    cases = manifest["cases"]
    candidates = {}
    for case in cases:
        candidates[(VARIANTS[0], case)] = candidate_geometry(VARIANTS[0], case)
        candidates[(VARIANTS[1], case)] = candidate_geometry(VARIANTS[1], case)

    # Candidate and all input hashes have been checked before opening references.
    reference_hashes = {}
    reference_geometries = {}
    route_scores = []
    relative_scores = []
    project = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True).transform
    for case in cases:
        reference_path = resolve_real_path(case, REAL_DIR, include_global_mapping=False)
        if reference_path is None or not reference_path.is_file():
            raise SystemExit(f"reference shapefile is missing for case: {case}")
        components = sorted(path for path in reference_path.parent.glob(
            f"{reference_path.stem}.*") if path.is_file())
        reference_hashes[case] = [
            {"path": str(path.resolve()), "bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in components]
        reference = load_real_path(reference_path, require_single_line=True,
                                   raise_errors=True)
        reference_geometries[case] = reference
        for variant in VARIANTS:
            route_path, geometry = candidates[(variant, case)]
            coordinates = [(float(x), float(y)) for x, y, *_ in geometry.coords]
            deviation = compute_deviation(coordinates, reference)
            error = eval_planned_route_error(coordinates, deviation)
            symmetric = error["route_similarity"]["symmetric"]
            route_scores.append({
                "case": case, "variant": variant,
                "candidate": str(route_path.relative_to(ROOT)),
                "length_km": error["route_similarity"]["length"]["planned_km"],
                "symmetric_p95_m": symmetric["p95_m"],
                "symmetric_mean_m": symmetric["mean_m"],
                "symmetric_sampled_hausdorff_m": symmetric["sampled_hausdorff_m"],
                "f1_100m": symmetric["coverage_f1_within_m"]["100"],
                "absolute_length_difference_pct": error["route_similarity"]["length"]["absolute_difference_pct"],
            })
        first = candidates[(VARIANTS[0], case)][1]
        second = candidates[(VARIANTS[1], case)][1]
        relative = compute_deviation(
            [(float(x), float(y)) for x, y, *_ in first.coords],
            [[(float(x), float(y)) for x, y, *_ in second.coords]])
        relative_scores.append({
            "case": case,
            "symmetric_p95_m_between_candidates": relative["route_similarity"]["symmetric"]["p95_m"],
            "symmetric_mean_m_between_candidates": relative["route_similarity"]["symmetric"]["mean_m"],
        })

    proxy_scores = manifest.get("frozen_mapped_settlement_proxy", [])
    if len(proxy_scores) != len(cases) * len(VARIANTS):
        raise SystemExit("frozen settlement proxy results are incomplete")

    score_by_key = {(row["case"], row["variant"]): row for row in route_scores}
    paired_p95 = [
        score_by_key[(case, VARIANTS[1])]["symmetric_p95_m"]
        - score_by_key[(case, VARIANTS[0])]["symmetric_p95_m"] for case in cases]
    import numpy as np
    paired_proxy = []
    excluded_proxy_cases = []
    proxy_by_key = {(row["case"], row["variant"]): row for row in proxy_scores}
    for case in cases:
        low = proxy_by_key[(case, VARIANTS[0])]["mapped_settlement_proxy_mean"]
        high = proxy_by_key[(case, VARIANTS[1])]["mapped_settlement_proxy_mean"]
        status_2 = proxy_by_key[(case, VARIANTS[0])]
        status_0 = proxy_by_key[(case, VARIANTS[1])]
        coverage_matches = (
            status_2["coverage_status"] == status_0["coverage_status"]
            and status_2["building_layer_status"] == status_0["building_layer_status"]
            and status_2["landuse_layer_status"] == status_0["landuse_layer_status"]
            and status_2["building_layer_status"] == "available"
            and status_2["landuse_layer_status"] == "available"
        )
        if low is not None and high is not None and coverage_matches:
            paired_proxy.append(float(low) - float(high))
        else:
            excluded_proxy_cases.append(case)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        ref = reference_geometries[case]
        ref_line = LineString(ref[0])
        projected_ref = transform(project, ref_line)
        x, y = projected_ref.xy
        ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, "k--", lw=1.25, label="参考线")
        for variant, color, label in ((VARIANTS[0], "#2563eb", "weight 2"),
                                      (VARIANTS[1], "#d97706", "weight 0")):
            line = transform(project, candidates[(variant, case)][1])
            x, y = line.xy
            ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000,
                    color=color, lw=1.2, label=label)
        p0 = score_by_key[(case, VARIANTS[0])]["symmetric_p95_m"] / 1000
        p1 = score_by_key[(case, VARIANTS[1])]["symmetric_p95_m"] / 1000
        ax.set_title(f"{case}\n双向P95：{p0:.1f} / {p1:.1f} km")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("居民区软惩罚反事实｜参考线仅用于候选冻结后的描述性评分")
    figure_path = POSTHOC / "settlement_counterfactual_12_overlay.png"
    figure_buffer = BytesIO()
    fig.savefig(figure_buffer, format="png", dpi=170)
    plt.close(fig)

    verify_records(manifest["candidate_and_input_files"])
    if MANIFEST.read_bytes() != manifest_bytes:
        raise SystemExit("candidate manifest changed during scoring")
    for case, components in reference_hashes.items():
        for item in components:
            path = Path(item["path"])
            if (not path.is_file() or path.stat().st_size != item["bytes"]
                    or sha256(path) != item["sha256"]):
                raise SystemExit(f"reference changed during scoring: {path}")
    POSTHOC.mkdir()
    figure_path.write_bytes(figure_buffer.getvalue())
    report = {
        "candidate_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "tool_sha256": own_record["sha256"],
        "reference_geometry_loaded_after_candidate_hash_verification": True,
        "reference_directory": str(REAL_DIR.resolve()),
        "reference_file_hashes": reference_hashes,
        "posthoc_route_scores": route_scores,
        "candidate_to_candidate_geometry": relative_scores,
        "frozen_mapped_settlement_proxy": proxy_scores,
        "weight_2_minus_weight_0_proxy_mean_difference_median": (
            float(np.median(paired_proxy)) if len(paired_proxy) == len(cases) else None),
        "paired_proxy_cases": len(paired_proxy),
        "proxy_excluded_cases": excluded_proxy_cases,
        "weight_0_minus_weight_2_reference_p95_difference_median_m":
            float(np.median(paired_p95)),
        "selection_or_weight_update_performed": False,
        "plot": str(figure_path.relative_to(ROOT)),
    }
    (POSTHOC / "posthoc_scores.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"scored {len(route_scores)} frozen routes; output={POSTHOC}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    global OUT, PLAN, MANIFEST, POSTHOC
    parser.add_argument("action", choices=("freeze", "score"))
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    OUT = args.experiment_dir.resolve()
    PLAN = OUT / "plan.json"
    MANIFEST = OUT / "candidate_manifest.json"
    POSTHOC = OUT / "posthoc_settlement"
    freeze() if args.action == "freeze" else score()


if __name__ == "__main__":
    main()
