"""Run the frozen turn-penalty 2x2 experiment without reading route references."""

import hashlib
import importlib.metadata
import json
import os
import platform
import psutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = Path(__file__).with_name("freeze.json")
PLAN = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
OUT = ROOT / PLAN["out_dir"]
RUNS = OUT / "runs"
LOGS = OUT / "logs"
SUMMARY_PATH = OUT / "run_summary.json"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_summary(summary, status):
    summary["status"] = status
    summary["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def installed_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def candidate_paths(run_dir):
    final_paths = list(run_dir.rglob("final_route.geojson"))
    astar_paths = list(run_dir.rglob("planned_route_aastar.geojson"))
    raw_astar_paths = list(run_dir.rglob("astar_raw_route.geojson"))
    if (len(final_paths) != 1 or len(astar_paths) != 1
            or len(raw_astar_paths) != 1):
        raise ValueError(
            f"expected one final and A* route in {run_dir}; "
            f"found {len(final_paths)} final, {len(astar_paths)} post-A*, "
            f"and {len(raw_astar_paths)} raw A* routes")
    return final_paths[0], astar_paths[0], raw_astar_paths[0]


def route_geometry_hash(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    features = document.get("features", [])
    if len(features) != 1 or features[0].get("geometry", {}).get("type") != "LineString":
        raise ValueError(f"expected exactly one LineString in {path}")
    canonical = json.dumps(
        features[0]["geometry"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


if OUT.exists():
    raise SystemExit(f"preserving existing experiment output; refusing to overwrite: {OUT}")

input_hashes = {}
for name, record in PLAN["inputs"].items():
    path = Path(record["path"])
    if not path.is_file():
        raise SystemExit(f"missing frozen input: {name}: {path}")
    actual = sha256(path)
    if actual != record["sha256"]:
        raise SystemExit(f"frozen input changed: {name}: {actual}")
    input_hashes[name] = actual

revision_marker = ROOT / "SOURCE_COMMIT"
if not revision_marker.is_file():
    raise SystemExit(f"missing source revision marker: {revision_marker}")
head = revision_marker.read_text(encoding="utf-8").strip()
if head != PLAN["source_git_head"]:
    raise SystemExit(f"source revision marker mismatch: {head}")

source_hashes = {}
for name, expected in PLAN["runtime_source_sha256"].items():
    path = ROOT / name
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"frozen source changed: {name}: {actual}")
    source_hashes[name] = actual

requirements_hash = sha256(ROOT / "requirements.txt")
if requirements_hash != PLAN["requirements_sha256"]:
    raise SystemExit(f"frozen requirements changed: {requirements_hash}")
runner_hash = sha256(Path(__file__))
if runner_hash != PLAN["runner_sha256"]:
    raise SystemExit(f"frozen experiment runner changed: {runner_hash}")

OUT.mkdir(parents=True)
RUNS.mkdir()
LOGS.mkdir()
summary = {
    "experiment_id": PLAN["experiment_id"],
    "status": "in_progress",
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "cpu": platform.processor(),
    "runtime_dependency_versions": installed_versions((
        "numpy", "rasterio", "shapely", "geopandas", "pyproj", "scipy",
        "matplotlib", "torch", "affine", "psutil")),
    "freeze_sha256": sha256(PLAN_PATH),
    "verified_input_sha256": input_hashes,
    "verified_runtime_source_sha256": source_hashes,
    "verified_requirements_sha256": requirements_hash,
    "verified_runner_sha256": runner_hash,
    "verified_source_git_head": head,
    "gpu_used": False,
    "gpu_reason": "A* search is CPU-bound; AI diagnostics are disabled and CUDA is hidden for each run.",
    "reference_geometry_read": False,
    "runs": [],
}
save_summary(summary, "in_progress")

dem = Path(PLAN["inputs"]["dem"]["path"])
constraints = Path(PLAN["inputs"]["constraints"]["path"])
land_mask = Path(PLAN["inputs"]["land_mask"]["path"])
hazards_dir = Path(PLAN["inputs"]["hazard_manifest"]["path"]).parent
osm_cache = Path(PLAN["inputs"]["osm_landuse"]["path"]).parent
fixed = PLAN["fixed_parameters"]

for case in PLAN["cases"]:
    for arm in PLAN["arms"]:
        tag = f"{case['case_id']}__{arm['arm']}"
        run_dir = RUNS / case["case_id"] / arm["arm"]
        log_path = LOGS / f"{tag}.log"
        if run_dir.exists() or log_path.exists():
            save_summary(summary, "paused_existing_output_preserved")
            raise SystemExit(f"preserving prior output; refusing to overwrite: {tag}")

        argv = [
            sys.executable, "-m", "transmission_path_planner.cli",
            "--dem", str(dem),
            "--constraint-geojson", str(constraints),
            "--land-mask-geojson", str(land_mask),
            "--official-hazards-dir", str(hazards_dir),
            "--official-hazard-weight", str(fixed["official_hazard_weight"]),
            "--settlement-weight", str(fixed["settlement_weight"]),
            "--ridge-penalty-weight", str(fixed["ridge_penalty_weight"]),
            "--building-clearance-m", str(fixed["building_clearance_m"]),
            "--landcover-penalty-weight", str(fixed["landcover_penalty_weight"]),
            "--grade-penalty-weight", str(fixed["grade_penalty_weight"]),
            "--terrain-slope-weight", str(fixed["terrain_slope_weight"]),
            "--roughness-weight", str(fixed["roughness_weight"]),
            "--turn-penalty-weight", str(arm["legacy_turn_penalty_weight"]),
            "--astar-heuristic-weight", str(fixed["astar_heuristic_weight"]),
            "--max-side", str(fixed["max_side"]),
            "--window-margin-deg", str(fixed["window_margin_deg"]),
            "--osm-cache-dir", str(osm_cache),
            "--no-ppp", "--no-vin",
            "--start", f"{case['start'][0]:.6f},{case['start'][1]:.6f}",
            "--end", f"{case['end'][0]:.6f},{case['end'][1]:.6f}",
            "--out-dir", str(run_dir),
        ]
        if fixed["max_ground_grade"] is not None:
            argv.extend(("--max-ground-grade", str(fixed["max_ground_grade"])))
        if fixed["max_surface_slope_deg"] is not None:
            argv.extend(("--max-surface-slope-deg",
                         str(fixed["max_surface_slope_deg"])))
        if fixed["max_terrain_elevation_m"] is not None:
            argv.extend(("--max-terrain-elevation-m",
                         str(fixed["max_terrain_elevation_m"])))
        if arm["search_turn_penalty_weight"] is not None:
            argv.extend(("--search-turn-penalty-weight",
                         str(arm["search_turn_penalty_weight"])))
        if arm["smoothing_turn_penalty_weight"] is not None:
            argv.extend(("--smoothing-turn-penalty-weight",
                         str(arm["smoothing_turn_penalty_weight"])))

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = str(fixed["omp_num_threads"])
        env["OPENBLAS_NUM_THREADS"] = str(fixed["openblas_num_threads"])
        env["MKL_NUM_THREADS"] = "1"
        env["MPLBACKEND"] = "Agg"

        run_dir.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        tick = time.monotonic()
        print(f"START {tag}", flush=True)
        peak_rss_bytes = 0
        min_available_memory_bytes = None
        resource_limit_reason = None
        with log_path.open("x", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                argv, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT)
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                rss = 0
                try:
                    tree = [monitored, *monitored.children(recursive=True)]
                    rss = sum(item.memory_info().rss for item in tree
                              if item.is_running())
                    peak_rss_bytes = max(peak_rss_bytes, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                available = psutil.virtual_memory().available
                min_available_memory_bytes = (
                    available if min_available_memory_bytes is None
                    else min(min_available_memory_bytes, available))
                guard = PLAN["resource_guard"]
                if rss >= guard["max_process_tree_rss_bytes"]:
                    resource_limit_reason = "process_tree_rss_limit"
                elif available <= guard["min_available_memory_bytes"]:
                    resource_limit_reason = "system_available_memory_limit"
                if resource_limit_reason:
                    process.terminate()
                    break
                time.sleep(1.0)
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
        elapsed = time.monotonic() - tick
        run_summary_path = run_dir / "run_summary.json"
        case_summary = {}
        if run_summary_path.is_file():
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            case_summary = {key: run_summary.get(key) for key in (
                "expected_cases", "completed_cases", "recorded_cases",
                "failed_cases", "no_route_cases", "complete")}
        record = {
            "tag": tag,
            "case_id": case["case_id"],
            "arm": arm["arm"],
            "legacy_turn_penalty_weight": arm["legacy_turn_penalty_weight"],
            "search_turn_penalty_weight": arm["expected_resolved_search_weight"],
            "smoothing_turn_penalty_weight": arm["expected_resolved_smoothing_weight"],
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(elapsed, 3),
            "returncode": returncode,
            "peak_process_tree_rss_bytes": peak_rss_bytes,
            "min_system_available_memory_bytes": min_available_memory_bytes,
            "resource_limit_reason": resource_limit_reason,
            "output_dir": str(run_dir),
            "log": str(log_path),
            "argv": argv,
            "run_summary": case_summary,
        }
        summary["runs"].append(record)
        if returncode != 0 or case_summary.get("complete") is not True:
            status = ("paused_after_resource_limit" if resource_limit_reason
                      else "paused_after_run_failure")
            save_summary(summary, status)
            print(f"PAUSE_AFTER_FAILURE {tag} exit={returncode} "
                  f"resource_limit={resource_limit_reason}", flush=True)
            raise SystemExit(1)
        save_summary(summary, "in_progress")
        print(f"DONE {tag} elapsed_s={elapsed:.1f}", flush=True)

frozen = {
    "experiment_id": PLAN["experiment_id"],
    "candidate_count": 0,
    "reference_geometry_read": False,
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_summary_sha256": sha256(SUMMARY_PATH),
    "candidates": [],
}
for record in summary["runs"]:
    run_dir = Path(record["output_dir"])
    final_path, astar_path, raw_astar_path = candidate_paths(run_dir)
    frozen["candidates"].append({
        "tag": record["tag"],
        "case_id": record["case_id"],
        "arm": record["arm"],
        "search_turn_penalty_weight": record["search_turn_penalty_weight"],
        "smoothing_turn_penalty_weight": record["smoothing_turn_penalty_weight"],
        "final_route_path": str(final_path),
        "final_route_file_sha256": sha256(final_path),
        "final_route_geometry_sha256": route_geometry_hash(final_path),
        "astar_route_path": str(astar_path),
        "astar_route_file_sha256": sha256(astar_path),
        "astar_route_geometry_sha256": route_geometry_hash(astar_path),
        "raw_astar_route_path": str(raw_astar_path),
        "raw_astar_route_file_sha256": sha256(raw_astar_path),
        "raw_astar_route_geometry_sha256": route_geometry_hash(raw_astar_path),
    })
frozen["candidate_count"] = len(frozen["candidates"])
expected_count = len(PLAN["cases"]) * len(PLAN["arms"])
if frozen["candidate_count"] != expected_count:
    save_summary(summary, "paused_candidate_freeze_count_mismatch")
    raise SystemExit(f"expected {expected_count} routes, got {frozen['candidate_count']}")
freeze_path = OUT / "candidate_freeze.json"
freeze_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
summary["candidate_freeze_sha256"] = sha256(freeze_path)
save_summary(summary, "complete")
print(f"FINAL complete candidates={frozen['candidate_count']} reference_geometry_read=false",
      flush=True)
