"""Run a frozen Taiwan DEM NoData ablation on the configured CPU server."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parent
PLAN_PATH = ROOT / "freeze.json"
PLAN = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
OUT = ROOT / "results"
RUNS = OUT / "runs"
LOGS = OUT / "logs"
SUMMARY_PATH = OUT / "run_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_summary(summary: dict, status: str) -> None:
    summary["status"] = status
    summary["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_hashes(source_root: Path) -> dict[str, str]:
    return {
        path.relative_to(source_root).as_posix(): sha256(path)
        for path in sorted(source_root.rglob("*.py"))
    }


def candidate_hashes(run_dir: Path) -> dict[str, str]:
    routes = list(run_dir.rglob("final_route.geojson"))
    maps = list(run_dir.rglob("path_comparison_*.png"))
    if len(routes) != 1:
        raise RuntimeError(f"expected one final route in {run_dir}, found {len(routes)}")
    document = json.loads(routes[0].read_text(encoding="utf-8"))
    features = document.get("features", [])
    if (len(features) != 1
            or features[0].get("geometry", {}).get("type") != "LineString"):
        raise RuntimeError(f"expected one LineString in {routes[0]}")
    geometry = json.dumps(
        features[0]["geometry"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    result = {
        "route_path": str(routes[0]),
        "route_file_sha256": sha256(routes[0]),
        "route_geometry_sha256": hashlib.sha256(geometry).hexdigest(),
    }
    if len(maps) == 1:
        result["map_path"] = str(maps[0])
        result["map_sha256"] = sha256(maps[0])
    return result


if OUT.exists():
    raise SystemExit(f"preserving existing output; refusing to overwrite: {OUT}")

verified_inputs = {}
for name, record in PLAN["inputs"].items():
    path = Path(record["path"])
    if not path.is_file():
        raise SystemExit(f"missing frozen input {name}: {path}")
    actual = sha256(path)
    if actual != record["sha256"]:
        raise SystemExit(f"frozen input changed for {name}: {actual}")
    verified_inputs[name] = actual

for arm in PLAN["arms"]:
    package = ROOT / "sources" / arm["arm"] / "transmission_path_planner"
    if not (package / "__main__.py").is_file():
        raise SystemExit(f"missing source package for {arm['arm']}: {package}")

RUNS.mkdir(parents=True)
LOGS.mkdir()
summary = {
    "experiment_id": PLAN["experiment_id"],
    "status": "in_progress",
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "cpu": platform.processor(),
    "dependency_versions": {},
    "freeze_sha256": sha256(PLAN_PATH),
    "runner_sha256": sha256(Path(__file__)),
    "verified_input_sha256": verified_inputs,
    "gpu_used": False,
    "reference_geometry_read": False,
    "runs": [],
}
for name in ("numpy", "rasterio", "shapely", "geopandas", "pyproj",
             "scipy", "matplotlib", "psutil"):
    try:
        summary["dependency_versions"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        summary["dependency_versions"][name] = None
save_summary(summary, "in_progress")

fixed = PLAN["fixed_parameters"]
for case in PLAN["cases"]:
    for arm in PLAN["arms"]:
        tag = f"{case['case_id']}__{arm['arm']}"
        run_dir = RUNS / case["case_id"] / arm["arm"]
        log_path = LOGS / f"{tag}.log"
        if run_dir.exists() or log_path.exists():
            save_summary(summary, "paused_existing_output_preserved")
            raise SystemExit(f"preserving prior run; refusing to overwrite: {tag}")

        source_root = ROOT / "sources" / arm["arm"]
        command = [
            sys.executable, "-m", "transmission_path_planner.cli",
            "--dem", PLAN["inputs"]["dem"]["path"],
            "--constraint-geojson", PLAN["inputs"]["constraints"]["path"],
            "--land-mask-geojson", PLAN["inputs"]["land_mask"]["path"],
            "--official-hazards-dir",
            str(Path(PLAN["inputs"]["hazard_manifest"]["path"]).parent),
            "--official-hazard-weight", str(fixed["official_hazard_weight"]),
            "--settlement-weight", str(fixed["settlement_weight"]),
            "--ridge-penalty-weight", str(fixed["ridge_penalty_weight"]),
            "--landcover-penalty-weight", str(fixed["landcover_penalty_weight"]),
            "--grade-penalty-weight", str(fixed["grade_penalty_weight"]),
            "--terrain-slope-weight", str(fixed["terrain_slope_weight"]),
            "--roughness-weight", str(fixed["roughness_weight"]),
            "--astar-heuristic-weight", str(fixed["astar_heuristic_weight"]),
            "--building-clearance-m", str(fixed["building_clearance_m"]),
            "--window-margin-deg", str(fixed["window_margin_deg"]),
            "--osm-cache-dir",
            str(Path(PLAN["inputs"]["osm_landuse"]["path"]).parent),
            "--max-side", str(fixed["max_side"]),
            "--no-ppp", "--no-vin",
            "--start", f"{case['start'][0]:.6f},{case['start'][1]:.6f}",
            "--end", f"{case['end'][0]:.6f},{case['end'][1]:.6f}",
            "--out-dir", str(run_dir),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source_root) + os.pathsep + env.get("PYTHONPATH", "")
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = str(fixed["omp_num_threads"])
        env["OPENBLAS_NUM_THREADS"] = str(fixed["openblas_num_threads"])
        env["MKL_NUM_THREADS"] = "1"
        env["MPLBACKEND"] = "Agg"

        run_dir.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        tick = time.monotonic()
        peak_rss = 0
        min_available = None
        resource_limit = None
        print(f"START {tag}", flush=True)
        with log_path.open("x", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                command, cwd=source_root, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            monitored = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    tree = [monitored, *monitored.children(recursive=True)]
                    rss = sum(child.memory_info().rss for child in tree if child.is_running())
                    peak_rss = max(peak_rss, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                available = psutil.virtual_memory().available
                min_available = available if min_available is None else min(min_available, available)
                if peak_rss >= PLAN["resource_guard"]["max_process_tree_rss_bytes"]:
                    resource_limit = "process_tree_rss_limit"
                elif available <= PLAN["resource_guard"]["min_available_memory_bytes"]:
                    resource_limit = "system_available_memory_limit"
                if resource_limit:
                    process.terminate()
                    break
                time.sleep(1)
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()

        record = {
            "tag": tag,
            "case_id": case["case_id"],
            "arm": arm["arm"],
            "source_revision": arm["source_revision"],
            "source_sha256": source_hashes(source_root / "transmission_path_planner"),
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.monotonic() - tick, 3),
            "returncode": returncode,
            "peak_process_tree_rss_bytes": peak_rss,
            "min_system_available_memory_bytes": min_available,
            "resource_limit_reason": resource_limit,
            "output_dir": str(run_dir),
            "log": str(log_path),
            "command": command,
            "reference_geometry_read": False,
        }
        if returncode == 0 and resource_limit is None:
            record["candidate"] = candidate_hashes(run_dir)
        summary["runs"].append(record)
        if returncode != 0 or resource_limit:
            save_summary(summary, "paused_after_resource_limit" if resource_limit
                         else "paused_after_run_failure")
            print(f"PAUSE {tag} rc={returncode} resource_limit={resource_limit}", flush=True)
            raise SystemExit(1)
        save_summary(summary, "in_progress")
        print(f"DONE {tag} elapsed_s={record['elapsed_s']:.1f} "
              f"peak_rss_gib={peak_rss / 1024**3:.2f}", flush=True)

expected = len(PLAN["cases"]) * len(PLAN["arms"])
if len(summary["runs"]) != expected:
    save_summary(summary, "paused_candidate_count_mismatch")
    raise SystemExit(f"expected {expected} routes, got {len(summary['runs'])}")
freeze = {
    "experiment_id": PLAN["experiment_id"],
    "candidate_count": expected,
    "reference_geometry_read": False,
    "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_summary_sha256": sha256(SUMMARY_PATH),
    "candidates": [
        {key: record[key] for key in (
            "tag", "case_id", "arm", "source_revision", "source_sha256", "candidate")}
        for record in summary["runs"]
    ],
}
freeze_path = OUT / "candidate_freeze.json"
freeze_path.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
summary["candidate_freeze_sha256"] = sha256(freeze_path)
save_summary(summary, "complete")
print(f"FINAL complete candidates={expected} reference_geometry_read=false", flush=True)
