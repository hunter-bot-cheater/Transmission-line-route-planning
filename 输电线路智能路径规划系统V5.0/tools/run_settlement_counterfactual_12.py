"""Pre-register and run a paired 12-case settlement-weight experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "settlement_counterfactual_12_20261025"
ENDPOINTS = Path(r"D:\大创\data\real_cases_tw\real_cases.csv")
OSM_DIR = Path(r"D:\大创\V5_final\data\downloaded")
HAZARD_DIR = ROOT / "data" / "official_sources" / "ardswc_115_mvt_z10_20261023"
DEM_RELATIVE = Path("data/taiwan_dem_35m_georef_fixed.tif")
WEIGHTS = (("settlement_weight_2", 2.0), ("settlement_weight_0", 0.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def git_head_and_require_clean() -> str:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).strip()
    if dirty:
        raise SystemExit("run requires a clean committed worktree")
    return head


def verify_hazard_inventory(files: list[Path]) -> None:
    checksums = {}
    for line in (HAZARD_DIR / "tiles.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        checksums[relative.strip().replace("\\", "/")] = digest.lower()
    pbf_files = [path for path in files if path.suffix.lower() == ".pbf"]
    if not pbf_files:
        raise SystemExit("official hazard package contains no PBF files")
    for path in pbf_files:
        relative = path.relative_to(HAZARD_DIR).as_posix()
        if checksums.get(relative) != sha256(path):
            raise SystemExit(f"official hazard checksum failed: {relative}")
    if len(checksums) != len(pbf_files):
        raise SystemExit("tiles.sha256 inventory does not match downloaded PBF inventory")


def package_versions() -> dict:
    versions = {}
    for package in ("numpy", "shapely", "pyproj", "rasterio", "scipy", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        import rasterio
        gdal_version = rasterio.__gdal_version__
    except ImportError:
        gdal_version = None
    try:
        import rasterio
        rasterio_version = rasterio.__version__
    except ImportError:
        rasterio_version = None
    return {"python": sys.version, "platform": platform.platform(),
            "packages": versions, "gdal": gdal_version,
            "rasterio": rasterio_version}


def endpoint_cases(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"name", "start_lat", "start_lon", "end_lat", "end_lon"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"endpoint CSV must contain columns: {sorted(required)}")
    names = [row["name"].strip() for row in rows]
    if len(names) != 12 or any(not name for name in names) or len(set(names)) != 12:
        raise SystemExit("endpoint CSV must contain exactly 12 unique non-empty case names")
    for index, row in enumerate(rows, start=2):
        try:
            start_lat, start_lon, end_lat, end_lon = (
                float(row[key]) for key in ("start_lat", "start_lon", "end_lat", "end_lon"))
        except (TypeError, ValueError) as error:
            raise SystemExit(f"invalid endpoint coordinates at CSV row {index}") from error
        if (not all(map(math.isfinite, (start_lat, start_lon, end_lat, end_lon)))
                or not (-90 <= start_lat <= 90 and -90 <= end_lat <= 90)
                or not (-180 <= start_lon <= 180 and -180 <= end_lon <= 180)):
            raise SystemExit(f"endpoint coordinates out of range at CSV row {index}")
    return names


def planner_args(name: str, settlement_weight: float, quarantine: Path) -> list[str]:
    return [
        "--dem", str((ROOT / DEM_RELATIVE).resolve()),
        "--osm-cache-dir", str(OSM_DIR.resolve()),
        "--real-cases-csv", str(ENDPOINTS.resolve()),
        "--real-dir", str(quarantine.resolve()),
        "--out-dir", str((OUT / name).resolve()),
        "--settlement-weight", str(settlement_weight),
        "--official-hazards-dir", str(HAZARD_DIR.resolve()),
        "--official-hazard-weight", "1.0",
        "--terrain-slope-weight", "0.75",
        "--roughness-weight", "0.25",
        "--ridge-penalty-weight", "0.1",
        "--landcover-penalty-weight", "0.0",
        "--grade-penalty-weight", "1.0",
        "--turn-penalty-weight", "0.0",
        "--astar-heuristic-weight", "1.0",
        "--building-clearance-m", "0.0",
        "--max-side", "4096",
        "--no-vin", "--no-ppp",
    ]


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f"experiment output directory already contains files: {OUT}")
    head = git_head_and_require_clean()
    endpoint_record = file_record(ENDPOINTS)
    case_ids = endpoint_cases(ENDPOINTS)
    dem_record = file_record(ROOT / DEM_RELATIVE)
    osm_files = sorted(path for path in OSM_DIR.iterdir() if path.is_file())
    if not osm_files:
        raise SystemExit(f"OSM cache directory is empty: {OSM_DIR}")
    required_osm = {"taiwan_buildings.pkl", "taiwan_landuse.pkl"}
    if not required_osm.issubset({path.name for path in osm_files}):
        raise SystemExit("building and land-use OSM cache files are required")
    hazard_files = sorted(path for path in HAZARD_DIR.rglob("*") if path.is_file())
    verify_hazard_inventory(hazard_files)
    source_files = sorted((ROOT / "transmission_path_planner").rglob("*.py"))
    runner_record = file_record(Path(__file__).resolve())
    scorer_record = file_record(ROOT / "tools" / "settlement_counterfactual_12.py")
    quarantine = OUT / "reference_quarantine"
    arguments = {name: planner_args(name, weight, quarantine)
                 for name, weight in WEIGHTS}
    plan = {
        "experiment": "settlement_weight_counterfactual_12_20261025",
        "hypothesis": (
            "settlement_weight=2 reduces the mapped OSM settlement proxy versus 0; "
            "all other candidate-generation inputs and parameters are fixed"),
        "cases": 12,
        "case_ids": case_ids,
        "endpoint_csv": str(ENDPOINTS.resolve()),
        "endpoint_csv_sha256": endpoint_record["sha256"],
        "dem": DEM_RELATIVE.as_posix(),
        "dem_sha256": dem_record["sha256"],
        "max_side": 4096,
        "osm_cache_dir": str(OSM_DIR.resolve()),
        "osm_cache_files": [
            {"relative_path": path.name, "bytes": path.stat().st_size,
             "sha256": sha256(path)} for path in osm_files],
        "official_hazards_dir": str(HAZARD_DIR.resolve()),
        "hazard_files": [
            {"relative_path": path.relative_to(HAZARD_DIR).as_posix(),
             "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in hazard_files],
        "planner_source_files": [
            {"relative_path": path.relative_to(ROOT).as_posix(),
             "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_files],
        "git_head": head,
        "generation_worktree_clean": True,
        "runner_script": {"path": str(Path(__file__).resolve()),
                          "sha256": runner_record["sha256"]},
        "freeze_score_script": {"path": str((ROOT / "tools" / "settlement_counterfactual_12.py").resolve()),
                                "sha256": scorer_record["sha256"]},
        "environment": package_versions(),
        "variants": [{"name": name, "weight": weight} for name, weight in WEIGHTS],
        "planner_cli_args_by_variant": arguments,
        "min_proxy_complete_cases_for_overall_summary": 12,
        "effective_parameters": {
            "settlement_weight": "as listed in variants",
            "official_hazard_weight": 1.0,
            "terrain_slope_weight": 0.75,
            "roughness_weight": 0.25,
            "ridge_penalty_weight": 0.1,
            "landcover_penalty_weight": 0.0,
            "grade_penalty_weight": 1.0,
            "turn_penalty_weight": 0.0,
            "astar_heuristic_weight": 1.0,
            "building_clearance_m": 0.0,
            "max_ground_grade": None,
            "max_surface_slope_deg": None,
            "max_terrain_elevation_m": None,
            "max_side": 4096,
            "costunet": False,
            "vin": False,
            "ppo": False,
            "tower_model": False,
            "official_hazards_dir": str(HAZARD_DIR.resolve()),
            "additional_constraint_geojson": None,
        },
        "reference_geometry_in_candidate_inputs": False,
        "reference_dir_policy": "empty reference_quarantine at startup; fixed endpoint CSV only",
        "primary_settlement_metric": (
            "mapped_settlement_proxy_mean over all-touched route cells; "
            "mapped-feature proxy, incomplete map coverage is not population exposure"),
        "scoring": (
            "freeze candidate and input hashes before loading historical geometry; "
            "post-hoc scores are descriptive and never select routes or update weights"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    quarantine.mkdir()
    (OUT / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plan_sha256 = sha256(OUT / "plan.json")
    print(f"pre-registered; {len(hazard_files)} official hazard files and "
          f"{len(osm_files)} OSM cache files are hashed")

    for name, _ in WEIGHTS:
        output = OUT / name
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"variant output directory is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-B", "-m", "transmission_path_planner",
                   *arguments[name]]
        log = OUT / f"{name}_console.log"
        print(f"running {name}; output log: {log}", flush=True)
        verify_inputs_unchanged(plan, plan_sha256)
        with log.open("w", encoding="utf-8", newline="") as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                    stderr=subprocess.STDOUT, check=False)
        verify_inputs_unchanged(plan, plan_sha256)
        if result.returncode:
            raise SystemExit(f"planner failed for {name}: exit={result.returncode}")
    print(f"both arms completed: {OUT}")


def verify_inputs_unchanged(plan: dict, plan_sha256: str) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, text=True).strip()
    if head != plan["git_head"] or dirty:
        raise SystemExit("Git state changed during candidate generation")
    if package_versions() != plan["environment"]:
        raise SystemExit("runtime environment changed during candidate generation")
    if sha256(OUT / "plan.json") != plan_sha256:
        raise SystemExit("pre-registration plan changed during candidate generation")
    quarantine = OUT / "reference_quarantine"
    if not quarantine.is_dir() or any(quarantine.rglob("*")):
        raise SystemExit("reference quarantine changed during candidate generation")
    if sha256(ENDPOINTS) != plan["endpoint_csv_sha256"]:
        raise SystemExit("endpoint CSV changed during candidate generation")
    if sha256(ROOT / DEM_RELATIVE) != plan["dem_sha256"]:
        raise SystemExit("DEM changed during candidate generation")
    actual_osm = sorted(path.name for path in OSM_DIR.iterdir() if path.is_file())
    if actual_osm != sorted(item["relative_path"] for item in plan["osm_cache_files"]):
        raise SystemExit("OSM cache file set changed during candidate generation")
    for item in plan["osm_cache_files"]:
        path = OSM_DIR / item["relative_path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise SystemExit(f"OSM cache changed during candidate generation: {path}")
    actual_hazards = sorted(path for path in HAZARD_DIR.rglob("*") if path.is_file())
    if len(actual_hazards) != len(plan["hazard_files"]):
        raise SystemExit("official hazard file set changed during candidate generation")
    by_relative_path = {item["relative_path"]: item for item in plan["hazard_files"]}
    for path in actual_hazards:
        relative = path.relative_to(HAZARD_DIR).as_posix()
        item = by_relative_path.get(relative)
        if (item is None or path.stat().st_size != item["bytes"]
                or sha256(path) != item["sha256"]):
            raise SystemExit(f"official hazard source changed during candidate generation: {path}")
    source_files = sorted((ROOT / "transmission_path_planner").rglob("*.py"))
    if {path.relative_to(ROOT).as_posix() for path in source_files} != {
            item["relative_path"] for item in plan["planner_source_files"]}:
        raise SystemExit("planner source file set changed during candidate generation")
    for item in plan["planner_source_files"]:
        path = ROOT / item["relative_path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise SystemExit(f"planner source changed during candidate generation: {path}")
    for key, path in (("runner_script", Path(__file__).resolve()),
                      ("freeze_score_script", ROOT / "tools" / "settlement_counterfactual_12.py")):
        if sha256(path) != plan[key]["sha256"]:
            raise SystemExit(f"experiment tool changed during candidate generation: {path}")


if __name__ == "__main__":
    main()
