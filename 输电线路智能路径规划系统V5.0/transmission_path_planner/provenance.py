"""Run provenance for reproducible route-planning benchmarks."""

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
import subprocess
import sys


OSM_CACHE_FILES = {
    "roads": "taiwan_roads.pkl",
    "water": "taiwan_water.pkl",
    "railways": "taiwan_railways.pkl",
    "faults": "taiwan_faults.pkl",
    "landuse": "taiwan_landuse.pkl",
    "buildings": "taiwan_buildings.pkl",
    "vegetation": "taiwan_vegetation.pkl",
    "protected_areas": "taiwan_protected.pkl",
}
DEPENDENCIES = (
    "numpy", "scipy", "pandas", "shapely", "rasterio", "geopandas",
    "pyproj", "pyogrio", "matplotlib", "torch", "osmnx",
)
PROVENANCE_FILENAME = "run_provenance.json"


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(role, path):
    path = Path(path).resolve()
    record = {"role": role, "path": str(path)}
    if not path.is_file():
        record["status"] = "missing"
        return record
    try:
        record.update(
            status="hashed",
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
    except OSError as exc:
        record.update(status="unreadable", error=str(exc))
    return record


def _arg(args, name, default=None):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def _planning_input_records(args, dem_path, default_osm_cache_dir, model_weights,
                            has_torch):
    records = []
    for name, role in (
        ("stations_csv", "station_table"),
        ("route_tasks_csv", "route_tasks"),
        ("engineering_profile_json", "engineering_profile"),
        ("real_cases_csv", "case_tasks"),
        ("constraint_geojson", "user_constraints"),
        ("land_mask_geojson", "land_mask"),
        ("population_raster", "population_raster"),
    ):
        path = _arg(args, name)
        if path:
            records.append(_file_record(role, path))
            if name == "engineering_profile_json" and Path(path).is_file():
                try:
                    profile = json.loads(Path(path).read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError):
                    profile = {}
                for source in profile.get("sources", []):
                    if not isinstance(source, dict):
                        continue
                    local_file = source.get("local_file")
                    if (not isinstance(local_file, str) or not local_file
                            or Path(local_file).is_absolute()
                            or ".." in Path(local_file).parts):
                        continue
                    records.append(_file_record(
                        "engineering_profile_source",
                        Path(path).resolve().parent / local_file))
    records.append(_file_record("dem", dem_path))

    cache_dir = Path(_arg(args, "osm_cache_dir") or default_osm_cache_dir)
    include_model_features = bool(
        _arg(args, "include_ai_diagnostics") and has_torch
        and not _arg(args, "no_vin"))
    layer_names = list(OSM_CACHE_FILES)
    if not include_model_features:
        layer_names = ["landuse", "buildings"]
    for layer_name in layer_names:
        records.append(_file_record(
            f"osm_cache:{layer_name}", cache_dir / OSM_CACHE_FILES[layer_name]))

    hazard_dir = Path(_arg(args, "official_hazards_dir"))
    if hazard_dir.is_dir():
        for fixed_name in ("manifest.json", "tiles.sha256"):
            path = hazard_dir / fixed_name
            if path.is_file():
                records.append(_file_record("official_hazard_manifest", path))
        for path in sorted(hazard_dir.rglob("*.pbf")):
            if path.is_file() and not path.is_symlink():
                records.append(_file_record("official_hazard_tile", path))
    else:
        records.append({
            "role": "official_hazards_dir",
            "path": str(hazard_dir.resolve()),
            "status": "missing",
        })

    if _arg(args, "use_costunet") and has_torch:
        for path in model_weights:
            if Path(path).is_file():
                records.append(_file_record("costunet_weights", path))
    return records


def _source_state(repo_path):
    def git(*arguments, cwd=None):
        return subprocess.run(
            ["git", "-C", str(cwd or repo_path), *arguments],
            check=False, capture_output=True, text=True, encoding="utf-8",
            errors="replace",
        )

    try:
        root = git("rev-parse", "--show-toplevel")
        if root.returncode != 0:
            return {"available": False,
                    "reason": root.stderr.strip() or "not a Git worktree"}
        git_root = Path(root.stdout.strip())
        commit = git("rev-parse", "HEAD", cwd=git_root)
        branch = git("branch", "--show-current", cwd=git_root)
        status = git("status", "--porcelain=v1", "--untracked-files=normal", cwd=git_root)
    except OSError as exc:
        return {"available": False, "reason": str(exc)}
    if commit.returncode != 0 or status.returncode != 0:
        return {"available": False, "reason": "unable to read Git state"}
    entries = [line for line in status.stdout.splitlines() if line]
    package_dir = Path(repo_path).resolve() / "transmission_path_planner"
    if not package_dir.is_dir():
        package_dir = git_root / "transmission_path_planner"
    source_files = []
    if package_dir.is_dir():
        for path in sorted(package_dir.rglob("*.py")):
            if "__pycache__" in path.parts or path.is_symlink():
                continue
            source_files.append({
                "path": path.relative_to(git_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return {
        "available": True,
        "root": str(git_root),
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() or None,
        "dirty": bool(entries),
        "status_porcelain": entries,
        "runtime_source_files": source_files,
    }


def _runtime_info():
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": versions,
    }


def _case_records(cases):
    return [
        {
            "case": name,
            "start_latlon": _json_value(start),
            "end_latlon": _json_value(end),
        }
        for name, start, end in cases
    ]


def capture_run_provenance(repo_path, args, cases, dem_path,
                           default_osm_cache_dir, model_weights=(),
                           has_torch=True, effective_runtime=None):
    """Capture run configuration and planning inputs before route generation.

    Reference geometry paths are recorded as configuration only; this function
    never opens, hashes, or walks ``real_dir`` or ``real_shp``.
    """
    configuration = _json_value(vars(args) if hasattr(args, "__dict__") else args)
    return {
        "schema_version": 1,
        "started_at_utc": _utc_now(),
        "command": {"argv": list(sys.argv), "cwd": str(Path.cwd())},
        "configuration": configuration,
        "cases": _case_records(cases),
        "source": _source_state(repo_path),
        "runtime": _runtime_info(),
        "effective_runtime": _json_value(
            effective_runtime or {"has_torch": has_torch}),
        "planning_inputs": _planning_input_records(
            args, dem_path, default_osm_cache_dir, model_weights, has_torch),
        "reference_geometry_policy": {
            "hashed_by_provenance": False,
            "paths_are_configuration_only": True,
        },
    }


def _is_within(path, parent):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _output_records(output_dir, excluded_reference_paths=()):
    root = Path(output_dir).resolve()
    excluded = [Path(path).resolve() for path in excluded_reference_paths if path]
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == PROVENANCE_FILENAME:
            continue
        resolved = path.resolve()
        if any(resolved == item or _is_within(resolved, item) for item in excluded):
            continue
        record = {"path": path.relative_to(root).as_posix()}
        try:
            record.update(size_bytes=path.stat().st_size, sha256=sha256_file(path),
                          status="hashed")
        except OSError as exc:
            record.update(status="unreadable", error=str(exc))
        records.append(record)
    return records


def write_run_provenance(snapshot, output_dir, excluded_reference_paths=()):
    """Write the final manifest after normal outputs exist, hashing each once."""
    document = dict(snapshot)
    document["completed_at_utc"] = _utc_now()
    document["output_directory"] = str(Path(output_dir).resolve())
    document["outputs"] = _output_records(output_dir, excluded_reference_paths)
    document["output_hash_policy"] = (
        "SHA-256 of regular files under output_directory, excluding this manifest "
        "and paths identified as reference inputs")
    path = Path(output_dir) / PROVENANCE_FILENAME
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path
