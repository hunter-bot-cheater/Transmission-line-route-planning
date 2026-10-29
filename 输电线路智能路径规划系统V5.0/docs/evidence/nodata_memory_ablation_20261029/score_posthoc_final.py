"""Hash-gated, server-side comparison against references after route freeze."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / "freeze.json").read_text(encoding="utf-8"))
RESULTS = ROOT / "results"
FROZEN = RESULTS / "candidate_freeze.json"
SUMMARY = RESULTS / "run_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_data(path: Path) -> tuple[str, list[tuple[float, float]], dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    features = document.get("features", [])
    if len(features) != 1:
        raise ValueError(f"expected one feature in {path}")
    feature = features[0]
    geometry = feature.get("geometry") or {}
    if geometry.get("type") != "LineString":
        raise ValueError(f"expected a LineString in {path}")
    coordinates = [(float(point[0]), float(point[1]))
                   for point in geometry.get("coordinates", [])]
    if len(coordinates) < 2:
        raise ValueError(f"route is too short: {path}")
    canonical = json.dumps(
        geometry, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")
    properties = feature.get("properties")
    return hashlib.sha256(canonical).hexdigest(), coordinates, properties if isinstance(properties, dict) else {}


def reference_lines(path: Path) -> list[list[tuple[float, float]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    lines: list[list[tuple[float, float]]] = []

    def collect(geometry: dict) -> None:
        kind = geometry.get("type")
        coordinates = geometry.get("coordinates", [])
        if kind == "LineString" and len(coordinates) >= 2:
            lines.append([(float(point[0]), float(point[1])) for point in coordinates])
        elif kind == "MultiLineString":
            for part in coordinates:
                if len(part) >= 2:
                    lines.append([(float(point[0]), float(point[1])) for point in part])
        elif kind == "GeometryCollection":
            for child in geometry.get("geometries", []):
                collect(child)

    for feature in document.get("features", []):
        if feature.get("geometry"):
            collect(feature["geometry"])
    if not lines:
        raise ValueError(f"no line geometry in reference: {path}")
    return lines


if not FROZEN.is_file() or not SUMMARY.is_file():
    raise SystemExit("candidate_freeze.json and run_summary.json are required")
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
expected = len(PLAN["cases"]) * len(PLAN["arms"])
if (summary.get("status") != "complete"
        or summary.get("experiment_id") != PLAN["experiment_id"]
        or frozen.get("experiment_id") != PLAN["experiment_id"]
        or frozen.get("candidate_count") != expected
        or len(frozen.get("candidates", [])) != expected
        or frozen.get("reference_geometry_read") is not False):
    raise SystemExit("route run is incomplete or candidate freeze is invalid")
if summary.get("candidate_freeze_sha256") != sha256(FROZEN):
    raise SystemExit("candidate freeze hash differs from run summary")

# Verify every route artifact before opening any historical route geometry.
candidate_routes = {}
results_root = RESULTS.resolve()
for item in frozen["candidates"]:
    route_path = Path(item["candidate"]["route_path"]).resolve()
    if results_root not in route_path.parents:
        raise SystemExit(f"candidate escapes experiment results: {route_path}")
    if sha256(route_path) != item["candidate"]["route_file_sha256"]:
        raise SystemExit(f"candidate file hash mismatch: {route_path}")
    geometry_hash, coordinates, properties = route_data(route_path)
    if geometry_hash != item["candidate"]["route_geometry_sha256"]:
        raise SystemExit(f"candidate geometry hash mismatch: {route_path}")
    pair = (item["case_id"], item["arm"])
    if pair in candidate_routes:
        raise SystemExit(f"duplicate frozen candidate: {pair}")
    candidate_routes[pair] = {
        "path": route_path,
        "coordinates": coordinates,
        "properties": properties,
    }
if len(candidate_routes) != expected:
    raise SystemExit(f"expected {expected} distinct frozen candidates")
print(f"Verified {expected} frozen candidate routes; opening references now.", flush=True)

reference_map = PLAN["reference_sources_posthoc_only"]
if set(reference_map) != {case["case_id"] for case in PLAN["cases"]}:
    raise SystemExit("reference map does not match frozen case ids")
for reference_path in map(Path, reference_map.values()):
    if not reference_path.is_file():
        raise SystemExit(f"post-hoc reference is unavailable: {reference_path}")

source_root = ROOT / "sources" / "chunked_full_nodata_fix"
sys.path.insert(0, str(source_root))
from transmission_path_planner.metrics import compute_deviation  # noqa: E402

posthoc = RESULTS / "posthoc_scored"
if posthoc.exists():
    raise SystemExit(f"preserving existing post-hoc output: {posthoc}")
posthoc.mkdir()
records = []
case_geometry = {}
for case in PLAN["cases"]:
    case_id = case["case_id"]
    reference_path = Path(reference_map[case_id])
    lines = reference_lines(reference_path)
    case_geometry[case_id] = {"reference": lines, "arms": {}}
    for arm in PLAN["arms"]:
        candidate = candidate_routes[(case_id, arm["arm"])]
        result = compute_deviation(candidate["coordinates"], lines)
        if result is None:
            raise RuntimeError(f"cannot score {case_id}/{arm['arm']}")
        similarity = result["route_similarity"]
        symmetric = similarity["symmetric"]
        length = similarity["length"]
        record = {
            "case_id": case_id,
            "arm": arm["arm"],
            "endpoint_qualification": PLAN["endpoint_qualification"],
            "reference_geojson": str(reference_path),
            "reference_sha256": sha256(reference_path),
            "planned_route": str(candidate["path"]),
            "planning_status": candidate["properties"].get("planning_status"),
            "planned_to_reference_p95_m": similarity["planned_to_reference"]["p95_m"],
            "reference_to_planned_p95_m": similarity["reference_to_planned"]["p95_m"],
            "symmetric_p95_m": symmetric["p95_m"],
            "planned_to_reference_coverage_100m": similarity[
                "planned_to_reference"]["coverage_within_m"]["100"],
            "reference_to_planned_coverage_100m": similarity[
                "reference_to_planned"]["coverage_within_m"]["100"],
            "f1_100m": symmetric["coverage_f1_within_m"]["100"],
            "planned_length_km": length["planned_km"],
            "reference_length_km": length["reference_km"],
            "length_error_pct": length.get("absolute_difference_pct"),
            "discrete_frechet_approx": similarity["discrete_frechet_approx"],
            "buffer_iou": similarity["buffer_iou"],
            "route_similarity": similarity,
        }
        records.append(record)
        case_geometry[case_id]["arms"][arm["arm"]] = candidate["coordinates"]

result_json = {
    "experiment_id": PLAN["experiment_id"],
    "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    "planning_used_historical_routes": False,
    "reference_geometry_read_after_candidate_freeze": True,
    "metric": "bidirectional arc-length sampled geometric similarity",
    "records": records,
}
json_path = posthoc / "route_similarity_posthoc.json"
json_path.write_text(json.dumps(result_json, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
csv_path = posthoc / "route_similarity_posthoc.csv"
columns = [key for key in records[0] if key != "route_similarity"]
with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pyproj import Transformer

project = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True).transform
colors = {
    "baseline": "#d95f02",
    "terrain_nodata_only": "#1b9e77",
    "full_nodata_fix": "#7570b3",
    "chunked_full_nodata_fix": "#e7298a",
}
for case_id, geometries in case_geometry.items():
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)

    def plot_lines(lines, color, linewidth, linestyle="-", zorder=2):
        for line in lines:
            points = [project(lon, lat) for lon, lat in line]
            ax.plot([point[0] / 1000 for point in points],
                    [point[1] / 1000 for point in points],
                    color=color, linewidth=linewidth, linestyle=linestyle,
                    zorder=zorder)

    plot_lines(geometries["reference"], "#111827", 2.8, zorder=5)
    for arm in PLAN["arms"]:
        plot_lines([geometries["arms"][arm["arm"]]], colors[arm["arm"]], 1.8)
    legend = [Line2D([0], [0], color="#111827", lw=2.8, label="实际线路(事后对比)")]
    legend.extend(Line2D([0], [0], color=colors[arm["arm"]], lw=1.8,
                         label=arm["arm"]) for arm in PLAN["arms"])
    ax.legend(handles=legend, loc="best", fontsize=8)
    ax.set_title(f"{case_id}: planned routes vs reference · EPSG:3826")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.grid(True, alpha=0.2)
    fig.savefig(posthoc / f"{case_id}_route_overlay.png", dpi=180)
    plt.close(fig)

print(f"POSTHOC complete: {json_path}; maps={len(case_geometry)}", flush=True)
