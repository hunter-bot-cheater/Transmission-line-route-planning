"""Freeze completed route outputs, then recompute their post-hoc geometry metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from shapely.geometry import LineString, MultiLineString, shape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transmission_path_planner.metrics import compute_deviation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_parts(path: Path) -> list[list[tuple[float, float]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for feature in document.get("features", []):
        geometry = shape(feature["geometry"])
        if isinstance(geometry, LineString):
            lines.append(list(geometry.coords))
        elif isinstance(geometry, MultiLineString):
            lines.extend(list(part.coords) for part in geometry.geoms if part.length > 0)
    if not lines:
        raise ValueError(f"no line geometry in {path}")
    return lines


def case_name(path: Path) -> str:
    document = json.loads(path.read_text(encoding="utf-8"))
    features = document.get("features", [])
    if len(features) != 1:
        raise ValueError(f"expected one final route feature in {path}")
    return str(features[0]["properties"]["case"])


def freeze(results_dir: Path) -> Path:
    source_manifest = results_dir / "input_manifest.json"
    candidates = []
    for folder in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        final_route = folder / "final_route.geojson"
        planned_route = folder / "planned_route_aastar.geojson"
        if not final_route.is_file() or not planned_route.is_file():
            continue
        candidates.append({
            "case": case_name(final_route),
            "final_route": str(final_route.resolve()),
            "final_route_sha256": sha256(final_route),
            "astar_route": str(planned_route.resolve()),
            "astar_route_sha256": sha256(planned_route),
        })
    if not candidates or len({item["case"] for item in candidates}) != len(candidates):
        raise SystemExit("no completed routes or duplicate case names")
    manifest = {
        "source_run_dir": str(results_dir.resolve()),
        "source_input_manifest": (str(source_manifest.resolve())
                                   if source_manifest.is_file() else None),
        "source_input_manifest_sha256": sha256(source_manifest)
        if source_manifest.is_file() else None,
        "candidate_count": len(candidates),
        "reference_geometry_read": False,
        "candidates": candidates,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = results_dir / "metric_v5_candidate_freeze.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return path


def score(results_dir: Path) -> tuple[Path, Path]:
    freeze_path = results_dir / "metric_v5_candidate_freeze.json"
    manifest = json.loads(freeze_path.read_text(encoding="utf-8"))
    if manifest.get("reference_geometry_read") is not False:
        raise SystemExit("freeze manifest does not confirm pre-score candidate isolation")
    candidates = manifest["candidates"]
    for item in candidates:
        for key, hash_key in (("final_route", "final_route_sha256"),
                              ("astar_route", "astar_route_sha256")):
            path = Path(item[key])
            if not path.is_file() or sha256(path) != item[hash_key]:
                raise SystemExit(f"frozen candidate changed: {path}")

    rows = []
    reference_hashes = {}
    for item in candidates:
        folder = Path(item["final_route"]).parent
        reference_path = folder / "real_route.geojson"
        if not reference_path.is_file():
            raise SystemExit(f"missing post-hoc reference geometry: {reference_path}")
        reference_hashes[item["case"]] = sha256(reference_path)
        planned = line_parts(Path(item["final_route"]))
        reference = line_parts(reference_path)
        planned_geometry = planned[0] if len(planned) == 1 else planned
        reference_geometry = reference[0] if len(reference) == 1 else reference
        result = compute_deviation(planned_geometry, reference_geometry)
        if result is None:
            raise SystemExit(f"metric computation failed for {item['case']}")
        rows.append({"case": item["case"], "A*(baseline)": {
            "method": "A*(baseline)", **result,
        }})

    for item in candidates:
        if sha256(Path(item["final_route"])) != item["final_route_sha256"]:
            raise SystemExit(f"candidate changed during scoring: {item['case']}")

    metric_versions = {row["A*(baseline)"]["route_similarity_version"] for row in rows}
    if len(metric_versions) != 1:
        raise SystemExit(f"inconsistent route similarity metric versions: {sorted(metric_versions)}")
    metric_version = metric_versions.pop()
    metrics_path = results_dir / f"real_comparison_metrics_v{metric_version}.json"
    metrics_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    score_manifest = {
        "candidate_freeze_sha256": sha256(freeze_path),
        "candidate_count": len(candidates),
        "metric_version": metric_version,
        "reference_geometry_read_after_candidate_hash_verification": True,
        "reference_route_geojson_sha256": reference_hashes,
        "metrics_sha256": sha256(metrics_path),
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    score_manifest_path = results_dir / f"metric_v{metric_version}_posthoc_manifest.json"
    score_manifest_path.write_text(
        json.dumps(score_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return metrics_path, score_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score"))
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    if args.mode == "freeze":
        print(f"Frozen candidate routes: {freeze(results_dir)}")
    else:
        metrics_path, manifest_path = score(results_dir)
        print(f"Post-hoc metrics: {metrics_path}")
        print(f"Score manifest: {manifest_path}")


if __name__ == "__main__":
    main()
