#!/usr/bin/env python3
"""Freeze completed SWC A/B route outputs before opening reference geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ARMS = ("control_20261030_01", "swc_soft_20261030_01")
CASE_IDS = tuple(f"B12-{index:02d}" for index in range(1, 13))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_geometry_hash(geometry: dict) -> str:
    kind = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if kind == "LineString":
        lines = [coordinates]
    elif kind == "MultiLineString":
        lines = coordinates
    else:
        raise ValueError(f"unsupported route geometry type: {kind!r}")
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
    encoded = json.dumps({"type": kind, "lines": normalized}, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_arm(root: Path, arm: str) -> dict:
    result_dir = root / f"results_{arm}"
    summary_path = result_dir / "run_summary.json"
    provenance_path = result_dir / "run_provenance.json"
    if not (result_dir / "exit_status.txt").is_file():
        raise ValueError(f"missing exit status for {arm}")
    if (result_dir / "exit_status.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError(f"run failed for {arm}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("complete") is not True
            or summary.get("expected_cases") != 12
            or summary.get("completed_cases") != 12
            or summary.get("failed_cases")
            or summary.get("no_route_cases")):
        raise ValueError(f"incomplete run for {arm}")
    if not provenance_path.is_file():
        raise ValueError(f"missing provenance for {arm}")

    candidates = []
    for case in CASE_IDS:
        matches = sorted(result_dir.glob(f"{case}-*/final_route.geojson"))
        if len(matches) != 1:
            raise ValueError(f"expected one final route for {arm}/{case}; found {len(matches)}")
        route_path = matches[0]
        document = json.loads(route_path.read_text(encoding="utf-8"))
        features = document.get("features")
        if (document.get("type") != "FeatureCollection"
                or not isinstance(features, list) or len(features) != 1):
            raise ValueError(f"invalid route feature collection: {route_path}")
        feature = features[0]
        if (not isinstance(feature, dict)
                or feature.get("properties", {}).get("case") != case):
            raise ValueError(f"candidate case ID mismatch: {route_path}")
        geometry = feature.get("geometry")
        if not isinstance(geometry, dict):
            raise ValueError(f"missing candidate geometry: {route_path}")
        rel = route_path.relative_to(root).as_posix()
        candidates.append({
            "case_id": case,
            "path": rel,
            "file_sha256": sha256(route_path),
            "canonical_geometry_sha256_9dp_direction_invariant":
                canonical_geometry_hash(geometry),
        })

    return {
        "schema_version": 1,
        "arm": arm,
        "status": "complete",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "run_complete": True,
        "candidate_routes_frozen_before_reference_geometry_read": True,
        "reference_geometry_read": False,
        "run_summary_sha256": sha256(summary_path),
        "run_provenance_sha256": sha256(provenance_path),
        "supporting_file_sha256": {
            f"results_{arm}/run_summary.json": sha256(summary_path),
            f"results_{arm}/run_provenance.json": sha256(provenance_path),
            f"results_{arm}/exit_status.txt": sha256(result_dir / "exit_status.txt"),
        },
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("experiment root must be a directory")
    output_paths = []
    for arm in ARMS:
        output = root / f"candidate_freeze_{arm}_20261030.json"
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
        payload = freeze_arm(root, arm)
        data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with output.open("x", encoding="utf-8") as stream:
            stream.write(data)
        output_paths.append(str(output))
    print(json.dumps({"status": "frozen", "outputs": output_paths},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
