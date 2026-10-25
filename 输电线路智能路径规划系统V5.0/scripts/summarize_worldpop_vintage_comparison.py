"""Validate and summarize paired A* runs with 2020/2025 WorldPop inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence" / "worldpop_vintage_3cases_cap1024_20261025"
CASES = {
    "TW173": "TW173-06a5036e04",
    "TW59": "TW59-70c0b5c4d3",
    "天輪~龍潭線": "route-b64eac92e5",
}
TREATMENTS = {
    "no_population": "关闭人口代价",
    "pop2020_w1": "2020 WorldPop，权重 1",
    "pop2025_w1": "2025 WorldPop，权重 1",
}
COLORS = {"no_population": "#2878B5", "pop2020_w1": "#E17C05", "pop2025_w1": "#3A923A"}
ALLOWED_CONFIG_DIFFS = {"out_dir", "population_raster", "population_weight"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def line_coordinates(path: Path):
    payload = read_json(path)
    result = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "LineString":
            result.append(geometry.get("coordinates", []))
        elif geometry.get("type") == "MultiLineString":
            result.extend(geometry.get("coordinates", []))
    return [line for line in result if len(line) > 1]


def configure_font():
    for path in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break


def run_dir(treatment: str, case: str) -> Path:
    return ROOT / "outputs" / f"worldpop_isolated_{treatment}_{case}"


def load_run(treatment: str, case: str):
    directory = run_dir(treatment, case)
    summary = read_json(directory / "run_summary.json")
    metric_records = read_json(directory / "comparison_metrics.json")
    provenance = read_json(directory / "run_provenance.json")
    if not summary.get("complete") or summary.get("completed_cases") != 1:
        raise ValueError(f"Incomplete run: {directory}")
    if summary.get("failed_cases") or summary.get("no_route_cases"):
        raise ValueError(f"Run has failures or no-route cases: {directory}")
    if len(metric_records) != 1 or metric_records[0].get("case") != case:
        raise ValueError(f"Unexpected case in metrics: {directory}")
    if provenance.get("source", {}).get("dirty") is not False:
        raise ValueError(f"Run was not made from a clean worktree: {directory}")
    if provenance.get("configuration", {}).get("include_ai_diagnostics"):
        raise ValueError(f"Unexpected AI diagnostic run: {directory}")
    if provenance.get("reference_geometry_policy", {}).get("reference_geometry_used_for_planning") is True:
        raise ValueError(f"Reference geometry leaked into planning: {directory}")

    reference_path = directory / CASES[case] / "real_route.geojson"
    planned_path = directory / CASES[case] / "planned_route_aastar.geojson"
    if not reference_path.is_file() or not planned_path.is_file():
        raise ValueError(f"Missing route artifact: {directory}")
    if not line_coordinates(reference_path) or not line_coordinates(planned_path):
        raise ValueError(f"Empty route geometry: {directory}")

    metric = metric_records[0]["metrics"]["A*(baseline)"]
    similarity = metric["route_similarity"]
    population = metric["population_data_coverage"]
    return {
        "case": case,
        "treatment": treatment,
        "complete": summary["complete"],
        "source_commit": provenance["source"].get("commit"),
        "source_dirty": provenance["source"].get("dirty"),
        "configuration": provenance["configuration"],
        "planning_inputs": provenance["planning_inputs"],
        "reference_sha256": sha256(reference_path),
        "planned_sha256": sha256(planned_path),
        "metrics": {
            "length_km": metric["length_km"],
            "p95_m": similarity["symmetric"]["p95_m"],
            "f1_100": similarity["symmetric"]["coverage_f1_within_m"]["100"],
            "f1_500": similarity["symmetric"]["coverage_f1_within_m"]["500"],
            "f1_1000": similarity["symmetric"]["coverage_f1_within_m"]["1000"],
            "length_difference_pct": similarity["length"]["absolute_difference_pct"],
            "max_sampled_ground_grade_pct": metric["max_sampled_ground_grade_pct"],
            "hard_violations": metric["hard_violations"],
            "hard_constraint_status": metric["hard_constraint_status"],
            "population_coverage_ratio": population["coverage_ratio"],
            "population_unknown_route_cells": population["unknown_route_cell_count"],
            "population_route_cells": population["route_cell_count"],
            "population_coverage_status": population["status"],
            "runtime_s": metric["time_s"],
        },
        "reference_path": reference_path,
        "planned_path": planned_path,
    }


def verify_pairing(records):
    by_case = {}
    for record in records:
        by_case.setdefault(record["case"], {})[record["treatment"]] = record
    expected_treatments = set(TREATMENTS)
    for case, arms in by_case.items():
        if set(arms) != expected_treatments:
            raise ValueError(f"Missing treatment for {case}: {set(arms)}")
        if len({item["source_commit"] for item in arms.values()}) != 1:
            raise ValueError(f"Source commit mismatch for {case}")
        if len({item["reference_sha256"] for item in arms.values()}) != 1:
            raise ValueError(f"Reference geometry mismatch for {case}")
        baseline = arms["no_population"]
        base_inputs = {
            (item["role"], item["path"]): item.get("sha256")
            for item in baseline["planning_inputs"]
            if item["role"] != "population_raster"
        }
        for treatment, item in arms.items():
            inputs = {
                (entry["role"], entry["path"]): entry.get("sha256")
                for entry in item["planning_inputs"]
                if entry["role"] != "population_raster"
            }
            if inputs != base_inputs:
                raise ValueError(f"Planning input mismatch for {case}/{treatment}")
            changed = {
                key for key in set(baseline["configuration"]) | set(item["configuration"])
                if baseline["configuration"].get(key) != item["configuration"].get(key)
            }
            if not changed <= ALLOWED_CONFIG_DIFFS:
                raise ValueError(f"Unexpected configuration changes for {case}/{treatment}: {changed}")


def write_outputs(records, output: Path):
    configure_font()
    output.mkdir(parents=True, exist_ok=True)
    fields = ["case", "treatment", *records[0]["metrics"].keys()]
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({"case": record["case"], "treatment": record["treatment"], **record["metrics"]})

    summary = {
        "study": "paired WorldPop vintage sensitivity; not a route-prior experiment",
        "cases": list(CASES),
        "treatments": TREATMENTS,
        "records": [
            {key: value for key, value in record.items() if key not in ("reference_path", "planned_path", "planning_inputs", "configuration")}
            for record in records
        ],
        "planning_input_hashes_by_case": {},
        "paired_medians": {},
    }
    for case in CASES:
        arms = {record["treatment"]: record for record in records if record["case"] == case}
        baseline_inputs = [
            item for item in arms["no_population"]["planning_inputs"]
            if item["role"] != "population_raster"
        ]
        population_inputs = {
            treatment: next(
                (item for item in record["planning_inputs"] if item["role"] == "population_raster"),
                None,
            )
            for treatment, record in arms.items()
        }
        summary["planning_input_hashes_by_case"][case] = {
            "common": [
                {
                    "role": item["role"],
                    "name": Path(item["path"]).name,
                    "status": item.get("status"),
                    "sha256": item.get("sha256"),
                    "size_bytes": item.get("size_bytes"),
                }
                for item in baseline_inputs
            ],
            "population_by_treatment": {
                treatment: ({
                    "name": Path(item["path"]).name,
                    "status": item.get("status"),
                    "sha256": item.get("sha256"),
                    "size_bytes": item.get("size_bytes"),
                } if item else None)
                for treatment, item in population_inputs.items()
            },
        }
    for treatment in TREATMENTS:
        selected = [record["metrics"] for record in records if record["treatment"] == treatment]
        summary["paired_medians"][treatment] = {
            field: sorted(row[field] for row in selected)[len(selected) // 2]
            for field in ("p95_m", "f1_100", "f1_500", "f1_1000", "length_difference_pct", "max_sampled_ground_grade_pct")
        }
        coverage = [row["population_coverage_ratio"] for row in selected if row["population_coverage_ratio"] is not None]
        summary["paired_medians"][treatment]["population_coverage_ratio"] = (
            sorted(coverage)[len(coverage) // 2] if coverage else None
        )
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
    fig, axes = plt.subplots(1, len(CASES), figsize=(18, 6), constrained_layout=True)
    for ax, case in zip(axes, CASES):
        arms = {record["treatment"]: record for record in records if record["case"] == case}
        reference_lines = line_coordinates(arms["no_population"]["reference_path"])
        for index, line in enumerate(reference_lines):
            x, y = transformer.transform([point[0] for point in line], [point[1] for point in line])
            ax.plot(x, y, color="#222222", linewidth=2.5, label="Reference (post-hoc)" if index == 0 else None, zorder=4)
        for treatment, label in TREATMENTS.items():
            for line in line_coordinates(arms[treatment]["planned_path"]):
                x, y = transformer.transform([point[0] for point in line], [point[1] for point in line])
                ax.plot(x, y, color=COLORS[treatment], linewidth=1.4, label=label, alpha=0.9)
        ax.set_title(case)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("TWD97 TM2 121E (m)")
        ax.set_ylabel("TWD97 TM2 121E (m)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Reference geometry (post-hoc) and A* routes under WorldPop treatments")
    fig.savefig(output / "route_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    records = [load_run(treatment, case) for treatment in TREATMENTS for case in CASES]
    verify_pairing(records)
    write_outputs(records, args.out_dir)
    print(f"Validated {len(records)} runs; wrote {args.out_dir}")


if __name__ == "__main__":
    main()
