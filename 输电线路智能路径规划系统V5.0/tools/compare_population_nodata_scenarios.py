"""Audit three paired WorldPop NoData policy runs without choosing a policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform


POLICIES = ("maximum", "median", "zero")
TOLERANCES_M = (100, 500, 1000)
PROJECT = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
COLORS = {"maximum": "#c44e52", "median": "#4c72b0", "zero": "#55a868"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _keyed_inputs(provenance):
    keyed = {}
    for item in provenance["planning_inputs"]:
        key = (item.get("role"), item.get("path"))
        if key in keyed:
            raise ValueError(f"duplicate planning input entry: {key}")
        keyed[key] = (item.get("sha256"), item.get("size_bytes"), item.get("status"))
    return keyed


def _runtime_sources(provenance):
    return {
        item["path"]: (item.get("sha256"), item.get("size_bytes"))
        for item in provenance["source"]["runtime_source_files"]
    }


def _case_artifacts(run_dir: Path):
    metrics_rows = read_json(run_dir / "comparison_metrics.json")
    metrics = {row["case"]: row["metrics"]["A*(baseline)"] for row in metrics_rows}
    if len(metrics) != len(metrics_rows):
        raise ValueError(f"duplicate case in comparison_metrics.json: {run_dir}")
    geometries, references = {}, {}
    for case_dir in run_dir.iterdir():
        if not case_dir.is_dir():
            continue
        planned_path = case_dir / "planned_route_aastar.geojson"
        reference_path = case_dir / "real_route.geojson"
        final_path = case_dir / "final_route.geojson"
        if not (planned_path.exists() and reference_path.exists() and final_path.exists()):
            continue
        case = read_json(final_path)["features"][0]["properties"]["case"]
        if case in geometries:
            raise ValueError(f"duplicate planned-route case {case!r} in {run_dir}")
        planned = shape(read_json(planned_path)["features"][0]["geometry"])
        reference = shape(read_json(reference_path)["features"][0]["geometry"])
        geometries[case] = transform(PROJECT.transform, planned)
        references[case] = reference_path
    if set(metrics) != set(geometries):
        raise ValueError(f"metric and planned-route case mismatch in {run_dir}")
    return metrics, geometries, references


def validate_triplet(run_dirs: dict[str, Path], expected_cases: int = 12):
    provenances, summaries, artifacts = {}, {}, {}
    for policy in POLICIES:
        run_dir = run_dirs[policy]
        provenance = read_json(run_dir / "run_provenance.json")
        summary = read_json(run_dir / "run_summary.json")
        config = provenance["configuration"]
        if config.get("population_nodata_policy") != policy:
            raise ValueError(f"{policy} path contains policy {config.get('population_nodata_policy')!r}")
        if not summary.get("complete") or summary.get("completed_cases") != expected_cases:
            raise ValueError(f"{policy} run is incomplete: {summary}")
        if summary.get("expected_cases") != expected_cases or summary.get("failed_cases") or summary.get("no_route_cases"):
            raise ValueError(f"{policy} run has unexpected case counts or failures: {summary}")
        if not config.get("population_raster") or float(config.get("population_weight", 0)) <= 0:
            raise ValueError(f"{policy} run must use the same positive-weight population raster")
        provenances[policy], summaries[policy] = provenance, summary
        artifacts[policy] = _case_artifacts(run_dir)

    anchor = provenances[POLICIES[0]]
    anchor_config = anchor["configuration"]
    anchor_inputs = _keyed_inputs(anchor)
    anchor_runtime = _runtime_sources(anchor)
    anchor_metrics, _, anchor_refs = artifacts[POLICIES[0]]
    cases = list(anchor_metrics)
    if len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} unique cases; found {len(cases)}")

    input_hashes = {
        (item.get("role"), item.get("path")): item.get("sha256")
        for item in anchor["planning_inputs"]
    }
    population_keys = [key for key in anchor_inputs if key[0] == "population_raster"]
    if len(population_keys) != 1:
        raise ValueError("each run must declare exactly one population_raster planning input")
    population_record = anchor_inputs[population_keys[0]]
    population_sha = population_record[0]
    if (population_record[2] != "hashed" or not isinstance(population_sha, str)
            or len(population_sha) != 64 or population_record[1] <= 0):
        raise ValueError("population raster planning input must be present and SHA-256 hashed")

    for policy in POLICIES[1:]:
        provenance = provenances[policy]
        config = provenance["configuration"]
        differences = sorted(
            key for key in (set(anchor_config) | set(config))
            if anchor_config.get(key) != config.get(key)
        )
        if set(differences) - {"out_dir", "population_nodata_policy"}:
            raise ValueError(f"unexpected {policy} configuration differences: {differences}")
        if _keyed_inputs(provenance) != anchor_inputs:
            raise ValueError(f"planning input hashes, sizes, or status differ for {policy}")
        if _runtime_sources(provenance) != anchor_runtime:
            raise ValueError(f"runtime source hashes differ for {policy}")
        metrics, _, references = artifacts[policy]
        if list(metrics) != cases:
            raise ValueError(f"case ordering/pairing differs for {policy}")
        if set(references) != set(anchor_refs):
            raise ValueError(f"reference case set differs for {policy}")
        for case in cases:
            if sha256(references[case]) != sha256(anchor_refs[case]):
                raise ValueError(f"reference geometry differs for {policy}/{case}")

    for policy in POLICIES:
        if _keyed_inputs(provenances[policy]) != anchor_inputs:
            raise ValueError(f"planning inputs differ for {policy}")

    return {
        "cases": cases,
        "provenances": provenances,
        "summaries": summaries,
        "artifacts": artifacts,
        "planning_input_hashes": [
            {"role": role, "path": path, "sha256": digest}
            for (role, path), digest in sorted(input_hashes.items())
        ],
        "runtime_source_hashes": [
            {"path": path, "sha256": value[0], "size_bytes": value[1]}
            for path, value in sorted(anchor_runtime.items())
        ],
    }


def _sample_points(line, spacing_m: float = 100.0):
    points = []
    parts = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
    for part in parts:
        if part.is_empty or part.length == 0:
            continue
        count = max(1, int(np.ceil(part.length / spacing_m)))
        points.extend(part.interpolate(float(i) / count, normalized=True) for i in range(count + 1))
    return points


def pair_geometry_metrics(line_a, line_b):
    points_a, points_b = _sample_points(line_a), _sample_points(line_b)
    distances_ab = np.asarray([point.distance(line_b) for point in points_a], dtype=float)
    distances_ba = np.asarray([point.distance(line_a) for point in points_b], dtype=float)
    distances = np.concatenate((distances_ab, distances_ba))
    result = {
        "mean_m": float(np.mean(distances)),
        "p95_m": float(np.percentile(distances, 95)),
        "sampled_hausdorff_m": float(max(np.max(distances_ab), np.max(distances_ba))),
        "sampling_interval_target_m": 100.0,
        "sampled_points_a": len(points_a),
        "sampled_points_b": len(points_b),
    }
    for tolerance in TOLERANCES_M:
        precision = float(np.mean(distances_ab <= tolerance))
        recall = float(np.mean(distances_ba <= tolerance))
        result[f"corridor_f1_{tolerance}m"] = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
    return result


def _exposure(metric):
    return metric.get("weight_independent_exposure") or {}


def build_rows(audit):
    rows, pair_rows = [], []
    cases = audit["cases"]
    artifacts = audit["artifacts"]
    for case in cases:
        row = {"case": case}
        for policy in POLICIES:
            metric = artifacts[policy][0][case]
            exposure = _exposure(metric)
            population = metric.get("population_data_coverage") or {}
            row.update({
                f"{policy}_length_km": metric.get("length_km"),
                f"{policy}_cum_relief_m": metric.get("cum_relief_m"),
                f"{policy}_cumulative_ascent_m": metric.get("cumulative_ascent_m"),
                f"{policy}_cumulative_descent_m": metric.get("cumulative_descent_m"),
                f"{policy}_max_ground_grade_pct": metric.get("max_sampled_ground_grade_pct"),
                f"{policy}_dem_sample_coverage_ratio": metric.get("dem_sample_coverage_ratio"),
                f"{policy}_hard_violations_configured_only": metric.get("hard_violations"),
                f"{policy}_grade_limit_status": metric.get("grade_limit_status"),
                f"{policy}_population_coverage_ratio_approx_cells": population.get("coverage_ratio"),
                f"{policy}_population_unknown_route_cells_approx": population.get("unknown_route_cell_count"),
                f"{policy}_population_cell_fraction_is_approximate": population.get("cell_fraction_is_approximate"),
                f"{policy}_slope_mean_deg_approx_cells": (exposure.get("surface_slope_deg") or {}).get("mean"),
                f"{policy}_slope_p95_deg_approx_cells": (exposure.get("surface_slope_deg") or {}).get("p95"),
                f"{policy}_roughness_mean_m_approx_cells": (exposure.get("roughness_27_m") or {}).get("mean"),
                f"{policy}_roughness_p95_m_approx_cells": (exposure.get("roughness_27_m") or {}).get("p95"),
                f"{policy}_building_density_mean_mapped_only": exposure.get("mapped_building_density_mean"),
                f"{policy}_residential_cell_fraction_mapped_only": exposure.get("mapped_residential_landuse_cell_fraction"),
                f"{policy}_hazard_coverage_status": json.dumps(exposure.get("hazard_layer_coverage_status"), ensure_ascii=False, sort_keys=True),
                f"{policy}_hazard_cell_fractions_approx": json.dumps(exposure.get("hazard_cell_fractions"), ensure_ascii=False, sort_keys=True),
                f"{policy}_custom_constraint_exposure": json.dumps(metric.get("custom_constraint_route_exposure"), ensure_ascii=False, sort_keys=True),
                f"{policy}_posthoc_reference_p95_m": ((metric.get("route_similarity") or {}).get("symmetric") or {}).get("p95_m"),
                f"{policy}_posthoc_reference_f1_100m": (((metric.get("route_similarity") or {}).get("symmetric") or {}).get("coverage_f1_within_m") or {}).get("100"),
            })
        for policy_a, policy_b in (("maximum", "median"), ("maximum", "zero"), ("median", "zero")):
            geometry = pair_geometry_metrics(
                artifacts[policy_a][1][case], artifacts[policy_b][1][case])
            pair_rows.append({"case": case, "policy_a": policy_a, "policy_b": policy_b, **geometry})
            for key, value in geometry.items():
                row[f"{policy_a}_vs_{policy_b}_{key}"] = value
        rows.append(row)
    return rows, pair_rows


def _write_csv(path: Path, rows):
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_route_map(audit, output: Path):
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False
    cases = audit["cases"]
    columns = min(4, len(cases))
    rows = int(np.ceil(len(cases) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.0 * rows), squeeze=False,
                             constrained_layout=True)
    for axis, case in zip(axes.flat, cases):
        for policy in POLICIES:
            line = audit["artifacts"][policy][1][case]
            parts = list(line.geoms) if line.geom_type == "MultiLineString" else [line]
            for part in parts:
                x, y = part.xy
                axis.plot(np.asarray(x) / 1000, np.asarray(y) / 1000,
                          color=COLORS[policy], linewidth=1.3, label=policy)
        axis.set_title(case, fontsize=9)
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
        axis.set_xlabel("TWD97 / TM2 121 X (km)")
        axis.set_ylabel("Y (km)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", ncol=3)
    fig.suptitle("WorldPop NoData policy routes (geometry stability only)", y=1.02)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def compare(run_dirs: dict[str, Path], output_dir: Path, expected_cases: int = 12):
    audit = validate_triplet(run_dirs, expected_cases)
    rows, pair_rows = build_rows(audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scenario_metrics.csv", rows)
    _write_csv(output_dir / "pairwise_geometry_metrics.csv", pair_rows)
    _save_route_map(audit, output_dir / "policy_routes_overview.png")
    report = {
        "schema_version": 1,
        "scope": "NoData sensitivity and geometry stability; no policy selection",
        "expected_cases": expected_cases,
        "cases": audit["cases"],
        "policies": list(POLICIES),
        "configuration_policy_differences": ["population_nodata_policy", "out_dir"],
        "planning_inputs_identical": True,
        "runtime_sources_identical": True,
        "reference_geometries_identical": True,
        "planning_input_hashes": audit["planning_input_hashes"],
        "runtime_source_hashes": audit["runtime_source_hashes"],
        "runs": {
            policy: {
                "directory": str(run_dirs[policy].resolve()),
                "run_summary_sha256": sha256(run_dirs[policy] / "run_summary.json"),
                "run_provenance_sha256": sha256(run_dirs[policy] / "run_provenance.json"),
            }
            for policy in POLICIES
        },
        "geometry_metrics_method": {
            "crs": "EPSG:3826",
            "sample_spacing_target_m": 100,
            "hausdorff": "maximum of bidirectional 100m-sampled point-to-line distances; approximate",
            "f1": "harmonic mean of sampled line coverage fractions within the stated corridor tolerance",
        },
        "coverage_limitations": (
            "Population, slope, roughness, hazard, and mapped feature coverage are existing route-cell approximations, "
            "not length-weighted unless the source metric explicitly says otherwise."
        ),
        "interpretation_limitations": [
            "Reference-line similarity is posthoc only and is not used to rank or select a NoData policy.",
            "hard_violations only reports configured hard constraints; it does not establish engineering feasibility.",
            "The 12-case set overlaps prior experiments and uses historical endpoints, not verified station terminals.",
            "NoData maximum, median, and zero are scenario assumptions, not observations of unknown population.",
        ],
        "outputs": {
            name: sha256(output_dir / name)
            for name in ("scenario_metrics.csv", "pairwise_geometry_metrics.csv", "policy_routes_overview.png")
        },
    }
    (output_dir / "scenario_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for policy in POLICIES:
        parser.add_argument(f"--{policy}", type=Path, required=True, dest=policy)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=12)
    args = parser.parse_args(argv)
    report = compare({policy: getattr(args, policy) for policy in POLICIES},
                     args.output_dir, args.expected_cases)
    print(json.dumps({"cases": len(report["cases"]), "output_dir": str(args.output_dir.resolve())},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
