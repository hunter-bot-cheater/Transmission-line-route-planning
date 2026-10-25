"""Audit and plot a paired no-population vs population-cost route experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_CONFIG_DIFFERENCES = {
    "out_dir", "population_raster", "population_weight",
    "population_nodata_policy",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_geometries(run_dir: Path):
    result = {}
    for case_dir in run_dir.iterdir():
        planned_path = case_dir / "planned_route_aastar.geojson"
        reference_path = case_dir / "real_route.geojson"
        if not case_dir.is_dir() or not planned_path.exists() or not reference_path.exists():
            continue
        planned_doc = read_json(planned_path)
        reference_doc = read_json(reference_path)
        final_doc = read_json(case_dir / "final_route.geojson")
        feature = planned_doc["features"][0]
        case = final_doc["features"][0]["properties"]["case"]
        result[case] = {
            "planned": shape(feature["geometry"]),
            "reference": shape(reference_doc["features"][0]["geometry"]),
            "planned_path": planned_path,
            "reference_path": reference_path,
        }
    return result


def metrics_by_case(run_dir: Path):
    rows = read_json(run_dir / "comparison_metrics.json")
    return {row["case"]: row["metrics"]["A*(baseline)"] for row in rows}


def configure_font():
    for candidate in (Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")):
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def validate_pair(baseline: Path, treatment: Path):
    base_prov = read_json(baseline / "run_provenance.json")
    treat_prov = read_json(treatment / "run_provenance.json")
    base_summary = read_json(baseline / "run_summary.json")
    treat_summary = read_json(treatment / "run_summary.json")
    for label, summary in (("baseline", base_summary), ("treatment", treat_summary)):
        if not summary.get("complete") or summary.get("completed_cases") != summary.get("expected_cases"):
            raise ValueError(f"{label} run is incomplete: {summary}")
        if summary.get("failed_cases") or summary.get("no_route_cases"):
            raise ValueError(f"{label} run contains failed/no-route cases")

    config_a = base_prov["configuration"]
    config_b = treat_prov["configuration"]
    config_keys = set(config_a) | set(config_b)
    config_differences = sorted(
        key for key in config_keys if config_a.get(key) != config_b.get(key))
    unexpected_config = sorted(set(config_differences) - ALLOWED_CONFIG_DIFFERENCES)
    if unexpected_config:
        raise ValueError(f"unexpected configuration differences: {unexpected_config}")
    if float(config_a.get("population_weight", -1)) != 0.0:
        raise ValueError("baseline population weight must be zero")
    if float(config_b.get("population_weight", 0)) <= 0.0 or not config_b.get("population_raster"):
        raise ValueError("treatment must enable a population raster and positive weight")

    def keyed(items):
        return {(x.get("role"), x.get("path")): x for x in items}

    inputs_a = keyed(base_prov["planning_inputs"])
    inputs_b = keyed(treat_prov["planning_inputs"])
    if not set(inputs_a) <= set(inputs_b):
        raise ValueError("treatment is missing shared planning inputs")
    changed_inputs = [key for key in inputs_a if (
        inputs_a[key].get("sha256") != inputs_b[key].get("sha256")
        or inputs_a[key].get("size_bytes") != inputs_b[key].get("size_bytes")
        or inputs_a[key].get("status") != inputs_b[key].get("status"))]
    if changed_inputs:
        raise ValueError(f"shared planning inputs changed: {changed_inputs}")
    treatment_only = set(inputs_b) - set(inputs_a)
    if {role for role, _ in treatment_only} != {"population_raster"}:
        raise ValueError(f"unexpected treatment-only planning inputs: {treatment_only}")

    def runtime_sources(provenance):
        return {item["path"]: (item.get("sha256"), item.get("size_bytes"))
                for item in provenance["source"]["runtime_source_files"]}

    if runtime_sources(base_prov) != runtime_sources(treat_prov):
        raise ValueError("runtime source file hashes differ between runs")

    geo_a, geo_b = case_geometries(baseline), case_geometries(treatment)
    metrics_a, metrics_b = metrics_by_case(baseline), metrics_by_case(treatment)
    cases = list(metrics_a)
    if cases != list(metrics_b) or set(cases) != set(geo_a) or set(cases) != set(geo_b):
        raise ValueError("route cases or route artifacts do not pair exactly")
    reference_hashes = {}
    geometry_changed = {}
    for case in cases:
        ref_a, ref_b = geo_a[case]["reference_path"], geo_b[case]["reference_path"]
        reference_hashes[case] = {"baseline": sha256(ref_a), "treatment": sha256(ref_b)}
        if reference_hashes[case]["baseline"] != reference_hashes[case]["treatment"]:
            raise ValueError(f"reference geometry differs for {case}")
        geometry_changed[case] = not geo_a[case]["planned"].equals_exact(
            geo_b[case]["planned"], tolerance=1e-10)

    return {
        "baseline_provenance": base_prov,
        "treatment_provenance": treat_prov,
        "baseline_summary": base_summary,
        "treatment_summary": treat_summary,
        "config_differences": config_differences,
        "shared_input_count": len(inputs_a),
        "treatment_only_inputs": [
            {"role": role, "path": path, "sha256": inputs_b[(role, path)].get("sha256")}
            for role, path in sorted(treatment_only)],
        "runtime_sources_equal": True,
        "cases": cases,
        "metrics_baseline": metrics_a,
        "metrics_treatment": metrics_b,
        "geometries_baseline": geo_a,
        "geometries_treatment": geo_b,
        "reference_hashes": reference_hashes,
        "geometry_changed": geometry_changed,
    }


def route_row(case: str, a: dict, b: dict, changed: bool):
    sim_a, sim_b = a["route_similarity"], b["route_similarity"]
    cov = b.get("population_data_coverage") or {}
    exp_a, exp_b = a.get("weight_independent_exposure") or {}, b.get("weight_independent_exposure") or {}
    length_a, length_b = float(a["length_km"]), float(b["length_km"])
    return {
        "case": case,
        "geometry_changed": changed,
        "baseline_length_km": length_a,
        "population_length_km": length_b,
        "length_delta_km": length_b - length_a,
        "length_delta_pct": (length_b - length_a) / length_a * 100 if length_a else None,
        "baseline_p95_m": sim_a["symmetric"]["p95_m"],
        "population_p95_m": sim_b["symmetric"]["p95_m"],
        "p95_delta_m": sim_b["symmetric"]["p95_m"] - sim_a["symmetric"]["p95_m"],
        "baseline_f1_100": sim_a["symmetric"]["coverage_f1_within_m"]["100"],
        "population_f1_100": sim_b["symmetric"]["coverage_f1_within_m"]["100"],
        "f1_100_delta": (sim_b["symmetric"]["coverage_f1_within_m"]["100"]
                          - sim_a["symmetric"]["coverage_f1_within_m"]["100"]),
        "baseline_f1_1000": sim_a["symmetric"]["coverage_f1_within_m"]["1000"],
        "population_f1_1000": sim_b["symmetric"]["coverage_f1_within_m"]["1000"],
        "baseline_max_ground_grade_pct": a.get("max_sampled_ground_grade_pct"),
        "population_max_ground_grade_pct": b.get("max_sampled_ground_grade_pct"),
        "population_coverage_ratio": cov.get("coverage_ratio"),
        "population_unknown_route_cell_count": cov.get("unknown_route_cell_count"),
        "baseline_mean_slope_deg": (exp_a.get("surface_slope_deg") or {}).get("mean"),
        "population_mean_slope_deg": (exp_b.get("surface_slope_deg") or {}).get("mean"),
        "baseline_mean_roughness_27_m": (exp_a.get("roughness_27_m") or {}).get("mean"),
        "population_mean_roughness_27_m": (exp_b.get("roughness_27_m") or {}).get("mean"),
        "baseline_building_density_mean": exp_a.get("mapped_building_density_mean"),
        "population_building_density_mean": exp_b.get("mapped_building_density_mean"),
        "baseline_residential_cell_fraction": exp_a.get("mapped_residential_landuse_cell_fraction"),
        "population_residential_cell_fraction": exp_b.get("mapped_residential_landuse_cell_fraction"),
        "baseline_hazard_cell_fractions": exp_a.get("hazard_cell_fractions"),
        "population_hazard_cell_fractions": exp_b.get("hazard_cell_fractions"),
    }


def save_metric_figure(rows: list[dict], output: Path):
    cases = [row["case"] for row in rows]
    x = np.arange(len(cases))
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    width = 0.38
    for ax, key_a, key_b, title, scale, ylabel in (
        (axes[0, 0], "baseline_p95_m", "population_p95_m", "双向距离 P95", 0.001, "km"),
        (axes[0, 1], "baseline_f1_100", "population_f1_100", "双向覆盖 F1 @ 100 m", 100, "%"),
        (axes[1, 0], "baseline_length_km", "population_length_km", "规划路线长度", 1, "km"),
        (axes[1, 1], "baseline_max_ground_grade_pct", "population_max_ground_grade_pct", "最大采样地面坡度", 1, "%"),
    ):
        vals_a = [row[key_a] for row in rows]
        vals_b = [row[key_b] for row in rows]
        ax.bar(x - width / 2, [v * scale if v is not None else 0 for v in vals_a],
               width, label="关闭人口代价", color="#2878B5")
        ax.bar(x + width / 2, [v * scale if v is not None else 0 for v in vals_b],
               width, label="WorldPop 权重 1.0", color="#E17C05")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, cases, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("12 条固定端点线路：人口软代价配对敏感性（历史线仅作后验评价）", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_route_overlay(cases, geometries_a, geometries_b, metrics_a, metrics_b, output: Path):
    cols = min(4, max(1, len(cases)))
    rows_count = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.7 * cols, 4.4 * rows_count), squeeze=False,
                             constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        metric = metrics_b[case]["route_similarity"]
        transformer = Transformer.from_crs("EPSG:4326", metric["comparison_crs"], always_xy=True)
        project = transformer.transform
        ref = transform(project, geometries_a[case]["reference"])
        base = transform(project, geometries_a[case]["planned"])
        pop = transform(project, geometries_b[case]["planned"])
        for geom in (ref, base, pop):
            parts = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
            for part in parts:
                xs, ys = np.asarray(part.coords).T[:2] / 1000.0
                if geom is ref:
                    ax.plot(xs, ys, color="#222222", linestyle="--", linewidth=1.35, label="参考线路")
                elif geom is base:
                    ax.plot(xs, ys, color="#2878B5", linewidth=1.25, label="关闭人口代价")
                else:
                    ax.plot(xs, ys, color="#E17C05", linewidth=1.25, label="人口权重 1.0")
        p95 = metrics_b[case]["route_similarity"]["symmetric"]["p95_m"] / 1000
        cov = metrics_b[case].get("population_data_coverage", {}).get("coverage_ratio")
        cov_text = "未知" if cov is None else f"{cov * 100:.0f}%"
        ax.set_title(f"{case}\n处理组 P95={p95:.2f} km；人口覆盖={cov_text}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.set_xlabel("局部东向距离 (km)")
        ax.set_ylabel("局部北向距离 (km)")
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    axes.flat[0].legend(loc="best", fontsize=8)
    fig.suptitle("规划路线叠图｜人口层只用于规划，历史线路仅用于事后比较", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def copy_evidence(run_dir: Path, case_geoms: dict, destination: Path, label: str):
    target = destination / label
    target.mkdir(parents=True, exist_ok=True)
    for name in ("run_provenance.json", "run_summary.json", "comparison_metrics.json"):
        shutil.copy2(run_dir / name, target / name)
    for case, files in case_geoms.items():
        case_dir = target / case
        case_dir.mkdir(exist_ok=True)
        shutil.copy2(files["planned_path"], case_dir / "planned_route.geojson")
        if label == "baseline":
            shutil.copy2(files["reference_path"], case_dir / "reference_route.geojson")


def write_manifest(output: Path):
    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "evidence_manifest.json":
            files.append({
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    (output / "evidence_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="population_weight=0 run directory")
    parser.add_argument("--treatment", type=Path, required=True, help="population-weighted run directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    baseline, treatment = args.baseline.resolve(), args.treatment.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_font()
    audit = validate_pair(baseline, treatment)
    cases = audit["cases"]
    rows = [route_row(case, audit["metrics_baseline"][case], audit["metrics_treatment"][case],
                      audit["geometry_changed"][case]) for case in cases]

    p95_a = [row["baseline_p95_m"] for row in rows]
    p95_b = [row["population_p95_m"] for row in rows]
    f1_a = [row["baseline_f1_100"] for row in rows]
    f1_b = [row["population_f1_100"] for row in rows]
    coverage = [row["population_coverage_ratio"] for row in rows if row["population_coverage_ratio"] is not None]
    summary = {
        "case_count": len(cases),
        "geometry_changed_count": sum(audit["geometry_changed"].values()),
        "p95_improved_count": sum(b < a for a, b in zip(p95_a, p95_b)),
        "f1_100_improved_count": sum(b > a for a, b in zip(f1_a, f1_b)),
        "baseline_median_p95_m": float(np.median(p95_a)),
        "treatment_median_p95_m": float(np.median(p95_b)),
        "baseline_median_f1_100": float(np.median(f1_a)),
        "treatment_median_f1_100": float(np.median(f1_b)),
        "treatment_mean_population_coverage_ratio": float(np.mean(coverage)) if coverage else None,
        "population_weight": audit["treatment_provenance"]["configuration"]["population_weight"],
        "population_nodata_policy": audit["treatment_provenance"]["configuration"]["population_nodata_policy"],
    }
    evidence = {
        "schema_version": 1,
        "pairing": {
            "valid": True,
            "baseline_commit": audit["baseline_provenance"]["source"].get("commit"),
            "treatment_commit": audit["treatment_provenance"]["source"].get("commit"),
            "baseline_worktree_dirty": audit["baseline_provenance"]["source"].get("dirty"),
            "treatment_worktree_dirty": audit["treatment_provenance"]["source"].get("dirty"),
            "config_differences": audit["config_differences"],
            "shared_planning_input_count": audit["shared_input_count"],
            "treatment_only_inputs": audit["treatment_only_inputs"],
            "runtime_source_hashes_equal": audit["runtime_sources_equal"],
            "reference_geometry_sha256": audit["reference_hashes"],
        },
        "summary": summary,
        "cases": rows,
        "notes": [
            "Historical route geometry was used only after planning for evaluation.",
            "This is an exploratory sensitivity check, not an independent holdout: three routes overlap an earlier population sensitivity study.",
            "WorldPop population values are modeled raster estimates; coverage gaps were penalized with a fixed maximum proxy, not interpreted as population counts.",
            "Route similarity metrics are diagnostics and were not used to choose the route or population weight.",
        ],
    }
    (output / "population_ab_metrics.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "population_ab_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    save_metric_figure(rows, output / "paired_metric_comparison.png")
    save_route_overlay(cases, audit["geometries_baseline"], audit["geometries_treatment"],
                       audit["metrics_baseline"], audit["metrics_treatment"],
                       output / "12_route_paired_overview.png")
    copy_evidence(baseline, audit["geometries_baseline"], output, "baseline")
    copy_evidence(treatment, audit["geometries_treatment"], output, "treatment")
    write_manifest(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Pairing audit passed; evidence written to {output}")


if __name__ == "__main__":
    main()
