"""Summarize a route-planner algorithm comparison without selecting by reference fit."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = (
    "A*(baseline)",
    "A*(8-neighbor exact experimental)",
    "A*(16-neighbor experimental)",
    "Theta*(experimental)",
)
COLORS = ("#f97316", "#8b5a4c", "#858585", "#d85fbd")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _route_similarity(metrics):
    return metrics.get("route_similarity") if isinstance(metrics, dict) else None


def _evaluation_reference_records(names, real_dir):
    if not real_dir:
        return []
    root = Path(real_dir)
    records = []
    for name in names:
        safe_name = "".join(ch if (ch.isalnum() or ch in "_-") else "_"
                            for ch in name)
        filename = f"{safe_name}_real_path.shp"
        direct = root / filename
        matches = [direct] if direct.is_file() else sorted(root.rglob(filename))
        if len(matches) != 1:
            records.append({"case": name, "status": "missing_or_ambiguous",
                            "matching_files": [str(path.resolve()) for path in matches]})
            continue
        shapefile = matches[0]
        for sidecar in sorted(shapefile.parent.glob(f"{shapefile.stem}.*")):
            if sidecar.is_file():
                digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
                records.append({"case": name, "status": "hashed",
                                "path": str(sidecar.resolve()),
                                "size_bytes": sidecar.stat().st_size,
                                "sha256": digest})
    return records


def summarize(run_dir, output_dir):
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = read_json(run_dir / "comparison_metrics.json")
    run_summary = read_json(run_dir / "run_summary.json")
    provenance_path = run_dir / "run_provenance.json"
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    names = [row["case"] for row in cases]
    failed = {row["case"]: row.get("error", "")
              for row in run_summary.get("failed_cases", [])}
    all_names = names + [name for name in failed if name not in names]

    measured = {method: {} for method in METHODS}
    csv_rows = []
    for name in all_names:
        case = next((row for row in cases if row["case"] == name), None)
        for method in METHODS:
            metrics = (case or {}).get("metrics", {}).get(method, {})
            similarity = _route_similarity(metrics)
            if similarity:
                status = "measured"
                measured[method][name] = {
                    **similarity,
                    "route_generation_time_s": metrics.get("time_s"),
                }
            elif metrics.get("hard_constraint_status") == "fail":
                status = "rejected_hard_constraint"
            elif name in failed:
                status = "planning_failed"
            else:
                status = "not_measured"
            route_length = metrics.get("length_km")
            symmetric = (similarity or {}).get("symmetric", {})
            length = (similarity or {}).get("length", {})
            csv_rows.append({
                "case": name,
                "method": method,
                "status": status,
                "length_km": route_length,
                "route_generation_time_s": metrics.get("time_s"),
                "hard_violations": metrics.get("hard_violations"),
                "symmetric_p95_m": symmetric.get("p95_m"),
                "sampled_hausdorff_m": symmetric.get("sampled_hausdorff_m"),
                "f1_100m": symmetric.get("coverage_f1_within_m", {}).get("100"),
                "f1_1km": symmetric.get("coverage_f1_within_m", {}).get("1000"),
                "length_absolute_difference_pct": length.get("absolute_difference_pct"),
                "route_similarity_version": (case or {}).get("metrics", {}).get(
                    method, {}).get("route_similarity_version"),
            })

    with (output_dir / "route_algorithm_metrics.csv").open(
            "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    common_cases = [name for name in names
                    if all(name in measured[method] for method in METHODS)]
    metrics_summary = {}
    for method in METHODS:
        vals = []
        for name in common_cases:
            route = measured[method][name]
            symmetric = route["symmetric"]
            length = route["length"]
            vals.append({
                "case": name,
                "route_generation_time_s": route.get("route_generation_time_s"),
                "symmetric_p95_km": symmetric["p95_m"] / 1000.0,
                "sampled_hausdorff_km": symmetric["sampled_hausdorff_m"] / 1000.0,
                "f1_100m_pct": symmetric["coverage_f1_within_m"]["100"] * 100.0,
                "f1_1km_pct": symmetric["coverage_f1_within_m"]["1000"] * 100.0,
                "length_absolute_difference_pct": length["absolute_difference_pct"],
            })
        metrics_summary[method] = {
            "measured_cases": len(measured[method]),
            "common_sample_medians": {
                key: statistics.median(row[key] for row in vals) if vals else None
                for key in (
                    "route_generation_time_s", "symmetric_p95_km",
                    "sampled_hausdorff_km", "f1_100m_pct",
                    "f1_1km_pct", "length_absolute_difference_pct")
            },
            "common_sample_by_case": vals,
        }

    projection_consistency = {}
    for case in cases:
        projections = {
            similarity["comparison_crs"]
            for metrics in case.get("metrics", {}).values()
            if (similarity := _route_similarity(metrics)) is not None
        }
        projection_consistency[case["case"]] = len(projections) == 1

    summary = {
        "expected_cases": run_summary.get("expected_cases"),
        "recorded_cases": run_summary.get("recorded_cases"),
        "complete": run_summary.get("complete"),
        "failed_cases": run_summary.get("failed_cases", []),
        "common_case_count_all_methods": len(common_cases),
        "common_cases_all_methods": common_cases,
        "method_results": metrics_summary,
        "same_comparison_crs_per_case": projection_consistency,
        "provenance_source_commit": provenance.get("source", {}).get("commit"),
        "provenance_source_dirty": provenance.get("source", {}).get("dirty"),
        "evaluation_only_reference_files": _evaluation_reference_records(
            names, provenance.get("configuration", {}).get("real_dir")),
        "run_directory": str(run_dir.resolve()),
    }
    (output_dir / "route_algorithm_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not all(projection_consistency.values()):
        raise ValueError("At least one case used inconsistent comparison projections")
    _plot(common_cases, measured, output_dir / "route_algorithm_comparison.png")
    return summary


def _plot(cases, measured, destination):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for method, color in zip(METHODS, COLORS):
        p95 = [measured[method][case]["symmetric"]["p95_m"] / 1000.0
               for case in cases]
        f1 = [100.0 * measured[method][case]["symmetric"][
            "coverage_f1_within_m"]["100"] for case in cases]
        axes[0].plot(cases, p95, marker="o", linewidth=1.8, color=color,
                     label=f"{method} (N={len(p95)})")
        axes[1].plot(cases, f1, marker="o", linewidth=1.8, color=color,
                     label=f"{method} (N={len(f1)})")
    axes[0].set_title("同一验证案例上的双向空间误差与100米覆盖率")
    axes[0].set_ylabel("双向较差方向 P95 (km, 越低越好)")
    axes[1].set_ylabel("F1@100m (%) (越高越好)")
    axes[1].set_xlabel("历史线路样本（仅用于规划后评估）")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.tick_params(axis="x", rotation=18)
        axis.legend(ncol=2, fontsize=8)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Planner output directory")
    parser.add_argument("output_dir", type=Path, help="Summary artifact directory")
    args = parser.parse_args()
    summarize(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
