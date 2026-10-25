"""Summarize a completed sampled-Pareto route run without selecting a route.

Historical reference scores are reported only as post-hoc diagnostics. The
script also measures candidate-to-candidate geometry in a local metric CRS so
the finite sample count is not mistaken for route diversity.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transmission_path_planner.metrics import compute_deviation


def read_line(path: Path) -> list[tuple[float, float]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    features = obj.get("features", [])
    if not features:
        raise ValueError(f"No route feature in {path}")
    geometry = features[0].get("geometry") or {}
    if geometry.get("type") != "LineString":
        raise ValueError(f"Expected LineString route in {path}")
    return [(float(x), float(y)) for x, y in geometry["coordinates"]]


def similarity(record: dict) -> dict:
    return record["route_similarity"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Completed planner output directory")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise SystemExit("Run summary is not complete; refusing to summarize partial results")
    metrics_list = json.loads((run_dir / "real_comparison_metrics.json").read_text(encoding="utf-8"))
    metrics = {record["case"]: record for record in metrics_list}

    rows: list[dict] = []
    pair_rows: list[dict] = []
    candidate_p95_by_case: list[list[float]] = []
    candidate_f1_by_case: list[list[float]] = []
    cases = sorted(
        (path for path in run_dir.iterdir()
         if path.is_dir() and (path / "sampled_pareto_candidates.json").is_file()),
        key=lambda path: path.name,
    )
    for case_dir in cases:
        manifest_path = case_dir / "sampled_pareto_candidates.json"
        freeze_path = case_dir / "sampled_pareto_freeze.json"
        if not manifest_path.is_file() or not freeze_path.is_file():
            raise ValueError(f"Missing candidate/freeze manifest in {case_dir}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        final_route = json.loads((case_dir / "final_route.geojson").read_text(encoding="utf-8"))
        case = final_route["features"][0]["properties"]["case"]
        if case not in metrics:
            raise ValueError(f"No post-hoc metrics for {case}")
        if manifest.get("reference_geometry_used_for_planning") is not False:
            raise ValueError(f"Reference geometry flag is not false for {case}")
        candidates = manifest.get("candidates", [])
        if not candidates or not all(c.get("candidate_frozen_before_reference_read") for c in candidates):
            raise ValueError(f"Candidate freeze validation failed for {case}")

        route_records = {"A*(baseline)": case_dir / "planned_route_aastar.geojson"}
        route_records.update({c["candidate_id"]: case_dir / c["geojson"] for c in candidates})
        route_geometries = {name: read_line(path) for name, path in route_records.items()}
        case_metrics = metrics[case]
        base = similarity(case_metrics["A*(baseline)"])
        baseline_f1 = base["symmetric"]["coverage_f1_within_m"]
        candidate_records = [case_metrics[c["candidate_id"]] for c in candidates]
        cand_p95 = [similarity(item)["symmetric"]["p95_m"] for item in candidate_records]
        cand_f1_100 = [similarity(item)["symmetric"]["coverage_f1_within_m"]["100"] for item in candidate_records]
        candidate_p95_by_case.append([value / 1000.0 for value in cand_p95])
        candidate_f1_by_case.append(cand_f1_100)

        pair_p95 = []
        near_100m_pairs = 0
        names = list(route_geometries)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                result = compute_deviation(route_geometries[left], route_geometries[right])
                if not result:
                    raise ValueError(f"Could not compare {case} routes {left} and {right}")
                sim = similarity(result)
                p95 = float(sim["symmetric"]["p95_m"])
                f1_100 = float(sim["symmetric"]["coverage_f1_within_m"]["100"])
                pair_p95.append(p95)
                if p95 <= 100.0:
                    near_100m_pairs += 1
                pair_rows.append({
                    "case": case,
                    "route_a": left,
                    "route_b": right,
                    "symmetric_p95_m": p95,
                    "f1_100m": f1_100,
                    "near_duplicate_p95_le_100m": p95 <= 100.0,
                })

        hashes = [c.get("post_validation_path_sha256") for c in candidates]
        unique_hash_count = len(set(hashes))
        length = base["length"]
        rows.append({
            "case": case,
            "astar_symmetric_p95_m": base["symmetric"]["p95_m"],
            "astar_symmetric_mean_m": base["symmetric"]["mean_m"],
            "astar_f1_100m": baseline_f1["100"],
            "astar_f1_500m": baseline_f1["500"],
            "astar_f1_1000m": baseline_f1["1000"],
            "astar_length_km": length["planned_km"],
            "reference_length_km": length["reference_km"],
            "astar_abs_length_difference_pct": length["absolute_difference_pct"],
            "candidate_count": len(candidates),
            "candidate_unique_path_hashes": unique_hash_count,
            "candidate_hard_violations_max": max(int(c.get("hard_violations", 0)) for c in candidates),
            "candidate_median_p95_m_posthoc": statistics.median(cand_p95),
            "candidate_median_f1_100m_posthoc": statistics.median(cand_f1_100),
            "candidate_min_p95_m_posthoc_oracle_only": min(cand_p95),
            "candidate_max_f1_100m_posthoc_oracle_only": max(cand_f1_100),
            "route_pair_count_including_astar": len(pair_p95),
            "pairwise_p95_median_m": statistics.median(pair_p95),
            "near_duplicate_pairs_p95_le_100m": near_100m_pairs,
            "pairwise_pairs_total": len(pair_p95),
        })

    csv_path = run_dir / "sampled_pareto_case_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pair_path = run_dir / "sampled_pareto_pairwise_geometry.csv"
    with pair_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    baseline_p95 = [row["astar_symmetric_p95_m"] / 1000 for row in rows]
    baseline_f1 = [row["astar_f1_100m"] * 100 for row in rows]
    labels = [row["case"] for row in rows]
    display_labels = [
        label.replace("天輪~龍潭線", "Tianlun-Longtan")
        .replace("深美～冬山線", "Shenmei-Dongshan")
        .replace("明潭-鳳林線", "Mingtan-Fenglin")
        for label in labels
    ]
    positions = list(range(len(rows)))
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, constrained_layout=True)
    axes[0].boxplot(candidate_p95_by_case, positions=positions, widths=0.48, showfliers=True)
    axes[0].scatter(positions, baseline_p95, marker="D", color="#c62828", label="Production A*", zorder=3)
    axes[0].set_ylabel("Bidirectional P95 distance (km)")
    axes[0].set_title("12-case terrain + source-attributed soft-constraint run")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].boxplot(
        [[100 * value for value in values] for values in candidate_f1_by_case],
        positions=positions, widths=0.48, showfliers=True,
    )
    axes[1].scatter(positions, baseline_f1, marker="D", color="#c62828", label="Production A*", zorder=3)
    axes[1].set_ylabel("F1 within 100 m (%)")
    axes[1].set_xticks(positions, display_labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")
    fig.text(
        0.01, 0.005,
        "Boxes show finite sampled candidates; red diamonds are production A*. "
        "All route-vs-reference scores are post-hoc; no best candidate is selected.",
        fontsize=9,
    )
    figure_path = run_dir / "sampled_pareto_12case_summary.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_out = {
        "schema_version": 1,
        "run_complete": True,
        "case_count": len(rows),
        "candidate_scope": "finite_scalarization_samples_only; experimental and non-selectable",
        "route_reference_scores": "post-hoc diagnostics only; not used for planning or candidate selection",
        "near_duplicate_definition": "candidate pair symmetric route-distance P95 <= 100 m",
        "astar_median_symmetric_p95_m": statistics.median(row["astar_symmetric_p95_m"] for row in rows),
        "astar_mean_symmetric_p95_m": statistics.mean(row["astar_symmetric_p95_m"] for row in rows),
        "astar_median_f1_100m": statistics.median(row["astar_f1_100m"] for row in rows),
        "astar_mean_f1_100m": statistics.mean(row["astar_f1_100m"] for row in rows),
        "candidate_median_case_median_p95_m_posthoc": statistics.median(row["candidate_median_p95_m_posthoc"] for row in rows),
        "candidate_mean_case_median_p95_m_posthoc": statistics.mean(row["candidate_median_p95_m_posthoc"] for row in rows),
        "candidate_median_case_median_f1_100m_posthoc": statistics.median(row["candidate_median_f1_100m_posthoc"] for row in rows),
        "candidate_mean_case_median_f1_100m_posthoc": statistics.mean(row["candidate_median_f1_100m_posthoc"] for row in rows),
        "cases_candidate_median_p95_lower_than_astar": sum(row["candidate_median_p95_m_posthoc"] < row["astar_symmetric_p95_m"] for row in rows),
        "cases_candidate_median_f1_100m_higher_than_astar": sum(row["candidate_median_f1_100m_posthoc"] > row["astar_f1_100m"] for row in rows),
        "candidate_pairs_near_duplicate_p95_le_100m": sum(row["near_duplicate_pairs_p95_le_100m"] for row in rows),
        "candidate_pairs_total": sum(row["pairwise_pairs_total"] for row in rows),
        "candidate_candidate_pairs_near_duplicate_p95_le_100m": sum(
            pair["near_duplicate_p95_le_100m"]
            for pair in pair_rows
            if pair["route_a"] != "A*(baseline)" and pair["route_b"] != "A*(baseline)"
        ),
        "candidate_candidate_pairs_total": sum(
            pair["route_a"] != "A*(baseline)" and pair["route_b"] != "A*(baseline)"
            for pair in pair_rows
        ),
        "astar_candidate_pairs_near_duplicate_p95_le_100m": sum(
            pair["near_duplicate_p95_le_100m"]
            for pair in pair_rows
            if pair["route_a"] == "A*(baseline)" or pair["route_b"] == "A*(baseline)"
        ),
        "astar_candidate_pairs_total": sum(
            pair["route_a"] == "A*(baseline)" or pair["route_b"] == "A*(baseline)"
            for pair in pair_rows
        ),
        "cases_with_fewer_than_5_unique_candidate_hashes": sum(row["candidate_unique_path_hashes"] < 5 for row in rows),
        "hard_violation_count_is_limited_to_configured_masks": True,
        "files": [csv_path.name, pair_path.name, figure_path.name],
    }
    (run_dir / "sampled_pareto_analysis_summary.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary_out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
