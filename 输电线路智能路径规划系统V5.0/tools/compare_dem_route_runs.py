#!/usr/bin/env python3
"""Compare post-hoc route metrics from two completed DEM sensitivity runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run(path: Path):
    summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError(f"Run is incomplete: {path}")
    metrics = json.loads((path / "real_comparison_metrics.json").read_text(encoding="utf-8"))
    result = {}
    for case in metrics:
        route = case.get("A*(baseline)")
        if not route:
            continue
        similarity = route["route_similarity"]
        result[case["case"]] = {
            "p95_m": float(similarity["symmetric"]["p95_m"]),
            "f1_100m": float(similarity["symmetric"]["coverage_f1_within_m"]["100"]),
            "length_error_pct": float(similarity["length"]["absolute_difference_pct"]),
        }
    if not result:
        raise ValueError(f"No A*(baseline) metrics found: {path}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True, help="First completed output directory")
    parser.add_argument("--label-a", default="DEM A")
    parser.add_argument("--run-b", type=Path, required=True, help="Second completed output directory")
    parser.add_argument("--label-b", default="DEM B")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path")
    args = parser.parse_args()

    a, b = load_run(args.run_a), load_run(args.run_b)
    if set(a) != set(b):
        raise ValueError("The runs must contain the same cases")
    cases = list(a)
    rows = [
        {
            "case": case,
            "a": a[case],
            "b": b[case],
            "delta_p95_m_a_minus_b": a[case]["p95_m"] - b[case]["p95_m"],
            "delta_f1_100m_pp_a_minus_b": 100 * (a[case]["f1_100m"] - b[case]["f1_100m"]),
        }
        for case in cases
    ]
    stats = {}
    for label, metrics in ((args.label_a, a), (args.label_b, b)):
        stats[label] = {
            "median_p95_m": statistics.median(x["p95_m"] for x in metrics.values()),
            "median_f1_100m_pct": 100 * statistics.median(x["f1_100m"] for x in metrics.values()),
            "median_length_error_pct": statistics.median(x["length_error_pct"] for x in metrics.values()),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric_scope": "post-hoc only; reference route geometry is not a planning input",
        "case_count": len(cases),
        "labels": [args.label_a, args.label_b],
        "runs": [str(args.run_a), str(args.run_b)],
        "statistics": stats,
        "paired_median_p95_delta_m_a_minus_b": statistics.median(
            row["delta_p95_m_a_minus_b"] for row in rows
        ),
        "paired_median_f1_100m_delta_pp_a_minus_b": statistics.median(
            row["delta_f1_100m_pp_a_minus_b"] for row in rows
        ),
        "paired_median_length_error_delta_pp_a_minus_b": statistics.median(
            a[case]["length_error_pct"] - b[case]["length_error_pct"]
            for case in cases
        ),
        "improved_case_counts": {
            "p95_distance": sum(row["delta_p95_m_a_minus_b"] < 0 for row in rows),
            "f1_100m": sum(row["delta_f1_100m_pp_a_minus_b"] > 0 for row in rows),
            "length_error": sum(
                a[row["case"]]["length_error_pct"] < b[row["case"]]["length_error_pct"]
                for row in rows
            ),
        },
        "rows": rows,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    display_names = {
        "天輪~龍潭線": "Tianlun-Longtan",
        "深美～冬山線": "Shenmei-Dongshan",
        "明潭-鳳林線": "Mingtan-Fenglin",
    }
    labels = [display_names.get(case, case) for case in cases]
    x = np.arange(len(cases))
    width = 0.38
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6.4), constrained_layout=True)
    colors = ("#2676b8", "#e58a2b")
    p95_a = [a[case]["p95_m"] / 1000 for case in cases]
    p95_b = [b[case]["p95_m"] / 1000 for case in cases]
    left.bar(x - width / 2, p95_a, width, label=args.label_a, color=colors[0])
    left.bar(x + width / 2, p95_b, width, label=args.label_b, color=colors[1])
    left.set_ylabel("Symmetric route distance P95 (km)")
    left.set_title("Route shape deviation")
    left.set_xticks(x, labels, rotation=35, ha="right")
    left.grid(axis="y", alpha=0.25)
    left.legend(frameon=False)

    f1_a = [100 * a[case]["f1_100m"] for case in cases]
    f1_b = [100 * b[case]["f1_100m"] for case in cases]
    right.bar(x - width / 2, f1_a, width, label=args.label_a, color=colors[0])
    right.bar(x + width / 2, f1_b, width, label=args.label_b, color=colors[1])
    right.set_ylabel("Bidirectional F1 within 100 m (%)")
    right.set_title("Tight corridor overlap")
    right.set_xticks(x, labels, rotation=35, ha="right")
    right.grid(axis="y", alpha=0.25)
    right.legend(frameon=False)

    fig.suptitle("Matched DEM sensitivity runs — same cases, A* settings and constraints", fontsize=14)
    fig.text(
        0.5, -0.025,
        "Reference lines are scored after route generation. F1 is a geometric overlap metric, not engineering feasibility.",
        ha="center", fontsize=9,
    )
    fig.savefig(args.out, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
