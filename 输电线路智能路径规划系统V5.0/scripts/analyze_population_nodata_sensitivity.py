"""Rebuild the frozen three-treatment WorldPop NoData comparison artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "no-pop": "pareto_nopop_120c_3cases_35m_20261024",
    "maximum": "pareto_population_w1_max_120c_3cases_35m_20261024",
    "median": "pareto_population_w1_median_demvalid_3cases_35m_20261024",
}
FOLDERS = {
    "TW173": "TW173-06a5036e04",
    "TW58": "TW58-cc00e6e4c2",
    "天輪~龍潭線": "route-b64eac92e5",
}
COLORS = {"no-pop": "#2878B5", "maximum": "#E17C05", "median": "#3A923A"}
LABELS = {"no-pop": "关闭人口代价", "maximum": "NoData按最大值填补", "median": "NoData按中位数填补"}
EXPECTED_COMMIT = "120c88f4ddd7498d2597356619d2a58293a2cd93"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geojson_lines(path: Path):
    payload = read_json(path)

    def walk(value):
        if not value:
            return
        if isinstance(value[0], (int, float)):
            yield value
        else:
            for child in value:
                yield from walk(child)

    lines = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        if geometry.get("type") == "LineString":
            lines.append(list(walk(geometry.get("coordinates", []))))
        elif geometry.get("type") == "MultiLineString":
            lines.extend(list(walk(line)) for line in geometry.get("coordinates", []))
    return [line for line in lines if len(line) > 1]


def configure_font():
    candidates = [Path("C:/Windows/Fonts/msyh.ttc"), Path("C:/Windows/Fonts/simhei.ttf")]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def check_pairing(provenance):
    baseline = provenance["no-pop"]
    for treatment, manifest in provenance.items():
        source = manifest["source"]
        assert source["commit"] == EXPECTED_COMMIT, (treatment, source["commit"])
        assert source["dirty"] is False, treatment
        summary_path = ROOT / "outputs" / RUNS[treatment] / "results" / "run_summary.json"
        summary = read_json(summary_path)
        assert summary["complete"] and summary["completed_cases"] == 3, treatment
        assert not summary["failed_cases"] and not summary["no_route_cases"], treatment

        a = {(x.get("role"), x.get("path")): (x.get("status"), x.get("sha256"), x.get("size_bytes"))
             for x in baseline["planning_inputs"]}
        b = {(x.get("role"), x.get("path")): (x.get("status"), x.get("sha256"), x.get("size_bytes"))
             for x in manifest["planning_inputs"]}
        assert all(a[k] == b[k] for k in a if k in b), treatment
        assert {k for k in a if k not in b} == set(), treatment
        extras = {k[0] for k in b if k not in a}
        assert extras == ({"population_raster"} if treatment != "no-pop" else set()), (treatment, extras)

    keys = set(baseline["configuration"]) | set(provenance["maximum"]["configuration"])
    allowed = {"out_dir", "population_raster", "population_weight", "population_nodata_policy"}
    for treatment, manifest in provenance.items():
        cfg = manifest["configuration"]
        different = {key for key in keys if baseline["configuration"].get(key) != cfg.get(key)}
        assert different <= allowed, (treatment, different)

    pop_hashes = {}
    for treatment in ("maximum", "median"):
        item = next(x for x in provenance[treatment]["planning_inputs"] if x.get("role") == "population_raster")
        pop_hashes[treatment] = item.get("sha256")
    assert pop_hashes["maximum"] and pop_hashes["maximum"] == pop_hashes["median"], pop_hashes


def build_artifacts(real_dir: Path | None):
    configure_font()
    provenance, summaries, metric_rows = {}, {}, {}
    for treatment, run in RUNS.items():
        result_dir = ROOT / "outputs" / run / "results"
        provenance[treatment] = read_json(result_dir / "run_provenance.json")
        summaries[treatment] = read_json(result_dir / "comparison_metrics.json")
        metric_rows[treatment] = {row["case"]: row["metrics"]["A*(baseline)"] for row in summaries[treatment]}
    check_pairing(provenance)

    cases = list(metric_rows["no-pop"])
    assert cases == list(metric_rows["maximum"]) == list(metric_rows["median"])
    output = ROOT / "docs" / "figures" / "population_nodata_sensitivity_20261025"
    output.mkdir(parents=True, exist_ok=True)

    reference_hashes = {}
    overlay_transform = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    for ax, case in zip(axes, cases):
        folder = FOLDERS[case]
        base_paths = {}
        for treatment, run in RUNS.items():
            case_dir = ROOT / "outputs" / run / "results" / folder
            reference = case_dir / "real_route.geojson"
            planned = case_dir / "planned_route_aastar.geojson"
            assert reference.exists() and planned.exists(), (treatment, case)
            reference_hashes.setdefault(case, {})[treatment] = sha256(reference)
            assert len(geojson_lines(reference)) > 0 and len(geojson_lines(planned)) > 0
            base_paths[treatment] = (reference, planned)
        assert len(set(reference_hashes[case].values())) == 1, case

        reference = base_paths["no-pop"][0]
        for line in geojson_lines(reference):
            xs, ys = overlay_transform.transform([p[0] for p in line], [p[1] for p in line])
            ax.plot(xs, ys, color="#202020", linewidth=2.5, label="真实线路" if ax is axes[0] else None, zorder=4)
        for treatment in RUNS:
            for line in geojson_lines(base_paths[treatment][1]):
                xs, ys = overlay_transform.transform([p[0] for p in line], [p[1] for p in line])
                ax.plot(xs, ys, color=COLORS[treatment], linewidth=1.35, alpha=0.9,
                        label=LABELS[treatment] if ax is axes[0] else None)
        ax.set_title(case)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("TWD97 TM2 121E (m)")
        ax.set_ylabel("TWD97 TM2 121E (m)")
        ax.grid(alpha=0.2)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("35 m DEM：真实线与三种人口 NoData 策略下的 A* 路线", fontsize=14)
    fig.savefig(output / "route_overlay_comparison.png", dpi=180)
    plt.close(fig)

    metrics = {}
    for case in cases:
        metrics[case] = {}
        for treatment in RUNS:
            m = metric_rows[treatment][case]
            similarity = m["route_similarity"]["symmetric"]
            coverage = m.get("population_data_coverage", {})
            metrics[case][treatment] = {
                "length_km": m["length_km"],
                "p95_m": similarity["p95_m"],
                "f1_100": similarity["coverage_f1_within_m"]["100"],
                "f1_500": similarity["coverage_f1_within_m"]["500"],
                "f1_1000": similarity["coverage_f1_within_m"]["1000"],
                "max_sampled_ground_grade_pct": m.get("max_sampled_ground_grade_pct"),
                "population_coverage_ratio": coverage.get("coverage_ratio"),
                "population_route_cell_count": coverage.get("route_cell_count"),
                "population_unknown_route_cell_count": coverage.get("unknown_route_cell_count"),
                "population_coverage_status": coverage.get("status"),
                "hard_violations": m.get("hard_violations"),
            }

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    panels = [
        ("p95_m", "双向路线距离 P95 (km)", 0.001),
        ("f1_100", "100 m 覆盖 F1 (%)", 100),
        ("f1_1000", "1 km 覆盖 F1 (%)", 100),
        ("population_coverage_ratio", "WorldPop 有效像元覆盖率 (%)", 100),
        ("max_sampled_ground_grade_pct", "路线最大采样地面坡度 (%)", 1),
    ]
    for ax, (field, title, scale) in zip(axes.flat, panels):
        x = list(range(len(cases)))
        width = 0.24
        for index, treatment in enumerate(RUNS):
            vals = [metrics[case][treatment][field] for case in cases]
            defined = [(position, value) for position, value in enumerate(vals) if value is not None]
            ax.bar([position + (index - 1) * width for position, _ in defined],
                   [value * scale for _, value in defined], width,
                   color=COLORS[treatment], label=LABELS[treatment])
        ax.set_title(title)
        ax.set_xticks(x, cases, rotation=12)
        ax.grid(axis="y", alpha=0.2)
        if field == "population_coverage_ratio":
            ax.text(0.02, 0.97, "关闭人口代价：不适用", transform=ax.transAxes, va="top", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    axes[1, 2].axis("off")
    macro = {}
    for treatment in RUNS:
        macro[treatment] = {
            field: sum(metrics[c][treatment][field] for c in cases) / len(cases)
            for field in ("p95_m", "f1_100", "f1_500", "f1_1000", "max_sampled_ground_grade_pct")
        }
        ratios = [metrics[c][treatment]["population_coverage_ratio"] for c in cases]
        defined = [v for v in ratios if v is not None]
        macro[treatment]["population_coverage_ratio"] = sum(defined) / len(defined) if defined else None
    axes[1, 2].text(0.03, 0.96, "三线路简单平均（仅描述）\n\n" + "\n\n".join(
        f"{LABELS[t]}\nP95 {macro[t]['p95_m']/1000:.2f} km\nF1@100 m {macro[t]['f1_100']*100:.2f}%\nF1@1 km {macro[t]['f1_1000']*100:.2f}%"
        for t in RUNS), va="top", fontsize=10)
    fig.suptitle("路线误差与数据覆盖敏感性；正式值取 A* baseline", fontsize=14)
    fig.savefig(output / "metric_comparison.png", dpi=180)
    plt.close(fig)

    freeze_check = {}
    for treatment, run in RUNS.items():
        freeze_check[treatment] = {}
        for case in cases:
            freeze = read_json(ROOT / "outputs" / run / "results" / FOLDERS[case] / "sampled_pareto_freeze.json")
            assert freeze["reference_geometry_used_for_planning"] is False
            assert freeze["frozen_before_reference_read"] is True
            assert all(c["post_validation_status"] == "pass" for c in freeze["candidates"])
            freeze_check[treatment][case] = {
                "reference_geometry_used_for_planning": freeze["reference_geometry_used_for_planning"],
                "frozen_before_reference_read": freeze["frozen_before_reference_read"],
                "candidate_count": len(freeze["candidates"]),
                "all_post_validation_pass": True,
            }

    references = {}
    if real_dir:
        source_names = {
            "TW173": "TW173_real_path",
            "TW58": "TW58_real_path",
            "天輪~龍潭線": "天輪_龍潭線_real_path",
        }
        for case, stem in source_names.items():
            references[case] = {}
            for suffix in ("shp", "shx", "dbf", "prj"):
                source_file = real_dir / f"{stem}.{suffix}"
                references[case][source_file.name] = sha256(source_file)

    data = {
        "schema_version": 1,
        "source_commit": EXPECTED_COMMIT,
        "source_worktree_clean": True,
        "treatments": RUNS,
        "case_order": cases,
        "metrics": metrics,
        "macro_arithmetic_means_equal_case_weight": macro,
        "reference_output_geojson_sha256_equal_across_treatments": reference_hashes,
        "reference_source_sidecar_sha256": references,
        "sampled_pareto_freeze_checks": freeze_check,
        "provenance_summaries": {
            treatment: {
                "started_at_utc": provenance[treatment].get("started_at_utc"),
                "completed_at_utc": provenance[treatment].get("completed_at_utc"),
                "planning_inputs": provenance[treatment].get("planning_inputs"),
                "configuration": provenance[treatment].get("configuration"),
            }
            for treatment in RUNS
        },
        "interpretation": {
            "population_coverage_denominator": "route grid cells; approximate, not length weighted",
            "pareto_candidates": "diagnostic after-search comparison; not eligible for final-route selection",
            "endpoints": "historic line termini from the case CSV; evaluation is conditional on line-derived endpoints",
        },
    }
    (output / "paired_metrics_and_provenance.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": cases, "macro": macro, "reference_hashes_equal": True}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dir", type=Path, default=None, help="Optional source shapefile directory for manual SHA-256 capture")
    args = parser.parse_args()
    build_artifacts(args.real_dir)


if __name__ == "__main__":
    main()
