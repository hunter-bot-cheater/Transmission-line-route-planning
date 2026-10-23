from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import LineString, shape
from shapely.ops import transform

from transmission_path_planner.metrics import (
    compute_deviation,
    eval_planned_route_error,
    load_real_path,
)

OUT = ROOT / "outputs" / "grade_penalty_2x2_20261024"
REAL_DIR = Path(r"D:\大创\data\real_cases_tw")
VARIANTS = ("baseline", "edge_grade_zero", "raster_slope_zero",
            "both_slope_terms_zero")
COLORS = {
    "baseline": "#2563eb",
    "edge_grade_zero": "#d97706",
    "raster_slope_zero": "#16a34a",
    "both_slope_terms_zero": "#dc2626",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_frozen_file(category: str, relative_path: str) -> Path:
    if category in ("plan", "experiment_script"):
        return OUT / relative_path
    if category == "endpoint_csv":
        return ROOT / "outputs" / "ablation_tw58_tw173_20261024" / relative_path
    if category in ("dem", "planner_source"):
        return ROOT / relative_path
    if category == "osm_cache":
        return Path(r"D:\大创\V5_final\data\downloaded") / relative_path
    if category == "official_hazard_cache":
        return ROOT / "data" / "official_sources" / "ardswc_115_mvt_z10_20261023" / relative_path
    if category == "candidate_artifact":
        return OUT / relative_path
    if category == "project_script":
        return ROOT / relative_path
    raise ValueError(f"unknown frozen-file category: {category}")


def verify_frozen_files(manifest: dict) -> None:
    for item in manifest["files"]:
        path = resolve_frozen_file(item["category"], item["path"])
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise SystemExit(f"frozen file is missing/changed: {item['category']}:{item['path']}")
        if sha256(path) != item["sha256"]:
            raise SystemExit(f"frozen SHA-256 mismatch: {item['category']}:{item['path']}")


def read_candidate(variant: str, case: str):
    matches = []
    for path in (OUT / variant).glob("*/final_route.geojson"):
        document = json.loads(path.read_text(encoding="utf-8"))
        feature = next((item for item in document.get("features", [])
                        if item.get("properties", {}).get("case") == case), None)
        if feature is not None:
            geometry = shape(feature["geometry"])
            if geometry.geom_type != "LineString":
                raise ValueError(f"expected a LineString for {variant}/{case}")
            coordinates = [(float(x), float(y)) for x, y, *_ in geometry.coords]
            matches.append((path, coordinates, geometry))
    if len(matches) != 1:
        raise ValueError(f"expected one frozen candidate for {variant}/{case}; found {len(matches)}")
    return matches[0]


def main() -> None:
    manifest_path = OUT / "candidate_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("reference_geometry_in_candidate_inputs") is not False:
        raise SystemExit("candidate manifest does not confirm reference isolation")

    # Verify all frozen candidates and non-reference inputs before opening a reference.
    verify_frozen_files(manifest)
    scorer_sha256 = next(
        item["sha256"] for item in manifest["files"]
        if item["category"] == "project_script"
        and item["path"] == "tools/score_grade_penalty_2x2.py")

    candidate_geometry = {}
    candidate_relative_metrics = []
    cases = ("TW173", "TW58")
    for case in cases:
        base_path, base_coordinates, base_geometry = read_candidate("baseline", case)
        candidate_geometry["baseline", case] = base_geometry
        for variant in VARIANTS[1:]:
            route_path, coordinates, geometry = read_candidate(variant, case)
            candidate_geometry[variant, case] = geometry
            dev = compute_deviation(coordinates, [base_coordinates])
            sim = dev["route_similarity"]["symmetric"]
            candidate_relative_metrics.append({
                "case": case,
                "variant": variant,
                "baseline_candidate_path": str(base_path.relative_to(ROOT)),
                "candidate_path": str(route_path.relative_to(ROOT)),
                "symmetric_p95_m_vs_baseline": sim["p95_m"],
                "symmetric_mean_m_vs_baseline": sim["mean_m"],
            })

    # Candidate hashes are frozen and verified; only now read reference geometry.
    reference_hashes = {}
    reference_scores = []
    for case in cases:
        reference_path = REAL_DIR / f"{case}_real_path.shp"
        reference_geometry = load_real_path(
            reference_path, require_single_line=True, raise_errors=True)
        reference_hashes[case] = {
            component.suffix.lower(): sha256(component)
            for component in reference_path.parent.glob(reference_path.stem + ".*")
            if component.is_file()
        }
        for variant in VARIANTS:
            route_path, coordinates, geometry = read_candidate(variant, case)
            deviation = compute_deviation(coordinates, reference_geometry)
            error = eval_planned_route_error(coordinates, deviation)
            symmetric = error["route_similarity"]["symmetric"]
            reference_scores.append({
                "case": case,
                "variant": variant,
                "route_path": str(route_path.relative_to(ROOT)),
                "length_km": error["route_similarity"]["length"]["planned_km"],
                "symmetric_p95_m": symmetric["p95_m"],
                "symmetric_mean_m": symmetric["mean_m"],
                "symmetric_sampled_hausdorff_m": symmetric["sampled_hausdorff_m"],
                "f1_100m": symmetric["coverage_f1_within_m"]["100"],
                "f1_1000m": symmetric["coverage_f1_within_m"]["1000"],
                "absolute_length_difference_pct": error["route_similarity"]["length"]["absolute_difference_pct"],
            })

    report = {
        "candidate_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "scoring_script_path": "tools/score_grade_penalty_2x2.py",
        "scoring_script_sha256": scorer_sha256,
        "reference_geometry_loaded_after_candidate_hash_verification": True,
        "reference_file_hashes": reference_hashes,
        "candidate_to_baseline_diagnostics": candidate_relative_metrics,
        "posthoc_reference_scores": reference_scores,
        "selection_or_weight_update_performed": False,
    }
    project = Transformer.from_crs("EPSG:4326", "EPSG:3826", always_xy=True).transform
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
    for ax, case in zip(axes, cases):
        reference_coordinates = load_real_path(
            REAL_DIR / f"{case}_real_path.shp",
            require_single_line=True, raise_errors=True)[0]
        ref_geo = LineString(reference_coordinates)
        ref_projected = transform(project, ref_geo)
        x, y = ref_projected.xy
        ax.plot([value / 1000 for value in x], [value / 1000 for value in y],
                color="#222222", linestyle="--", linewidth=2.0, label="参考线")
        rows = [item for item in reference_scores if item["case"] == case]
        for variant in VARIANTS:
            geometry = transform(project, candidate_geometry[variant, case])
            x, y = geometry.xy
            score = next(item for item in rows if item["variant"] == variant)
            ax.plot([value / 1000 for value in x], [value / 1000 for value in y],
                    color=COLORS[variant], linewidth=1.5,
                    label=f"{variant} | P95 {score['symmetric_p95_m']/1000:.2f} km")
        ax.set_title(case)
        ax.set_xlabel("EPSG:3826 东坐标（km）")
        ax.set_ylabel("EPSG:3826 北坐标（km）")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("坡度双因子候选路线｜参考线仅在候选冻结后用于评分")
    figure_buffer = io.BytesIO()
    fig.savefig(figure_buffer, format="png", dpi=180)
    plt.close(fig)

    # Publish results only after confirming that all frozen inputs stayed intact.
    verify_frozen_files(manifest)
    report_path = OUT / "posthoc_scores.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    (OUT / "posthoc_route_overlay.png").write_bytes(figure_buffer.getvalue())
    print(f"scored {len(reference_scores)} frozen routes; wrote {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
