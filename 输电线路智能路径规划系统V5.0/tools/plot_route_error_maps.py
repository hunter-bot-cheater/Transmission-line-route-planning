"""Render correctly aligned, bidirectional route-to-reference error maps."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import transform


def load_line(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    lines = []
    for feature in document.get("features", []):
        geometry = shape(feature["geometry"])
        if geometry.geom_type == "LineString":
            lines.append(geometry)
        elif geometry.geom_type == "MultiLineString":
            lines.extend(part for part in geometry.geoms if part.length > 0)
    if not lines:
        raise ValueError(f"No LineString geometry in {path}")
    return lines[0] if len(lines) == 1 else MultiLineString(lines)


def line_parts(line):
    return [line] if line.geom_type == "LineString" else list(line.geoms)


def color_segments(line, other, interval_m: float):
    segments, values = [], []
    for part in line_parts(line):
        distances = np.arange(0.0, part.length, interval_m)
        if not len(distances) or distances[-1] < part.length:
            distances = np.append(distances, part.length)
        if len(distances) < 2:
            continue
        coords = np.asarray([(point.x, point.y)
                             for point in (part.interpolate(float(d)) for d in distances)])
        segments.append(np.stack((coords[:-1], coords[1:]), axis=1))
        values.extend(other.distance(part.interpolate(float((a + b) / 2))) / 1000
                      for a, b in zip(distances[:-1], distances[1:]))
    if not segments:
        return np.empty((0, 2, 2)), np.empty(0)
    return np.concatenate(segments), np.asarray(values)


def draw_colored_route(ax, line, other, cmap, norm, interval_m: float):
    segments, values = color_segments(line, other, interval_m)
    collection = LineCollection(segments / 1000, cmap=cmap, norm=norm, linewidth=2.2, zorder=3)
    collection.set_array(values)
    ax.add_collection(collection)
    return collection


def line_xy(line):
    return [np.asarray(part.coords)[:, :2] / 1000 for part in line_parts(line)]


def draw_line(ax, line, **kwargs):
    for coordinates in line_xy(line):
        ax.plot(coordinates[:, 0], coordinates[:, 1], **kwargs)


def prepare_routes(results_dir: Path, metrics_path: Path | None = None):
    metrics_path = metrics_path or results_dir / "real_comparison_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_by_case = {}
    for row in metrics:
        method = row.get("A*(baseline)") or row.get("A*(weighted)")
        if method is None:
            raise ValueError(f"No A* route metrics for route case {row.get('case')!r}")
        metrics_by_case[row["case"]] = method
    routes = {}
    for folder in results_dir.iterdir():
        final_path = folder / "final_route.geojson"
        planned_path = folder / "planned_route_aastar.geojson"
        reference_path = folder / "real_route.geojson"
        if not folder.is_dir() or not final_path.exists() or not planned_path.exists() or not reference_path.exists():
            continue
        final = json.loads(final_path.read_text(encoding="utf-8"))["features"][0]
        case = final["properties"]["case"]
        if case not in metrics_by_case:
            raise ValueError(f"No metrics row for route case {case!r} in {folder}")
        crs = CRS.from_user_input(final["properties"]["route_similarity"]["comparison_crs"])
        project = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform
        planned = transform(project, load_line(planned_path))
        reference = transform(project, load_line(reference_path))
        routes[case] = (planned, reference, metrics_by_case[case])
    if set(routes) != set(metrics_by_case):
        missing = sorted(set(metrics_by_case) - set(routes))
        raise ValueError(f"Missing route artifacts for cases: {missing}")
    return {case: routes[case] for case in metrics_by_case}


def set_limits(ax, planned, reference):
    planned_xy = np.concatenate(line_xy(planned))
    reference_xy = np.concatenate(line_xy(reference))
    all_xy = np.concatenate((planned_xy, reference_xy))
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    pad = max(xmax - xmin, ymax - ymin) * 0.04
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xlabel("局部东向距离（km）")
    ax.set_ylabel("局部北向距离（km）")


def plot_case(case, planned, reference, metrics, out_path, cmap, norm, interval_m):
    sim = metrics["route_similarity"]
    frechet = sim.get("discrete_frechet_approx")
    frechet_km = (frechet["distance_m"] / 1000.0) if frechet else None
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), constrained_layout=True)
    for ax, colored, other, title in (
        (axes[0], planned, reference, "规划线 → 参考线"),
        (axes[1], reference, planned, "参考线 → 规划线"),
    ):
        draw_colored_route(ax, colored, other, cmap, norm, interval_m)
        draw_line(ax, other, color="#343a40", linestyle="--", linewidth=1.35, zorder=2)
        ax.set_title(title)
        set_limits(ax, planned, reference)
    parts = line_parts(planned)
    start, end = parts[0].coords[0], parts[-1].coords[-1]
    axes[0].scatter([start[0] / 1000], [start[1] / 1000], s=25, color="#16803c", zorder=5)
    axes[0].scatter([end[0] / 1000], [end[1] / 1000], s=25, color="#c0392b", zorder=5)
    axes[1].scatter([start[0] / 1000], [start[1] / 1000], s=25, color="#16803c", zorder=5)
    axes[1].scatter([end[0] / 1000], [end[1] / 1000], s=25, color="#c0392b", zorder=5)
    f1 = sim["symmetric"]["coverage_f1_within_m"]["100"] * 100
    p95 = sim["symmetric"]["p95_m"] / 1000
    frechet_label = (f"顺序 Fréchet@100m {frechet_km:.2f} km"
                     if frechet_km is not None else "顺序 Fréchet 未定义")
    fig.suptitle(f"{case} | 较差方向 P95 {p95:.2f} km | F1@100米 {f1:.1f}%（覆盖诊断）"
                 f" | {frechet_label}")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="彩线到另一条线路的距离（km）")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_overview(routes, out_path, cmap, norm, interval_m):
    cases = list(routes)
    cols = min(4, max(1, len(cases)))
    rows = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.75 * rows),
                             squeeze=False, constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        planned, reference, metrics = routes[case]
        sim = metrics["route_similarity"]
        draw_colored_route(ax, planned, reference, cmap, norm, interval_m)
        draw_line(ax, reference, color="#343a40", linestyle="--", linewidth=1.05, zorder=2)
        f1 = sim["symmetric"]["coverage_f1_within_m"]["100"] * 100
        p95 = sim["symmetric"]["p95_m"] / 1000
        frechet = sim.get("discrete_frechet_approx")
        frechet_label = (f"Fréchet@100m {frechet['distance_m'] / 1000:.2f} km"
                         if frechet else "Fréchet 未定义")
        ax.set_title(f"{case}\nP95 {p95:.1f} km | F1@100m {f1:.1f}%"
                     f"\n{frechet_label}", fontsize=9)
        set_limits(ax, planned, reference)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    legend = [Line2D([0], [0], color="#343a40", linestyle="--", label="参考线")]
    axes.flat[0].legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.85)
    methods = sorted({metrics.get("method", "A*") for _, _, metrics in routes.values()})
    method_label = "/".join(methods)
    fig.suptitle(
        f"配对参考线空间偏差｜{len(cases)} 条线路｜{method_label}",
        fontsize=15,
    )
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="规划线到参考线距离（km）", shrink=0.75)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_chain_error_profile(case, profile, out_path):
    """Plot post-hoc error by equal-distance bins in both route directions."""
    if not profile or profile.get("version") != 1:
        return False
    directions = profile.get("directions", {})
    direction_specs = (
        ("planned_to_reference", "规划线 → 参考线"),
        ("reference_to_planned", "参考线 → 规划线"),
    )
    if any(not isinstance(directions.get(key), list) for key, _ in direction_specs):
        return False

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             constrained_layout=True)
    try:
        for ax, (key, label) in zip(axes, direction_specs):
            bins = directions[key]
            fractions = [
                50.0 * (float(item["start_fraction"]) + float(item["end_fraction"]))
                for item in bins
            ]
            p50_km = [
                float(item["p50_m"]) / 1000 if item.get("p50_m") is not None else np.nan
                for item in bins
            ]
            p95_km = [
                float(item["p95_m"]) / 1000 if item.get("p95_m") is not None else np.nan
                for item in bins
            ]
            coverage_100m_pct = [
                100.0 * float(item["coverage_within_m"]["100"])
                if item.get("coverage_within_m", {}).get("100") is not None else np.nan
                for item in bins
            ]
            ax.plot(fractions, p50_km, marker="o", label="距离中位数 P50")
            ax.plot(fractions, p95_km, marker="s", linestyle="--",
                    label="距离 P95")
            ax.set_ylabel("到另一条线路的距离（km）")
            ax.set_title(label)
            ax.grid(alpha=0.25)

            coverage_ax = ax.twinx()
            coverage_ax.plot(fractions, coverage_100m_pct, color="#238b57",
                             marker=".", linestyle=":", label="单向 100 米内覆盖")
            coverage_ax.set_ylim(0, 100)
            coverage_ax.set_ylabel("100 米内覆盖（%）")

            handles, labels = ax.get_legend_handles_labels()
            coverage_handles, coverage_labels = coverage_ax.get_legend_handles_labels()
            ax.legend(handles + coverage_handles, labels + coverage_labels,
                      loc="upper left", ncol=3, fontsize=8)

        axes[-1].set_xlabel("按较近端点对齐的线路长度位置（%）")
        fig.suptitle(
            f"{case}｜端点对齐分段误差（{profile.get('bin_count', len(directions['planned_to_reference']))} 等分）"
        )
        fig.text(0.5, -0.01, "事后诊断；端点对齐不代表站端身份已核验，也不参与规划选路。",
                 ha="center", fontsize=9)
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
    finally:
        plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--metrics-file", type=Path, default=None,
                        help="指标 JSON，默认读取 results-dir/real_comparison_metrics.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="图件目录，默认写入 results-dir/error_maps_100m")
    parser.add_argument("--sample-interval-m", type=float, default=20.0,
                        help="沿线颜色采样间距（米），默认 20 米")
    args = parser.parse_args()
    if not math.isfinite(args.sample_interval_m) or args.sample_interval_m <= 0:
        parser.error("--sample-interval-m must be a positive finite number")
    results_dir = args.results_dir.resolve()
    out_dir = (args.output_dir or results_dir / "error_maps_100m").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    metrics_file = args.metrics_file.resolve() if args.metrics_file else None
    routes = prepare_routes(results_dir, metrics_file)
    max_distance_km = max(
        m["route_similarity"]["symmetric"]["sampled_hausdorff_m"] / 1000
        for _, _, m in routes.values()
    )
    norm = Normalize(0, np.ceil(max_distance_km))
    cmap = plt.get_cmap("turbo")
    plot_overview(routes, out_dir / "12_route_error_overview.png", cmap, norm, args.sample_interval_m)

    for case, (planned, reference, metrics) in routes.items():
        safe_name = re.sub(r"[^\w.-]+", "_", case, flags=re.UNICODE).strip("_.")
        plot_case(case, planned, reference, metrics, out_dir / f"{safe_name}_error_map.png", cmap, norm, args.sample_interval_m)
        plot_chain_error_profile(
            case, metrics.get("route_similarity", {}).get("chain_error_profile"),
            out_dir / f"{safe_name}_chain_error_profile.png",
        )

    print(f"Rendered {len(routes)} route cases to {out_dir}")


if __name__ == "__main__":
    main()
