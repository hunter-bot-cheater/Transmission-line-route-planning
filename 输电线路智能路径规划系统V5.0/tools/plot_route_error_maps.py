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
from shapely.geometry import shape
from shapely.ops import transform


def load_line(path: Path):
    feature = json.loads(path.read_text(encoding="utf-8"))["features"][0]
    return shape(feature["geometry"])


def color_segments(line, other, interval_m: float):
    distances = np.arange(0.0, line.length, interval_m)
    if not len(distances) or distances[-1] < line.length:
        distances = np.append(distances, line.length)
    points = [line.interpolate(float(d)) for d in distances]
    coords = np.asarray([(p.x, p.y) for p in points])
    segments = np.stack((coords[:-1], coords[1:]), axis=1)
    values = [other.distance(line.interpolate(float((a + b) / 2))) / 1000 for a, b in zip(distances[:-1], distances[1:])]
    return segments, np.asarray(values)


def draw_colored_route(ax, line, other, cmap, norm, interval_m: float):
    segments, values = color_segments(line, other, interval_m)
    collection = LineCollection(segments / 1000, cmap=cmap, norm=norm, linewidth=2.2, zorder=3)
    collection.set_array(values)
    ax.add_collection(collection)
    return collection


def line_xy(line):
    return np.asarray(line.coords)[:, :2] / 1000


def prepare_routes(results_dir: Path):
    metrics = json.loads((results_dir / "real_comparison_metrics.json").read_text(encoding="utf-8"))
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
    px, py = line_xy(planned).T
    rx, ry = line_xy(reference).T
    xmin, xmax = min(px.min(), rx.min()), max(px.max(), rx.max())
    ymin, ymax = min(py.min(), ry.min()), max(py.max(), ry.max())
    pad = max(xmax - xmin, ymax - ymin) * 0.04
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_xlabel("局部东向距离（km）")
    ax.set_ylabel("局部北向距离（km）")


def plot_case(case, planned, reference, metrics, out_path, cmap, norm, interval_m):
    sim = metrics["route_similarity"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), constrained_layout=True)
    for ax, colored, other, title in (
        (axes[0], planned, reference, "规划线 → 参考线"),
        (axes[1], reference, planned, "参考线 → 规划线"),
    ):
        draw_colored_route(ax, colored, other, cmap, norm, interval_m)
        ax.plot(*line_xy(other).T, color="#343a40", linestyle="--", linewidth=1.35, zorder=2)
        ax.set_title(title)
        set_limits(ax, planned, reference)
    start, end = planned.coords[0], planned.coords[-1]
    axes[0].scatter([start[0] / 1000], [start[1] / 1000], s=25, color="#16803c", zorder=5)
    axes[0].scatter([end[0] / 1000], [end[1] / 1000], s=25, color="#c0392b", zorder=5)
    axes[1].scatter([start[0] / 1000], [start[1] / 1000], s=25, color="#16803c", zorder=5)
    axes[1].scatter([end[0] / 1000], [end[1] / 1000], s=25, color="#c0392b", zorder=5)
    f1 = sim["symmetric"]["coverage_f1_within_m"]["100"] * 100
    p95 = sim["symmetric"]["p95_m"] / 1000
    fig.suptitle(f"{case} | 双向 F1@100米 {f1:.1f}% | 较差方向 P95 {p95:.2f} km")
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
        ax.plot(*line_xy(reference).T, color="#343a40", linestyle="--", linewidth=1.05, zorder=2)
        f1 = sim["symmetric"]["coverage_f1_within_m"]["100"] * 100
        p95 = sim["symmetric"]["p95_m"] / 1000
        ax.set_title(f"{case}\nF1@100m {f1:.1f}% | 较差方向 P95 {p95:.1f} km", fontsize=9)
        set_limits(ax, planned, reference)
    for ax in axes.flat[len(cases):]:
        ax.axis("off")
    legend = [Line2D([0], [0], color="#343a40", linestyle="--", label="参考线")]
    axes.flat[0].legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.85)
    methods = sorted({metrics.get("method", "A*") for _, _, metrics in routes.values()})
    method_label = "/".join(methods)
    fig.suptitle(
        f"严格无线路先验评测｜{len(cases)} 条线路｜20 米来源 DEM｜{method_label}",
        fontsize=15,
    )
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="规划线到参考线距离（km）", shrink=0.75)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--sample-interval-m", type=float, default=200.0)
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    out_dir = results_dir / "error_maps_100m"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    routes = prepare_routes(results_dir)
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

    print(f"Rendered {len(routes)} route cases to {out_dir}")


if __name__ == "__main__":
    main()
