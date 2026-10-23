# transmission_path_planner/visualization.py
"""结果可视化（多算法对比、价值图、指标柱状图）。"""

from .config import *
from .geometry import *
from .data import _hillshade, _fill
from .io_export import *


def _block_mean_for_display(values, factor):
    values = np.asarray(values)
    if factor <= 1:
        return values
    height, width = values.shape
    out_h, out_w = math.ceil(height / factor), math.ceil(width / factor)
    padded = np.pad(
        values,
        ((0, out_h * factor - height), (0, out_w * factor - width)),
        mode="edge",
    )
    return padded.reshape(out_h, factor, out_w, factor).mean(axis=(1, 3))


def _block_max_for_display(mask, factor):
    mask = np.asarray(mask)
    if factor <= 1:
        return mask
    height, width = mask.shape
    out_h, out_w = math.ceil(height / factor), math.ceil(width / factor)
    padded = np.pad(
        mask.astype(np.uint8),
        ((0, out_h * factor - height), (0, out_w * factor - width)),
        mode="constant",
        constant_values=0,
    )
    return padded.reshape(out_h, factor, out_w, factor).max(axis=(1, 3))


def plot_comparison(dem, transform, methods_paths, start_ll, end_ll, out_path, title="多算法路径对比",
                    real_ll=None, hazard_masks=None):
    H, W = dem.shape
    left = transform.c; bottom = transform.f + transform.e * H
    right = transform.c + transform.a * W; top = transform.f
    extent = [left, right, bottom, top]
    fig, ax = plt.subplots(figsize=(12, 9), dpi=110)
    try:
        factor = max(1, math.ceil(max(H, W) / 1200))
        display_dem = _block_mean_for_display(dem, factor)
        hill = _hillshade(display_dem)
        ax.imshow(hill, extent=extent, cmap="gray", origin="upper", aspect="auto")
        z = _fill(display_dem).astype(np.float32, copy=False)
        ax.imshow(z, extent=extent, cmap="terrain", origin="upper", aspect="auto",
                  alpha=0.30, vmin=np.nanpercentile(z, 2), vmax=np.nanpercentile(z, 98))
        hazard_colors = {
            "debris_flow_impact": ("#f39c12", "土石流影响范围"),
            "landslide_potential": ("#9b59b6", "大规模崩塌潜势区"),
            "landslide_impact": ("#e74c3c", "大规模崩塌影响范围"),
        }
        hazard_legend = None
        if hazard_masks:
            from matplotlib.colors import ListedColormap
            from matplotlib.patches import Patch
            handles = []
            for key, (color, label) in hazard_colors.items():
                mask = hazard_masks.get(key)
                if mask is None:
                    continue
                mask = np.asarray(mask)
                if factor > 1:
                    mask = _block_max_for_display(mask, factor)
                if not mask.any():
                    continue
                ax.imshow(np.ma.masked_where(mask == 0, mask),
                          extent=extent, cmap=ListedColormap([color]),
                          origin="upper", aspect="auto", alpha=0.28,
                          interpolation="nearest", zorder=2)
                handles.append(Patch(facecolor=color, alpha=0.28, label=label))
            if handles:
                hazard_legend = ax.legend(handles=handles, loc="lower left", framealpha=0.85,
                                          title="官方灾害面（软风险）")
        colors = {"VIN-Grad": "#1f77ff", "PPO": "#2ca02c", "A*(baseline)": "#ff7f0e",
                  "A*(weighted)": "#9467bd", "A*(turn-aware)": "#17becf"}
        for name, geo in methods_paths.items():
            if not geo:
                continue
            ax.plot([p[0] for p in geo], [p[1] for p in geo],
                    color=colors.get(name, "#555"),
                    linestyle=("--" if "A*" in name else "-"),
                    linewidth=2.2, label=f"{name} ({path_length_km(geo):.1f}km)", zorder=4)
        if real_ll and len(real_ll) >= 1:
            from matplotlib.collections import LineCollection
            polys = real_ll if isinstance(real_ll[0][0], (list, tuple)) else [real_ll]
            segs = [[(p[0], p[1]) for p in poly] for poly in polys if len(poly) >= 2]
            if segs:
                lc = LineCollection(segs, colors="black", linestyles=":", linewidths=2.2, zorder=6)
                ax.add_collection(lc)
                ax.plot([], [], color="black", linestyle=":", linewidth=2.6,
                        label=f"配对参考线 ({real_length_km(real_ll):.0f}km)", zorder=6)
        if start_ll:
            ax.scatter([start_ll[1]], [start_ll[0]], c="green", s=130, marker="o", edgecolors="k", label="起点", zorder=5)
        if end_ll:
            ax.scatter([end_ll[1]], [end_ll[0]], c="red", s=150, marker="*", edgecolors="k", label="终点", zorder=5)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.legend(loc="upper right", framealpha=0.9)
        if hazard_legend:
            ax.add_artist(hazard_legend)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(out_path, dpi=110)
    finally:
        plt.close(fig)
    print(f"  对比图保存: {out_path}")


def plot_value_map(V, transform, geo_vin, start_ll, end_ll, out_path):
    H, W = V.shape
    left = transform.c; bottom = transform.f + transform.e * H
    right = transform.c + transform.a * W; top = transform.f
    extent = [left, right, bottom, top]
    fig, ax = plt.subplots(figsize=(11, 9), dpi=160)
    vmin, vmax = np.nanpercentile(V, 1), np.nanpercentile(V, 99)
    im = ax.imshow(V, extent=extent, cmap="viridis", origin="upper", aspect="auto", vmin=vmin, vmax=vmax)
    if geo_vin:
        ax.plot([p[0] for p in geo_vin], [p[1] for p in geo_vin], color="white", linewidth=1.6, zorder=4)
    if start_ll:
        ax.scatter([start_ll[1]], [start_ll[0]], c="lime", s=120, marker="o", edgecolors="k", zorder=5)
    if end_ll:
        ax.scatter([end_ll[1]], [end_ll[0]], c="red", s=150, marker="*", edgecolors="k", zorder=5)
    ax.set_title("神经值传播概率图 V(s) 与梯度追踪路径")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="datalim")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="值(代价-到此为止)")
    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
    print(f"  值传播概率图保存: {out_path}")


def plot_algo_comparison(all_results, out_path):
    """多算法指标聚合柱状图。"""
    methods = ["VIN-Grad", "PPO", "A*(baseline)", "A*(weighted)", "A*(turn-aware)"]
    metrics_keys = ["length_km", "mean_cost", "sinuosity", "hard_violations", "time_s", "cum_relief_m"]
    agg = {m: {k: [] for k in metrics_keys} for m in methods}
    for rec in all_results:
        for m, met in rec["metrics"].items():
            if m not in agg:   # 跳过最终路线记录等非量化对比条目
                continue
            for k in metrics_keys:
                if k in met and met[k] is not None and not (isinstance(met[k], float) and math.isnan(met[k])):
                    agg[m][k].append(met[k])
    means = {m: {k: (sum(v) / len(v) if v else 0.0) for k, v in agg[m].items()} for m in methods}
    labels = {
        "length_km": "路径长度 (km)", "mean_cost": "平均建设代价",
        "sinuosity": "弯曲度", "hard_violations": "硬约束违规(点)",
        "time_s": "耗时 (s)", "cum_relief_m": "累计高差 (m)"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    axes = axes.ravel()
    colors = {"VIN-Grad": "#1f77ff", "PPO": "#2ca02c", "A*(baseline)": "#ff7f0e",
              "A*(weighted)": "#9467bd", "A*(turn-aware)": "#17becf"}
    for i, k in enumerate(metrics_keys):
        ax = axes[i]
        vals = [means[m][k] for m in methods]
        bars = ax.bar(methods, vals, color=[colors[m] for m in methods])
        ax.set_title(labels[k])
        ax.set_ylabel(labels[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("多算法路径规划量化对比（各走廊均值）", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        fig.savefig(out_path, dpi=140)
    except Exception:
        import time as _time
        try:
            _time.sleep(0.2)
            _alt = out_path.with_name(out_path.stem + "_1" + out_path.suffix)
            fig.savefig(_alt, dpi=140)
            out_path = _alt
        except Exception as _se:
            plt.close(fig)
            print(f"  [WARN] 指标对比图保存失败(已跳过): {_se}")
            return
    plt.close(fig)
    print(f"  指标对比图保存: {out_path}")
