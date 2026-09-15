"""
v3_dl: 输电线路智能路径规划系统 — 深度学习端到端路径规划主入口
版本: v3.20260525
作者: path_planning_team
变更记录:
  - v3.20260525: CNN成本模型+神经路径规划(无A*), 多省份支持
  - v2.20260525: 随机森林+A*+7项质量门控
  - v1.20260525: 初始版本
依赖: v3/config, v3/src/*, shared/data_acquisition

用法:
  python main.py                          # 默认: 台湾, 核三厂→台北
  python main.py --region sichuan         # 切换至四川
  python main.py --start 22.0,120.5 --end 25.0,121.5    # 自定义起止点
  python main.py --synthetic              # 使用合成数据(无GIS文件时)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import rasterio
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import config as cfg
from src.data_acquisition import acquire_all, switch_region
from src.preprocessing import (
    derive_terrain_factors, generate_hard_mask, generate_soft_mask,
    align_all_rasters, normalize_features,
)
from src.cost_model import (
    build_feature_stack, generate_pseudo_labels,
    train_cost_unet, predict_cost_surface, save_cost_geotiff,
)
from src.path_planning import (
    neural_path_planning, geo_to_grid, grid_to_geo,
    compute_path_length_km, quality_gate,
)
from src.output import (
    export_path, compute_statistics, plot_single_path_overview,
    export_quality_report,
)


def parse_args():
    parser = argparse.ArgumentParser(description="v3 DL输电线路路径规划")
    parser.add_argument("--region", type=str, default="taiwan", help="分析区域")
    parser.add_argument("--start", type=str, default=None, help="起点 lat,lon")
    parser.add_argument("--end", type=str, default=None, help="终点 lat,lon")
    parser.add_argument("--synthetic", action="store_true", help="使用合成数据")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练(使用已保存模型)")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    return parser.parse_args()


def main():
    args = parse_args()
    t_total = time.time()

    print("=" * 60)
    print("  v3 输电线路智能路径规划系统 (深度学习端到端)")
    print(f"  版本: v3.20260525 | 设备: {cfg.DEVICE}")
    print("=" * 60)

    # 区域切换
    if args.region != "taiwan":
        switch_region(args.region)

    # 输出目录
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = cfg.OUTPUT_DIR / f"run_{args.region}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Phase 1: 数据获取
    # ============================================================
    print("\n" + "="*60)
    print("  Phase 1: 数据获取")
    print("="*60)
    data = acquire_all(use_synthetic=args.synthetic)
    dem = data["dem"]
    transform = data["dem_transform"]
    crs = data["dem_crs"]

    # ============================================================
    # Phase 2: 预处理
    # ============================================================
    print("\n" + "="*60)
    print("  Phase 2: 数据预处理")
    print("="*60)
    terrain_factors = derive_terrain_factors(dem, transform)
    data.update(terrain_factors)

    osm_data = {k: v for k, v in data.items() if k.startswith("osm_")}
    aligned = align_all_rasters(data, dem, transform, osm_data)
    aligned = normalize_features(aligned)

    hard_mask = generate_hard_mask(aligned, aligned["transform"], aligned["shape"])
    soft_mask = generate_soft_mask(aligned, aligned["transform"], aligned["shape"])

    # ============================================================
    # Phase 3: 成本建模 (CNN)
    # ============================================================
    print("\n" + "="*60)
    print("  Phase 3: 深度学习成本建模 (CNN)")
    print("="*60)
    feature_stack = build_feature_stack(aligned)
    labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)

    if not args.skip_training:
        model, training_history = train_cost_unet(feature_stack, labels, hard_mask)
    else:
        from src.dl_models import CostUNet, load_model
        try:
            model = load_model(CostUNet(n_features=cfg.N_FEATURES), "cost_unet_best")
            training_history = {}
        except FileNotFoundError:
            print("  未找到已保存模型, 重新训练...")
            model, training_history = train_cost_unet(feature_stack, labels, hard_mask)

    cost_surface = predict_cost_surface(model, feature_stack, hard_mask)
    save_cost_geotiff(cost_surface, aligned["transform"], crs,
                      out_dir / "cost_surface_v3.tif")

    # ============================================================
    # Phase 4: 神经路径规划
    # ============================================================
    print("\n" + "="*60)
    print("  Phase 4: 神经路径规划 (无A*)")
    print("="*60)

    # 起止点
    if args.start:
        start_lat, start_lon = map(float, args.start.split(","))
    else:
        start_lat, start_lon = cfg.START_POINT
    if args.end:
        end_lat, end_lon = map(float, args.end.split(","))
    else:
        end_lat, end_lon = cfg.END_POINT

    start_rc = geo_to_grid(start_lat, start_lon, aligned["transform"])
    end_rc = geo_to_grid(end_lat, end_lon, aligned["transform"])
    print(f"  起点: ({start_lat:.4f}, {start_lon:.4f}) -> {start_rc}")
    print(f"  终点: ({end_lat:.4f}, {end_lon:.4f}) -> {end_rc}")

    path_coords = neural_path_planning(
        cost_surface, hard_mask, aligned["transform"],
        start_rc, end_rc, feature_stack, soft_mask,
    )

    if path_coords is None:
        print("  错误: 路径规划失败!")
        return

    # ============================================================
    # Phase 5: 质量审查与输出
    # ============================================================
    print("\n" + "="*60)
    print("  Phase 5: 质量审查与输出")
    print("="*60)

    straight_km = compute_path_length_km([
        grid_to_geo(start_rc[0], start_rc[1], aligned["transform"]),
        grid_to_geo(end_rc[0], end_rc[1], aligned["transform"]),
    ])

    quality_result = quality_gate(
        path_coords, aligned, hard_mask, cost_surface,
        aligned["transform"], straight_km,
    )

    export_path(path_coords, out_dir, "optimal")
    stats = compute_statistics(
        path_coords, cost_surface, aligned.get("dem"),
        aligned.get("slope"), hard_mask, aligned["transform"],
        out_dir, "optimal",
    )
    export_quality_report(quality_result, out_dir, "optimal")

    # 可视化
    extent = aligned.get("extent")
    plot_single_path_overview(
        path_coords, aligned.get("dem"), aligned.get("slope"),
        hard_mask, aligned["transform"], "optimal", out_dir, extent,
    )

    # ============================================================
    # 总耗时
    # ============================================================
    elapsed = time.time() - t_total
    print("\n" + "="*60)
    print(f"  v3路径规划完成! 总耗时: {elapsed:.1f}s")
    print(f"  路径长度: {stats.get('length_km', 0):.2f} km")
    print(f"  质量门控: {'通过' if quality_result['passed'] else '未通过'}")
    print(f"  输出目录: {out_dir}")
    print("="*60)

    return path_coords, stats, quality_result


if __name__ == "__main__":
    main()
