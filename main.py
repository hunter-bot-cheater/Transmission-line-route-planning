"""
台湾输电线路智能路径规划系统 — 主入口
完整5阶段流程: 数据获取 → 预处理 → 成本建模 → 路径搜索 → 输出

使用方法:
    python main.py
    python main.py --start 21.95 120.75 --end 25.03 121.53
    python main.py --skip-phase 1  # 跳过数据获取(使用已有缓存)
"""
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from src.data_acquisition import acquire_all
from src.preprocessing import derive_terrain_factors, generate_hard_mask, generate_soft_mask, align_all_rasters
from src.cost_model import (
    build_feature_stack, generate_pseudo_labels,
    train_random_forest, predict_cost_surface, save_cost_geotiff,
)
from src.path_planning import (
    fuse_cost_surface, astar_search, smooth_path,
    geo_to_grid, grid_to_geo_coords, compute_path_length_km,
)
from src.output import (
    export_path, compute_statistics, create_overview_map,
    create_detail_map, create_elevation_profile, create_interactive_map,
)


def parse_args():
    parser = argparse.ArgumentParser(description="台湾输电线路智能路径规划")
    parser.add_argument("--start-lat", type=float, default=cfg.START_POINT[0], help="起点纬度")
    parser.add_argument("--start-lon", type=float, default=cfg.START_POINT[1], help="起点经度")
    parser.add_argument("--end-lat", type=float, default=cfg.END_POINT[0], help="终点纬度")
    parser.add_argument("--end-lon", type=float, default=cfg.END_POINT[1], help="终点经度")
    parser.add_argument("--skip-phase", type=int, nargs="+", default=[], help="跳过的阶段(1-5)")
    parser.add_argument("--config", type=str, default=None, help="start_end.json配置文件路径")
    return parser.parse_args()


def load_config_file(config_path):
    """从JSON文件加载起止点配置"""
    if config_path is None:
        config_path = cfg.OUTPUT_DIR / "start_end.json"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (data.get("start_lat", cfg.START_POINT[0]),
                data.get("start_lon", cfg.START_POINT[1]),
                data.get("end_lat", cfg.END_POINT[0]),
                data.get("end_lon", cfg.END_POINT[1]))
    return None


def main():
    print("=" * 60)
    print("  台湾输电线路智能路径规划系统")
    print("  Taiwan Transmission Line Intelligent Path Planning")
    print("=" * 60)

    args = parse_args()

    # 尝试加载配置文件
    config = load_config_file(args.config)
    if config:
        start_lat, start_lon, end_lat, end_lon = config
        print(f"  使用配置文件起止点")
    else:
        start_lat, start_lon = args.start_lat, args.start_lon
        end_lat, end_lon = args.end_lat, args.end_lon

    print(f"  起点: ({start_lat:.4f}, {start_lon:.4f})")
    print(f"  终点: ({end_lat:.4f}, {end_lon:.4f})")
    print()

    total_start = time.time()

    # ============================================================
    # Phase 1: 数据获取
    # ============================================================
    if 1 not in args.skip_phase:
        print("=" * 50)
        print("  Phase 1/5: 数据获取")
        print("=" * 50)
        data = acquire_all()
        print("  Phase 1 完成\n")
    else:
        print("  Phase 1: 已跳过\n")
        # FIXME: 从缓存加载... 此处简化，假设phase 1必须运行
        data = acquire_all()

    # ============================================================
    # Phase 2: 数据预处理
    # ============================================================
    if 2 not in args.skip_phase:
        print("=" * 50)
        print("  Phase 2/5: 数据预处理")
        print("=" * 50)

        dem = data["dem"]
        transform = data["dem_transform"]
        crs = data["dem_crs"]

        # 地形因子
        terrain_factors = derive_terrain_factors(dem, transform)
        data.update(terrain_factors)

        # 栅格对齐
        aligned = align_all_rasters(data, dem, transform)
        dst_transform = aligned["transform"]
        dst_shape = aligned["shape"]

        # 约束掩膜
        hard_mask = generate_hard_mask(aligned, dst_transform, dst_shape)
        soft_mask = generate_soft_mask(aligned, dst_transform, dst_shape)

        aligned["hard_mask"] = hard_mask
        aligned["soft_mask"] = soft_mask

        # 保存中间结果
        print("  Phase 2 完成\n")
    else:
        print("  Phase 2: 已跳过\n")

    # ============================================================
    # Phase 3: 成本建模
    # ============================================================
    if 3 not in args.skip_phase:
        print("=" * 50)
        print("  Phase 3/5: 成本建模(随机森林)")
        print("=" * 50)

        # 特征堆叠
        feature_stack = build_feature_stack(aligned)

        # 伪标签
        taiwan_lines = data.get("taiwan_lines")
        labels = generate_pseudo_labels(aligned, taiwan_lines, hard_mask)

        # 训练
        rf, scaler, importances = train_random_forest(feature_stack, labels, hard_mask)

        # 预测成本表面
        predicted_cost = predict_cost_surface(rf, scaler, feature_stack, hard_mask)

        # 保存成本栅格
        save_cost_geotiff(predicted_cost, dst_transform, crs, cfg.OUTPUT_DIR / "cost_surface.tif")

        print("  Phase 3 完成\n")
    else:
        print("  Phase 3: 已跳过\n")

    # ============================================================
    # Phase 4: 路径搜索
    # ============================================================
    if 4 not in args.skip_phase:
        print("=" * 50)
        print("  Phase 4/5: 路径规划(A*)")
        print("=" * 50)

        # 成本融合
        final_cost = fuse_cost_surface(predicted_cost, soft_mask, hard_mask)

        # 保存最终成本表面
        save_cost_geotiff(final_cost, dst_transform, crs, cfg.OUTPUT_DIR / "final_cost_surface.tif")

        # 保存硬约束掩膜
        import rasterio
        with rasterio.open(
            cfg.OUTPUT_DIR / "constraint_mask.tif", "w",
            driver="GTiff", height=hard_mask.shape[0], width=hard_mask.shape[1],
            count=1, dtype=np.uint8, crs=crs, transform=dst_transform,
        ) as dst:
            dst.write(hard_mask.astype(np.uint8), 1)

        # 坐标转换
        start_rc = geo_to_grid(start_lat, start_lon, dst_transform)
        end_rc = geo_to_grid(end_lat, end_lon, dst_transform)

        # A*搜索
        path_cells = astar_search(final_cost, start_rc, end_rc)

        if path_cells is None:
            print("  [错误] A*未找到路径!")
            print("  请检查起止点是否在硬约束区域内或增大搜索范围")
            sys.exit(1)

        # 路径平滑
        smoothed_coords = smooth_path(path_cells, dst_transform, hard_mask)

        # 路径长度
        path_len = compute_path_length_km(smoothed_coords)
        print(f"  最优路径长度: {path_len:.2f} km")
        print(f"  路径顶点数: {len(smoothed_coords)}")

        print("  Phase 4 完成\n")
    else:
        print("  Phase 4: 已跳过\n")

    # ============================================================
    # Phase 5: 输出与可视化
    # ============================================================
    if 5 not in args.skip_phase:
        print("=" * 50)
        print("  Phase 5/5: 输出与可视化")
        print("=" * 50)

        output_dir = cfg.OUTPUT_DIR

        # SHP / GeoJSON
        export_path(smoothed_coords, output_dir)

        # 统计
        stats = compute_statistics(
            smoothed_coords, final_cost,
            aligned.get("dem"), aligned.get("slope"),
            hard_mask, dst_transform, output_dir,
        )

        # 可视化
        create_overview_map(
            smoothed_coords, final_cost, aligned.get("dem"),
            data.get("taiwan_lines"), hard_mask, dst_transform, output_dir,
        )

        create_detail_map(
            smoothed_coords, final_cost, aligned.get("dem"),
            hard_mask, dst_transform, output_dir,
        )

        create_elevation_profile(
            smoothed_coords, aligned.get("dem"), aligned.get("slope"),
            dst_transform, output_dir,
        )

        create_interactive_map(
            smoothed_coords, final_cost, aligned.get("dem"),
            data.get("taiwan_lines"), hard_mask, dst_transform, output_dir,
        )

        print("  Phase 5 完成\n")

    # ============================================================
    # 汇总
    # ============================================================
    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"  [完成] 全部5个阶段运行完毕!")
    print(f"  总耗时: {total_elapsed / 60:.1f} 分钟")
    print(f"  输出目录: {cfg.OUTPUT_DIR}")
    print(f"  - optimal_path.shp / .geojson  路径矢量文件")
    print(f"  - cost_surface.tif             成本表面栅格")
    print(f"  - constraint_mask.tif          约束掩膜")
    print(f"  - statistics.json / .xlsx      统计指标")
    print(f"  - map_overview.png             概览图")
    print(f"  - map_detail.png               局部放大图")
    print(f"  - elevation_profile.png        高程剖面图")
    print(f"  - interactive_map.html         交互式地图(浏览器打开)")
    print("=" * 60)


if __name__ == "__main__":
    main()
