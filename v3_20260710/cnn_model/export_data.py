"""
CNN 训练数据导出 — 从 V2 管线提取特征栈 + 伪标签
版本: v3.20260710
输出: feature_stack.npy (H×W×18), labels.npy (H×W), mask.npy (H×W bool)
"""
import sys
from pathlib import Path

_v3_dir = Path(__file__).resolve().parent.parent
_v2_dir = _v3_dir.parent / "v2_20260525"
_v2_src = _v2_dir / "src"
_shared_dir = _v3_dir.parent / "shared"
for _d in reversed([str(_v3_dir), str(_v2_dir), str(_v2_src), str(_shared_dir)]):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np
import time
import config as cfg
from data_acquisition import acquire_all
from preprocessing import derive_terrain_factors, generate_hard_mask, align_all_rasters
from cost_model import build_feature_stack, generate_pseudo_labels

print("=" * 60)
print("  CNN 训练数据导出")
print("=" * 60)

# Phase 1-2: 数据 + 预处理 (复用 V2)
print("\n[1/3] 数据获取 + 预处理...")
t0 = time.time()
data = acquire_all()
dem = data["dem"]
transform = data["dem_transform"]
terrain_factors = derive_terrain_factors(dem, transform)
data.update(terrain_factors)

osm_data = {
    "osm_landuse": data.get("osm_landuse"),
    "osm_buildings": data.get("osm_buildings"),
    "osm_railways": data.get("osm_railways"),
    "osm_airports": data.get("osm_airports"),
}
aligned = align_all_rasters(data, dem, transform, osm_data=osm_data)
dst_transform = aligned["transform"]

aligned_for_mask = dict(aligned)
for key in ["osm_water", "osm_protected"]:
    if key in data and data[key] is not None:
        aligned_for_mask[key] = data[key]
hard_mask = generate_hard_mask(aligned_for_mask, dst_transform, aligned["shape"])
print(f"  预处理完成, 耗时: {time.time() - t0:.0f}s")

# Phase 3: 特征栈 + 伪标签
print("\n[2/3] 生成特征栈 + 伪标签...")
t0 = time.time()
feature_stack = build_feature_stack(aligned)  # (H, W, 18)
labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)  # (H, W)

# NaN 填充
labels = np.nan_to_num(labels, nan=0.0)
feature_stack = np.nan_to_num(feature_stack, nan=0.0)

# 有效掩膜 (硬约束区域内不训练)
valid_mask = (hard_mask == 1) & (labels > 0.001)
print(f"  特征栈: {feature_stack.shape}, 标签: {labels.shape}")
print(f"  有效像素: {valid_mask.sum():,} / {valid_mask.size:,} ({100*valid_mask.sum()/valid_mask.size:.1f}%)")
print(f"  标签范围: [{labels[valid_mask].min():.4f}, {labels[valid_mask].max():.4f}]")
print(f"  耗时: {time.time() - t0:.0f}s")

# Phase 4: 保存
print("\n[3/3] 保存 .npy 文件...")
out_dir = Path(__file__).resolve().parent
np.save(out_dir / "feature_stack.npy", feature_stack.astype(np.float32))
np.save(out_dir / "labels.npy", labels.astype(np.float32))
np.save(out_dir / "valid_mask.npy", valid_mask)
print(f"  已保存到: {out_dir}")

# 文件大小
for fname in ["feature_stack.npy", "labels.npy", "valid_mask.npy"]:
    fpath = out_dir / fname
    if fpath.exists():
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  {fname}: {size_mb:.1f} MB")

print("\n导出完成!")
