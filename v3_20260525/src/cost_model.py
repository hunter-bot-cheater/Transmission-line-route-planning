"""
v3_dl: 模块3 — 深度学习成本建模 (CNN替代随机森林)
版本: v3.20260525
作者: path_planning_team
变更记录:
  - v3.20260525: CostUNet替代RandomForest, 端到端CNN训练, 组合损失函数, 分块预测
  - v2.20260525: 随机森林+伪标签
  - v1.20260525: 初始版本
依赖: v3/config, v3/src/dl_models, v3/src/preprocessing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import rasterio
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
warnings.filterwarnings("ignore")

import config as cfg
from src.dl_models import (
    CostUNet, CostPredictionLoss, save_model, load_model,
)


# ============================================================
# 特征堆叠 — v3: 26维
# ============================================================
def build_feature_stack(aligned):
    """将所有对齐后的栅格堆叠为3D特征数组 (H, W, N_features)"""
    print("[Phase3] 构建特征堆叠 (v3: 26维)...")
    shape = aligned["shape"]
    n_features = cfg.N_FEATURES
    stack = np.full((shape[0], shape[1], n_features), 0.0, dtype=np.float32)

    aligned_to_feature = {
        "dem": "elevation",
        "slope": "slope",
        "aspect_cos": "aspect_cos",
        "aspect_sin": "aspect_sin",
        "tri": "tri",
        "tpi_100": "tpi_100",
        "tpi_300": "tpi_300",
        "tpi_900": "tpi_900",
        "profile_curvature": "profile_curvature",
        "plan_curvature": "plan_curvature",
        "roughness_3": "roughness_3",
        "roughness_9": "roughness_9",
        "roughness_27": "roughness_27",
        "dist_road": "dist_road",
        "dist_water": "dist_water",
        "dist_existing_line": "dist_existing_line",
        "dist_railway": "dist_railway",
        "dist_fault": "dist_fault",
        "landuse_code": "landuse_code",
        "building_density": "building_density",
        "vegetation_height": "vegetation_height",
        "typhoon_risk": "typhoon_risk",
        "seismic_risk": "seismic_risk",
        "landslide_risk": "landslide_risk",
        "ice_cover_risk": "ice_cover_risk",
        "lightning_risk": "lightning_risk",
    }

    for aligned_key, feature_name in aligned_to_feature.items():
        if feature_name in cfg.FEATURE_BANDS and aligned_key in aligned:
            idx = cfg.FEATURE_BANDS[feature_name]
            arr = aligned[aligned_key]
            if arr.shape[:2] == shape:
                stack[:, :, idx] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    stack = np.nan_to_num(stack, nan=0, posinf=0, neginf=0)
    filled = [name for name in cfg.FEATURE_BANDS
              if np.any(stack[:, :, cfg.FEATURE_BANDS[name]] != 0)]
    missing = [name for name in cfg.FEATURE_BANDS if name not in filled]
    if missing:
        print(f"  实际缺失特征: {missing}")
    print(f"  特征堆叠: {stack.shape}, 实际有效波段: {len(filled)}/{n_features}")
    return stack


# ============================================================
# 伪标签生成 — v3: 扩展至10项因素
# ============================================================
def generate_pseudo_labels(aligned, taiwan_lines, hard_mask):
    """
    v3: 扩展伪标签生成 — 10项因素加权。
    """
    print("[Phase3] 生成训练伪标签 (v3扩展: 10项因素)...")
    shape = aligned["shape"]
    dem = aligned.get("dem")
    slope = aligned.get("slope")
    dist_existing = aligned.get("dist_existing_line")
    dist_water = aligned.get("dist_water")
    dist_road = aligned.get("dist_road")
    dist_railway = aligned.get("dist_railway")
    dist_fault = aligned.get("dist_fault")
    landuse = aligned.get("landuse_code")
    build_density = aligned.get("building_density")
    landslide = aligned.get("landslide_risk")
    roughness = aligned.get("roughness_9")
    ice_cover = aligned.get("ice_cover_risk")
    lightning = aligned.get("lightning_risk")
    veg = aligned.get("vegetation_height")

    pp = cfg.PSEUDO_LABEL_PARAMS

    # 1. 距现有线路
    if dist_existing is not None:
        d_existing_score = 1.0 - np.exp(-dist_existing / pp["dist_existing_decay"])
    else:
        d_existing_score = np.full(shape, 0.4, dtype=np.float32)

    # 2. 坡度
    if slope is not None:
        s_score = np.clip(slope / pp["slope_threshold"], 0, 1)
        s_score = np.where(slope > pp["slope_extra_penalty"], s_score * 1.5, s_score)
    else:
        s_score = np.full(shape, 0.3, dtype=np.float32)

    # 3. 土地利用
    if landuse is not None:
        lu_cost_map = {1: 0.05, 2: 0.25, 3: 0.15, 4: 0.80, 5: 0.95, 6: 0.90, 7: 0.70, 8: 0.30}
        lu_score = np.zeros(shape, dtype=np.float32)
        for code in range(1, 9):
            lu_score[landuse == code] = lu_cost_map.get(code, 0.30)
    else:
        lu_score = np.full(shape, 0.30, dtype=np.float32)

    # 4. 建筑密度
    if build_density is not None and np.any(build_density > 0):
        b_score = np.clip(build_density / 5000.0, 0, 1)
    else:
        b_score = np.full(shape, 0.1, dtype=np.float32)

    # 5. 道路可达性
    if dist_road is not None:
        r_score = 1.0 - np.exp(-dist_road / 2000)
    else:
        r_score = np.full(shape, 0.3, dtype=np.float32)

    # 6. 铁路邻近
    if dist_railway is not None:
        rail_score = 1.0 - np.exp(-dist_railway / 3000)
    else:
        rail_score = np.full(shape, 0.3, dtype=np.float32)

    # 7. 水域穿越
    if dist_water is not None:
        w_score = np.exp(-dist_water / pp["water_decay"])
    else:
        w_score = np.full(shape, 0.0, dtype=np.float32)

    # 8. 滑坡风险
    if landslide is not None:
        ls_score = landslide
    else:
        ls_score = np.full(shape, 0.2, dtype=np.float32)

    # 9. 粗糙度
    if roughness is not None:
        rough_score = np.clip(roughness / pp["roughness_threshold"], 0, 1)
    else:
        rough_score = np.full(shape, 0.2, dtype=np.float32)

    # 10. [v3新增] 断裂带
    if dist_fault is not None:
        fault_score = np.exp(-dist_fault / pp["fault_decay"])
    else:
        fault_score = np.full(shape, 0.1, dtype=np.float32)

    # 11. [v3新增] 覆冰
    if ice_cover is not None:
        ice_score = np.clip(ice_cover / pp["ice_threshold"], 0, 1)
    else:
        ice_score = np.full(shape, 0.1, dtype=np.float32)

    # 12. [v3新增] 雷击
    if lightning is not None:
        light_score = np.clip(lightning / pp["lightning_threshold"], 0, 1)
    else:
        light_score = np.full(shape, 0.1, dtype=np.float32)

    # 13. [v3新增] 植被
    if veg is not None:
        veg_score = np.clip(veg / 30.0, 0, 1)
    else:
        veg_score = np.full(shape, 0.1, dtype=np.float32)

    # 加权
    w = cfg.LABEL_WEIGHTS
    labels = (
        w["dist_existing"] * d_existing_score +
        w["slope"] * s_score +
        w["landuse"] * (0.5 * lu_score + 0.3 * b_score + 0.2 * rough_score) +
        w["road_access"] * r_score +
        w["railway"] * rail_score +
        w["water"] * w_score +
        w["protected"] * ls_score +
        w["roughness"] * rough_score +
        w["fault"] * fault_score +
        w["ice_cover"] * ice_score +
        w["lightning"] * light_score +
        w["vegetation"] * veg_score
    )

    rng = np.random.RandomState(42)
    noise = rng.uniform(-0.03, 0.03, size=shape).astype(np.float32)
    labels = np.clip(labels + noise, 0, 1)

    labels = labels.astype(np.float32)
    if hard_mask is not None:
        labels[hard_mask == 0] = np.nan

    print(f"  伪标签: 范围[{np.nanmin(labels):.3f}, {np.nanmax(labels):.3f}]")
    return labels


# ============================================================
# 数据准备
# ============================================================
def prepare_training_data(feature_stack, labels, hard_mask, train_split=None):
    """准备PyTorch训练数据集"""
    print("[Phase3] 准备训练数据...")
    H, W, C = feature_stack.shape

    valid = ~np.isnan(labels)
    if hard_mask is not None:
        valid = valid & (hard_mask == 1)

    y = labels.copy()
    y[~valid] = 0.0
    valid_frac = valid.sum() / valid.size
    print(f"  有效训练像元: {valid.sum()} ({valid_frac*100:.1f}%)")

    X_t = torch.from_numpy(feature_stack).permute(2, 0, 1)
    y_t = torch.from_numpy(y).unsqueeze(0)
    mask_t = torch.from_numpy(valid).unsqueeze(0)

    # 切分train/val
    split = train_split or cfg.DL_TRAIN_SPLIT
    split_col = int(W * split)
    train_X = X_t[:, :, :split_col]
    train_y = y_t[:, :, :split_col]
    train_mask = mask_t[:, :, :split_col]
    val_X = X_t[:, :, split_col:]
    val_y = y_t[:, :, split_col:]
    val_mask = mask_t[:, :, split_col:]

    print(f"  训练区域: {train_X.shape[1:]} (左{int(split*100)}%)")
    print(f"  验证区域: {val_X.shape[1:]} (右{int((1-split)*100)}%)")
    return train_X, train_y, train_mask, val_X, val_y, val_mask


# ============================================================
# CNN训练
# ============================================================
def train_cost_unet(feature_stack, labels, hard_mask):
    """
    v3: 训练CostUNet深度学习成本预测模型。
    """
    print("[Phase3] 训练 CostUNet 深度学习成本模型...")
    print(f"  设备: {cfg.DEVICE}")

    train_X, train_y, train_mask, val_X, val_y, val_mask = \
        prepare_training_data(feature_stack, labels, hard_mask)

    model = CostUNet(
        n_features=cfg.N_FEATURES,
        dropout=cfg.UNET_DROPOUT,
    ).to(cfg.DEVICE)

    criterion = CostPredictionLoss(
        mse_weight=1.0,
        grad_weight=0.3,
        constraint_weight=2.0,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.DL_LEARNING_RATE,
        weight_decay=cfg.DL_WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True,
    )

    # 提取patches进行训练
    patch_size = 128
    stride = 64

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"  开始训练... (epochs={cfg.DL_N_EPOCHS}, batch_size={cfg.DL_BATCH_SIZE})")
    print(f"  早停patience={cfg.DL_EARLY_STOP}")

    t0 = time.time()
    for epoch in range(cfg.DL_N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # 随机采样patches
        _, H, W = train_X.shape
        for _ in range(200):
            r = np.random.randint(0, max(1, H - patch_size))
            c = np.random.randint(0, max(1, W - patch_size))
            patch_X = train_X[:, r:r+patch_size, c:c+patch_size].unsqueeze(0).to(cfg.DEVICE)
            patch_y = train_y[:, r:r+patch_size, c:c+patch_size].unsqueeze(0).to(cfg.DEVICE)
            patch_mask = train_mask[:, r:r+patch_size, c:c+patch_size].unsqueeze(0).to(cfg.DEVICE)

            if patch_mask.sum() < 100:
                continue

            optimizer.zero_grad()
            pred = model(patch_X)
            hard_patch = (patch_mask.float() == 0).float()
            loss, loss_components = criterion(pred, patch_y, hard_patch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        n_val = 0
        _, vH, vW = val_X.shape
        with torch.no_grad():
            for _ in range(50):
                r = np.random.randint(0, max(1, vH - patch_size))
                c = np.random.randint(0, max(1, vW - patch_size))
                pv_X = val_X[:, r:r+patch_size, c:c+patch_size].unsqueeze(0).to(cfg.DEVICE)
                pv_y = val_y[:, r:r+patch_size, c:c+patch_size].unsqueeze(0).to(cfg.DEVICE)
                pv_mask = val_mask[:, r:r+patch_size, c:c+patch_size].unsqueeze(0).to(cfg.DEVICE)

                if pv_mask.sum() < 100:
                    continue

                pred = model(pv_X)
                hard_pv = (pv_mask.float() == 0).float()
                vloss, _ = criterion(pred, pv_y, hard_pv)
                val_loss += vloss.item()
                n_val += 1

        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{cfg.DL_N_EPOCHS} | "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model(model, "cost_unet_best")
        else:
            patience_counter += 1
            if patience_counter >= cfg.DL_EARLY_STOP:
                print(f"  早停触发 (epoch {epoch+1}), 最佳val loss: {best_val_loss:.4f}")
                break

    elapsed = time.time() - t0
    print(f"  训练完成 ({elapsed:.1f}s), 最佳val loss: {best_val_loss:.4f}")

    # 加载最佳模型
    try:
        model = load_model(CostUNet(n_features=cfg.N_FEATURES), "cost_unet_best")
    except FileNotFoundError:
        save_model(model, "cost_unet_best")
        print(f"  模型已保存 (最终epoch)")

    return model, {"train_losses": train_losses, "val_losses": val_losses}


# ============================================================
# 成本预测
# ============================================================
def predict_cost_surface(model, feature_stack, hard_mask):
    """使用训练好的CostUNet预测全区域成本表面"""
    print("[Phase3] CNN预测成本表面...")
    model.eval()
    H, W, C = feature_stack.shape

    cost = model.predict_cost_surface(feature_stack, batch_size=4)

    if hard_mask is not None:
        cost[hard_mask == 0] = np.inf

    cost = np.clip(cost, 0, None)
    if np.any(cost < np.inf):
        print(f"  成本表面: [{np.min(cost[cost < np.inf]):.4f}, "
              f"{np.max(cost[cost < np.inf]):.4f}]")
    print(f"  成本表面预测完成")
    return cost


def save_cost_geotiff(cost, transform, crs, path):
    """保存成本表面为GeoTIFF"""
    cost_save = np.where(np.isinf(cost), -9999, cost).astype(np.float32)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=cost.shape[0],
        width=cost.shape[1],
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(cost_save, 1)
    print(f"  成本栅格已保存: {path}")
