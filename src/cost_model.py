"""
模块3: 成本建模
- 构建多维特征数据集
- 伪标签生成(基于现有输电线+地形)
- 随机森林训练与预测
- 输出成本表面
"""
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

import config as cfg


# ============================================================
# 特征堆叠
# ============================================================
def build_feature_stack(aligned):
    """将所有对齐后的栅格堆叠为3D特征数组 (H, W, N_features)"""
    print("[Phase3] 构建特征堆叠...")

    shape = aligned["shape"]
    n_features = cfg.N_FEATURES
    stack = np.full((shape[0], shape[1], n_features), 0.0, dtype=np.float32)

    # 对齐字典键 -> 特征名映射
    key_to_feature = {
        "dem": "elevation",
        "slope": "slope",
        "aspect_cos": "aspect_cos",
        "aspect_sin": "aspect_sin",
        "tri": "tri",
        "tpi": "tpi",
        "profile_curvature": "profile_curvature",
        "plan_curvature": "plan_curvature",
        "roughness": "roughness",
        "dist_road": "dist_road",
        "dist_water": "dist_water",
        "dist_existing_line": "dist_existing_line",
        "typhoon_risk": "typhoon_risk",
        "seismic_risk": "seismic_risk",
        "landslide_risk": "landslide_risk",
    }

    for aligned_key, feature_name in key_to_feature.items():
        if feature_name in cfg.FEATURE_BANDS and aligned_key in aligned:
            idx = cfg.FEATURE_BANDS[feature_name]
            arr = aligned[aligned_key]
            if arr.shape[:2] == shape:
                stack[:, :, idx] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 缺失的特征(landuse_code, building_density)默认为0
    # 这些需要从OSM数据导出的特征暂时用0填充

    # 无限值处理
    stack = np.nan_to_num(stack, nan=0, posinf=0, neginf=0)

    print(f"  特征堆叠: {stack.shape}")
    return stack


# ============================================================
# 伪标签生成
# ============================================================
def generate_pseudo_labels(aligned, taiwan_lines, hard_mask):
    """
    基于现有输电线+地形因素生成伪标签
    现有线路60m缓冲区 = 低成本
    远离线路 + 陡坡 = 高成本
    标签范围: 0(低成本) ~ 1(高成本)
    """
    print("[Phase3] 生成训练伪标签...")

    shape = aligned["shape"]
    transform = aligned["transform"]
    dem = aligned.get("dem")
    slope = aligned.get("slope")
    dist_existing = aligned.get("dist_existing_line")
    dist_water = aligned.get("dist_water")
    dist_road = aligned.get("dist_road")

    # 1. 距现有线路距离得分
    if dist_existing is not None:
        # 距离越远成本越高
        d_score = 1.0 - np.exp(-dist_existing / 800)
    else:
        d_score = np.full(shape, 0.5, dtype=np.float32)

    # 2. 坡度得分
    if slope is not None:
        s_score = np.clip(slope / cfg.MAX_SLOPE, 0, 1)
    else:
        s_score = np.full(shape, 0.3, dtype=np.float32)

    # 3. 土地利用代理 (基于坡度+DEM粗糙度)
    if dem is not None:
        from scipy.ndimage import uniform_filter
        roughness = np.abs(dem - uniform_filter(np.nan_to_num(dem, nan=0), size=3))
        r_score = np.clip(roughness / 100, 0, 1)
    else:
        r_score = np.full(shape, 0.3, dtype=np.float32)

    # 4. 道路可达性: 距道路越近成本越低(材料运输方便)
    if dist_road is not None:
        road_score = 1.0 - np.exp(-dist_road / 1500)
    else:
        road_score = np.full(shape, 0.3, dtype=np.float32)

    # 5. 水域惩罚
    if dist_water is not None:
        water_score = np.exp(-dist_water / 100)
    else:
        water_score = np.full(shape, 0.0, dtype=np.float32)

    # 6. 保护区惩罚: 已在hard_mask中体现

    # 加权组合
    w = cfg.LABEL_WEIGHTS
    labels = (
        w["dist_existing"] * d_score +
        w["slope"] * s_score +
        w["landuse"] * r_score +
        w["road_access"] * road_score +
        w["water"] * water_score +
        w["protected"] * 0.0  # 保护区已在硬约束中
    )

    # 硬约束区域标签置为NaN(不参与训练)
    labels = labels.astype(np.float32)
    if hard_mask is not None:
        labels[hard_mask == 0] = np.nan

    labels = np.clip(labels, 0, 1)
    print(f"  伪标签生成完成, 范围: [{np.nanmin(labels):.3f}, {np.nanmax(labels):.3f}]")
    return labels


# ============================================================
# 随机森林训练
# ============================================================
def train_random_forest(feature_stack, labels, hard_mask):
    """
    训练随机森林回归模型预测建设成本
    """
    print("[Phase3] 训练随机森林成本模型...")

    H, W, N = feature_stack.shape

    # 展平为 (n_pixels, n_features)
    X = feature_stack.reshape(-1, N)
    y = labels.ravel()

    # 只使用有效样本(非NaN标签 且 非硬约束区域)
    if hard_mask is not None:
        valid = (~np.isnan(y)) & (hard_mask.ravel() == 1)
    else:
        valid = ~np.isnan(y)

    X_valid = X[valid]
    y_valid = y[valid]

    # 采样: 如果样本过多, 随机采样以加速训练
    max_samples = 100000
    if X_valid.shape[0] > max_samples:
        idx = np.random.RandomState(cfg.RF_RANDOM_STATE).choice(
            X_valid.shape[0], max_samples, replace=False
        )
        X_train_full = X_valid[idx]
        y_train_full = y_valid[idx]
    else:
        X_train_full = X_valid
        y_train_full = y_valid

    print(f"  训练样本: {X_train_full.shape[0]}")

    # 处理无效值
    X_train_full = np.nan_to_num(X_train_full, nan=0, posinf=0, neginf=0)

    # 分割训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full,
        test_size=cfg.RF_TEST_SIZE,
        random_state=cfg.RF_RANDOM_STATE,
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练
    print(f"  训练中... n_estimators={cfg.RF_N_ESTIMATORS}, max_depth={cfg.RF_MAX_DEPTH}")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_depth=cfg.RF_MAX_DEPTH,
        min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        n_jobs=-1,
        random_state=cfg.RF_RANDOM_STATE,
        verbose=0,
    )
    rf.fit(X_train_scaled, y_train)

    # 评估
    train_score = rf.score(X_train_scaled, y_train)
    test_score = rf.score(X_test_scaled, y_test)
    elapsed = time.time() - t0
    print(f"  训练完成 ({elapsed:.1f}s)")
    print(f"  训练R^2: {train_score:.4f}")
    print(f"  测试R^2: {test_score:.4f}")

    # 特征重要性
    importances = rf.feature_importances_
    band_names = [name for name, _ in sorted(cfg.FEATURE_BANDS.items(), key=lambda x: x[1])]
    print("  特征重要性 Top 5:")
    top_idx = np.argsort(importances)[::-1][:5]
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {band_names[idx]}: {importances[idx]:.4f}")

    # 保存模型
    model_path = cfg.MODELS_DIR / "rf_cost_model.pkl"
    scaler_path = cfg.MODELS_DIR / "scaler.pkl"
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  模型已保存: {model_path}")

    return rf, scaler, importances


# ============================================================
# 成本预测
# ============================================================
def predict_cost_surface(rf, scaler, feature_stack, hard_mask):
    """
    使用训练好的RF预测全区域成本表面
    """
    print("[Phase3] 预测成本表面...")

    H, W, N = feature_stack.shape
    X = feature_stack.reshape(-1, N)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_scaled = scaler.transform(X)

    # 分批预测(避免内存溢出)
    batch_size = 50000
    predicted = np.zeros(X.shape[0], dtype=np.float32)
    for i in range(0, X.shape[0], batch_size):
        end = min(i + batch_size, X.shape[0])
        predicted[i:end] = rf.predict(X_scaled[i:end]).astype(np.float32)
        if i % (batch_size * 10) == 0:
            print(f"  预测进度: {i / X.shape[0] * 100:.0f}%")

    predicted = predicted.reshape(H, W)

    # 硬约束区域设为inf
    if hard_mask is not None:
        predicted[hard_mask == 0] = np.inf

    predicted = np.clip(predicted, 0, None)
    print(f"  成本表面预测完成, 范围: [{np.nanmin(predicted[predicted < np.inf]):.4f}, {np.nanmax(predicted[predicted < np.inf]):.4f}]")
    return predicted


# ============================================================
# 保存成本栅格为GeoTIFF
# ============================================================
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
