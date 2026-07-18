"""
注意力增强A*启发式 — 训练神经网络预测剩余代价
用A*最优路径上的(特征, 到终点距离) → 实际剩余代价 作为训练数据
"""
import sys
from pathlib import Path
_v3_dir = Path(__file__).resolve().parent.parent
_v2_dir = _v3_dir.parent / "v2_20260525"; _v2_src = _v2_dir / "src"
_shared_dir = _v3_dir.parent / "shared"
for _d in reversed([str(_v3_dir / "src"), str(_v3_dir), str(_v2_dir), str(_v2_src), str(_shared_dir)]):
    sys.path.insert(0, _d)

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {DEVICE}")

# ── 1. 生成训练数据: A*最优路径上每个点的(特征, 距离) → 剩余真实代价 ──
print("生成训练数据...")
import config as cfg
from data_acquisition_v3 import load_taiwan_lines, acquire_all
from preprocessing import derive_terrain_factors, generate_hard_mask, generate_soft_mask, align_all_rasters
from cost_model import build_feature_stack, generate_pseudo_labels, train_random_forest, predict_cost_surface
from path_planning import fuse_cost_surface, astar_search, smooth_path, compute_path_length_km, haversine_m, geo_to_grid

# 预处理 (复用管线)
t0 = time.time()
data = acquire_all()
dem = data["dem"]; transform = data["dem_transform"]; crs_data = data["dem_crs"]
terrain_factors = derive_terrain_factors(dem, transform); data.update(terrain_factors)
test_way_ids = {case["way_id"] for case in cfg.TEST_CASES}
tl = data.get("taiwan_lines")
if tl is not None: data["taiwan_lines"] = tl[~tl["id"].isin(test_way_ids)].copy()
aligned = align_all_rasters(data, dem, transform, {})
dst_transform = aligned["transform"]; dst_shape = aligned["shape"]
hard_mask = generate_hard_mask(dict(aligned), dst_transform, dst_shape)
soft_mask = generate_soft_mask(dict(aligned), dst_transform, dst_shape)
feature_stack = build_feature_stack(aligned)
labels = generate_pseudo_labels(aligned, data.get("taiwan_lines"), hard_mask)
rf, scaler, _ = train_random_forest(feature_stack, labels, hard_mask)
predicted_cost = predict_cost_surface(rf, scaler, feature_stack, hard_mask)
dist_existing = aligned.get("dist_existing_line")
final_cost = fuse_cost_surface(predicted_cost, soft_mask, hard_mask, dist_existing=dist_existing)
print(f"  预处理完成 ({time.time()-t0:.0f}s)")

# 跑A*收集训练数据
gdf = load_taiwan_lines()
X_list, Y_list = [], []
MAX_SAMPLES = 50000

for case in cfg.TEST_CASES[:5]:  # 只用前5条线路训练
    rows = gdf[gdf["id"] == case["way_id"]]
    if len(rows) == 0: continue
    rc = list(rows.iloc[0].geometry.coords)
    slat, slon = rc[0][1], rc[0][0]
    elat, elon = rc[-1][1], rc[-1][0]
    start_rc = geo_to_grid(slat, slon, dst_transform)
    end_rc = geo_to_grid(elat, elon, dst_transform)
    end_latlon = (elat, elon)

    path = astar_search(final_cost, start_rc, end_rc)
    if path is None: continue

    # 从终点向前回溯, 计算每步的真实剩余代价
    H, W, C = feature_stack.shape
    total_cost = 0.0
    for i in range(len(path)-2, -1, -1):  # 从倒数第二步回溯到起点
        r, c = path[i]
        nr, nc = path[i+1]
        step_cost = final_cost[nr, nc] * (np.sqrt(2) if r!=nr and c!=nc else 1.0)
        total_cost += step_cost
        feat = feature_stack[r, c, :]  # (18,)
        dist_to_goal = haversine_m(elon, elat, dst_transform.c + c*dst_transform.a, dst_transform.f + r*dst_transform.e)
        X_list.append(np.concatenate([feat, [dist_to_goal/1000.0]]))  # 18+1=19维
        Y_list.append(total_cost)

    print(f"  {case['case_id']}: {len(path)} 路径点, {len(path)-1} 样本")
    if len(X_list) > MAX_SAMPLES: break

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32).reshape(-1, 1)
print(f"总样本: {len(X)}")

# ── 2. 训练小MLP ──
class HeuristicNet(nn.Module):
    def __init__(self, in_dim=19):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x)

# 划分
n = len(X); n_train = int(n*0.8)
idx = np.random.RandomState(42).permutation(n)
X_train, Y_train = X[idx[:n_train]], Y[idx[:n_train]]
X_val, Y_val = X[idx[n_train:]], Y[idx[n_train:]]

# 归一化
x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)+1e-8
X_train = (X_train - x_mean) / x_std; X_val = (X_val - x_mean) / x_std

train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
val_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))

model = HeuristicNet().to(DEVICE)
opt = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.SmoothL1Loss()

print(f"\n训练 MLP 启发式网络 ({n_train} train, {len(Y_val)} val)...")
best_val = float('inf')
for epoch in range(100):
    model.train()
    for xb, yb in DataLoader(train_ds, batch_size=256, shuffle=True):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad(); loss = criterion(model(xb), yb); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_t := torch.from_numpy(X_val).to(DEVICE)),
                            torch.from_numpy(Y_val).to(DEVICE)).item()
    if val_loss < best_val:
        best_val = val_loss
        torch.save({'model': model.state_dict(), 'x_mean': x_mean, 'x_std': x_std},
                   _v3_dir / "cnn_model" / "heuristic_net.pth")
    if (epoch+1) % 20 == 0:
        print(f"  epoch {epoch+1}: val_loss={val_loss:.6f}")

print(f"\n训练完成! 最佳 val_loss={best_val:.6f}")
print(f"模型保存: cnn_model/heuristic_net.pth")
