"""
CNN 成本预测模型 — U-Net 像素级回归
版本: v3.20260710
输入: 18通道特征栈 → U-Net → 1通道成本表面
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# ── 超参数 ──
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 80
TILE_SIZE = 128
TRAIN_RATIO = 0.8
MAX_TRAIN_SAMPLES = 20000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── U-Net 模型 ──
class DoubleConv(nn.Module):
    """(Conv → BN → ReLU) × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """轻量 U-Net: 18通道输入 → 1通道成本输出"""
    def __init__(self, in_channels=18, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # 编码器
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f

        # 瓶颈
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 解码器
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.out = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skip = []
        for down in self.downs:
            x = down(x)
            skip.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)                    # 上采样
            s = skip[-(i // 2 + 1)]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, size=s.shape[2:], mode="bilinear")
            x = torch.cat([x, s], dim=1)          # 跳连
            x = self.ups[i + 1](x)                 # 双卷积

        return self.out(x)


# ── 数据集 ──
class TileDataset(Dataset):
    """从大栅格中随机采样 256×256 瓦片"""
    def __init__(self, features, labels, mask, tile_size=256, n_samples=10000):
        self.features = features  # (H, W, C)
        self.labels = labels      # (H, W)
        self.mask = mask          # (H, W)
        self.tile_size = tile_size
        self.n_samples = n_samples
        self.H, self.W = mask.shape

        # 采样中心点 (只在有效像素中)
        valid_ys, valid_xs = np.where(mask)
        indices = np.random.RandomState(42).choice(
            len(valid_ys), min(n_samples, len(valid_ys)), replace=False
        )
        self.centers = list(zip(valid_ys[indices], valid_xs[indices]))

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        cy, cx = self.centers[idx]
        hs = self.tile_size // 2
        y1, y2 = max(0, cy - hs), min(self.H, cy + hs)
        x1, x2 = max(0, cx - hs), min(self.W, cx + hs)

        x = self.features[y1:y2, x1:x2, :]  # (h, w, C)
        y = self.labels[y1:y2, x1:x2]        # (h, w)
        m = self.mask[y1:y2, x1:x2]          # (h, w)

        # Padding 到 tile_size
        if x.shape[0] < self.tile_size or x.shape[1] < self.tile_size:
            pad_h = max(0, self.tile_size - x.shape[0])
            pad_w = max(0, self.tile_size - x.shape[1])
            x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            y = np.pad(y, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
            m = np.pad(m, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        x = torch.from_numpy(x).float().permute(2, 0, 1)  # (C, H, W)
        y = torch.from_numpy(y).float().unsqueeze(0)       # (1, H, W)
        m = torch.from_numpy(m).float().unsqueeze(0)       # (1, H, W)
        return x, y, m


# ── 训练 ──
def train():
    print(f"设备: {DEVICE}")
    data_dir = Path(__file__).resolve().parent

    # 加载数据
    print("加载数据...")
    features = np.load(data_dir / "feature_stack.npy")   # (H, W, 18)
    labels = np.load(data_dir / "labels.npy")              # (H, W)
    mask = np.load(data_dir / "valid_mask.npy")            # (H, W)
    print(f"  特征: {features.shape}, 标签: {labels.shape}")

    # 归一化
    f_mean = features.mean(axis=(0, 1), keepdims=True)
    f_std = features.std(axis=(0, 1), keepdims=True) + 1e-8
    features = (features - f_mean) / f_std
    np.save(data_dir / "norm_stats.npy", {"mean": f_mean, "std": f_std})

    # 划分训练/验证集
    n_train = int(MAX_TRAIN_SAMPLES * TRAIN_RATIO)
    n_val = MAX_TRAIN_SAMPLES - n_train

    print(f"  训练样本: {n_train}, 验证样本: {n_val}")
    train_ds = TileDataset(features, labels, mask, TILE_SIZE, n_train)
    val_ds = TileDataset(features, labels, mask, TILE_SIZE, n_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型
    model = UNet(in_channels=18, features=[32, 64, 128, 256]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params:,} (~{n_params/1e6:.1f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.SmoothL1Loss(beta=0.1)

    best_val_loss = float("inf")
    print(f"\n开始训练 ({EPOCHS} epochs)...")
    print(f"  {'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'LR':<10} {'Time':<8}")
    print(f"  {'─'*50}")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # 训练
        model.train()
        train_loss = 0.0
        for x, y, m in train_loader:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred * m, y * m) / (m.mean() + 1e-8)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                pred = model(x)
                loss = criterion(pred * m, y * m) / (m.mean() + 1e-8)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"  {epoch+1:<8} {train_loss:<12.6f} {val_loss:<12.6f} {optimizer.param_groups[0]['lr']:<10.2e} {elapsed:<8.1f}s")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), data_dir / "cnn_cost_model.pth")
            print(f"    → 保存最佳模型 (val_loss={best_val_loss:.6f})")

    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存: {data_dir / 'cnn_cost_model.pth'}")


if __name__ == "__main__":
    train()
