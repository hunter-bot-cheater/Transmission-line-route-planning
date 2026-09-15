"""
CNN 全图推理 — 生成 CNN 成本表面
版本: v3.20260710
"""
import sys
from pathlib import Path
import numpy as np
import torch
from train_cnn import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 128
OVERLAP = 32  # 窗口重叠避免边缘伪影


def predict_full_image(model, features, norm_stats, hard_mask=None):
    """
    滑动窗口全图推理

    Args:
        features: (H, W, 18) 特征栈
        norm_stats: {"mean": ..., "std": ...}
        hard_mask: (H, W) 硬约束掩膜, 1=可行

    Returns:
        cost_surface: (H, W) 预测成本表面, 硬约束区=inf
    """
    model.eval()
    H, W, C = features.shape
    f_mean = norm_stats["mean"]
    f_std = norm_stats["std"]

    # 归一化
    f_norm = (features - f_mean) / f_std
    f_tensor = torch.from_numpy(f_norm).float().to(DEVICE)  # (H, W, C)

    # 输出缓存
    output = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    stride = TILE_SIZE - OVERLAP

    with torch.no_grad():
        for y in range(0, H - OVERLAP, stride):
            for x in range(0, W - OVERLAP, stride):
                y1, y2 = y, min(y + TILE_SIZE, H)
                x1, x2 = x, min(x + TILE_SIZE, W)

                tile = f_tensor[y1:y2, x1:x2, :]  # (h, w, C)
                h, w_c = tile.shape[0], tile.shape[1]

                # Padding
                pad_h = TILE_SIZE - h
                pad_w = TILE_SIZE - w_c
                if pad_h > 0 or pad_w > 0:
                    tile = torch.nn.functional.pad(
                        tile.permute(2, 0, 1), (0, pad_w, 0, pad_h), mode="reflect"
                    ).permute(1, 2, 0)

                tile = tile.permute(2, 0, 1).unsqueeze(0)  # (1, C, T, T)
                pred = model(tile).squeeze().cpu().numpy()  # (T, T)

                # 取有效区域
                pred = pred[:h, :w_c]
                output[y1:y2, x1:x2] += pred
                count[y1:y2, x1:x2] += 1.0

    # 平均重叠区域
    output = output / np.maximum(count, 1.0)
    output = np.clip(output, 0, None)

    # 硬约束区域标记为 inf
    if hard_mask is not None:
        output = np.where(hard_mask == 1, output, np.inf)

    valid = output < np.inf
    if valid.any():
        print(f"  CNN成本范围: [{output[valid].min():.4f}, {output[valid].max():.4f}]")

    return output.astype(np.float32)


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent
    print(f"设备: {DEVICE}")

    # 加载数据
    print("加载数据...")
    features = np.load(data_dir / "feature_stack.npy")
    mask = np.load(data_dir / "valid_mask.npy")
    norm_stats = np.load(data_dir / "norm_stats.npy", allow_pickle=True).item()
    print(f"  特征: {features.shape}")

    # 加载模型
    print("加载 CNN 模型...")
    model = UNet(in_channels=18, features=[32, 64, 128, 256]).to(DEVICE)
    model.load_state_dict(torch.load(data_dir / "cnn_cost_model.pth", map_location=DEVICE))
    print(f"  模型已加载 (最佳 val_loss)")

    # 全图推理
    print("全图推理...")
    import time
    t0 = time.time()
    cnn_cost = predict_full_image(model, features, norm_stats, hard_mask=mask)
    elapsed = time.time() - t0
    print(f"  推理完成, 耗时: {elapsed:.1f}s")

    # 保存
    out_path = data_dir / "cnn_cost_surface.npy"
    np.save(out_path, cnn_cost)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  已保存: {out_path} ({size_mb:.1f} MB)")
