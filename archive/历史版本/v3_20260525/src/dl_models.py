"""
v3_dl: 深度学习模型架构定义
版本: v3.20260525
作者: path_planning_team
变更记录:
  - v3.20260525: 重建 — MultiScaleValuePropNet + DifferentiableMinPool, 自监督Bellman损失
  - v2.20260525: CostUNet, ValuePropNet, PathRefiner, NeuralHeuristic
依赖: torch, v3/config
说明:
  - CostUNet: CNN encoder-decoder, 输入多波段特征, 输出逐像元建设成本
  - MultiScaleValuePropNet: 多尺度可微值传播, 替代Dijkstra, 纯神经网络
  - PathRefiner: MLP+Conv1D+Attention路径精炼器
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

import config as cfg


# ============================================================
# 基础模块
# ============================================================
class DoubleConv(nn.Module):
    """双卷积 + BatchNorm + ReLU"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.conv(x))


class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


# ============================================================
# CostUNet: CNN编码-解码成本预测网络
# ============================================================
class CostUNet(nn.Module):
    """
    U-Net架构成本预测模型。
    输入: (B, N_FEATURES, H, W) 多波段特征堆叠
    输出: (B, 1, H, W) 逐像元建设成本 (0-1归一化)
    """

    def __init__(
        self,
        n_features: int = cfg.N_FEATURES,
        encoder_channels: List[int] = None,
        decoder_channels: List[int] = None,
        bottleneck: int = None,
        dropout: float = None,
    ):
        super().__init__()
        enc_ch = encoder_channels or cfg.UNET_ENCODER_CHANNELS
        dec_ch = decoder_channels or cfg.UNET_DECODER_CHANNELS
        bn = bottleneck or cfg.UNET_BOTTLENECK
        dr = dropout if dropout is not None else cfg.UNET_DROPOUT

        self.encoder_blocks = nn.ModuleList()
        in_ch = n_features
        for out_ch in enc_ch:
            self.encoder_blocks.append(DoubleConv(in_ch, out_ch, dr))
            in_ch = out_ch

        self.bottleneck = nn.Sequential(
            DoubleConv(enc_ch[-1], bn, dr),
            ResidualBlock(bn, dr),
            ResidualBlock(bn, dr),
        )

        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i, out_ch in enumerate(dec_ch):
            skip_ch = enc_ch[-(i + 1)] if i < len(enc_ch) else enc_ch[-1]
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                DoubleConv(out_ch + skip_ch, out_ch, dr)
            )
            in_ch = out_ch

        self.final_conv = nn.Sequential(
            nn.Conv2d(dec_ch[-1], 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[2:]
        skips = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for i, (up, dec) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            x = up(x)
            skip = skips[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.final_conv(x)
        if x.shape[2:] != input_shape:
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=True)
        return x

    def predict_cost_surface(
        self, feature_stack: np.ndarray, batch_size: int = 4
    ) -> np.ndarray:
        """对大型特征堆叠进行分块预测"""
        self.eval()
        H, W, C = feature_stack.shape
        x = torch.from_numpy(feature_stack).permute(2, 0, 1).unsqueeze(0).to(cfg.DEVICE)
        x = x.to(cfg.TORCH_DTYPE)

        if H <= 512 and W <= 512:
            with torch.no_grad():
                pred = self.forward(x)
            return pred.squeeze().cpu().numpy()

        patch_size = 256
        overlap = 32
        result = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        for r in range(0, H, patch_size - overlap):
            for c in range(0, W, patch_size - overlap):
                r_end = min(r + patch_size, H)
                c_end = min(c + patch_size, W)
                r_start = max(0, r_end - patch_size)
                c_start = max(0, c_end - patch_size)

                patch = x[:, :, r_start:r_end, c_start:c_end]
                with torch.no_grad():
                    pred = self.forward(patch)
                pred_np = pred.squeeze().cpu().numpy()

                result[r_start:r_end, c_start:c_end] += pred_np
                weight[r_start:r_end, c_start:c_end] += 1.0

        result = result / np.clip(weight, 1e-6, None)
        return result


# ============================================================
# 损失函数
# ============================================================
class CostPredictionLoss(nn.Module):
    """成本预测组合损失: MSE + 梯度 + 硬约束惩罚"""

    def __init__(self, mse_weight=1.0, grad_weight=0.3, constraint_weight=2.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight
        self.constraint_weight = constraint_weight

    def forward(self, pred, target, hard_mask=None):
        valid = ~torch.isnan(target)
        mse = F.mse_loss(pred[valid], target[valid]) if valid.any() else 0.0

        grad_loss = 0.0
        if self.grad_weight > 0:
            pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
            target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
            pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
            grad_loss = F.l1_loss(pred_dy, target_dy) + F.l1_loss(pred_dx, target_dx)

        constraint_loss = 0.0
        if self.constraint_weight > 0 and hard_mask is not None:
            blocked = (hard_mask == 0)
            if blocked.any():
                constraint_loss = pred[blocked].mean()

        total = (self.mse_weight * mse +
                 self.grad_weight * grad_loss +
                 self.constraint_weight * constraint_loss)
        return total, {"mse": float(mse), "grad": float(grad_loss), "constraint": float(constraint_loss)}


# ============================================================
# MultiScaleValuePropNet: 多尺度可微值传播网络
# ============================================================
class DifferentiableMinPool(nn.Module):
    """
    可微分8邻域min-pooling。
    使用softmin近似, 使Bellman更新完全可微。
    """

    def __init__(self, temperature: float = None):
        super().__init__()
        self.temperature = temperature or cfg.SOFTMIN_TEMPERATURE

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        B, C, H, W = value.shape
        padded = F.pad(value, [1, 1, 1, 1], mode='replicate')
        patches = F.unfold(padded, kernel_size=3, stride=1)  # (B, 9, H*W)
        patches = patches.view(B, 9, H, W)
        weights = F.softmin(patches / self.temperature, dim=1)
        pooled = (weights * patches).sum(dim=1, keepdim=True)
        return pooled


class ValuePropStage(nn.Module):
    """
    单级值传播: K次可微Bellman迭代。
    V_{k+1}(s) = cost(s) + gamma * min_{s'} V_k(s')
    """

    def __init__(self, n_extra_channels: int = 5, hidden_dim: int = None, k: int = 20):
        super().__init__()
        self.k = k
        hd = hidden_dim or cfg.VALUE_PROP_HIDDEN_DIM
        in_ch = 1 + n_extra_channels

        self.state_encoder = nn.Sequential(
            nn.Conv2d(in_ch, hd, 3, padding=1, bias=False),
            nn.BatchNorm2d(hd),
            nn.ReLU(inplace=True),
            nn.Conv2d(hd, hd, 3, padding=1, bias=False),
            nn.BatchNorm2d(hd),
            nn.ReLU(inplace=True),
        )

        self.transition = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hd + 1, hd, 3, padding=1, bias=False),
                nn.BatchNorm2d(hd),
                nn.ReLU(inplace=True),
                ResidualBlock(hd),
            )
            for _ in range(3)
        ])

        self.gamma = nn.Parameter(torch.tensor(cfg.VALUE_PROP_GAMMA))
        self.min_pool = DifferentiableMinPool()

        self.value_head = nn.Sequential(
            nn.Conv2d(hd, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Softplus(),
        )

    def forward(self, cost_surface: torch.Tensor, goal_mask: torch.Tensor,
                extra_features: Optional[torch.Tensor] = None,
                init_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, _, H, W = cost_surface.shape
        if extra_features is None:
            extra_features = torch.zeros(B, 5, H, W, device=cost_surface.device)

        state = torch.cat([cost_surface, goal_mask, extra_features], dim=1)
        state_enc = self.state_encoder(state)

        if init_value is not None:
            value = init_value
        else:
            value = torch.zeros(B, 1, H, W, device=cost_surface.device)

        for _ in range(self.k):
            min_neighbor = self.min_pool(value)
            target_value = cost_surface + self.gamma * min_neighbor
            trans_input = torch.cat([state_enc, value], dim=1)
            for layer in self.transition:
                trans_input = layer(trans_input)
            delta = self.value_head(trans_input)
            value = value + 0.1 * (target_value - value) + 0.05 * delta

        return value


class MultiScaleValuePropNet(nn.Module):
    """
    多尺度值传播网络。
    粗尺度(256×256, K=20) → 细尺度(1024×1024, K=10)。
    全程可微, 无图搜索。
    """

    def __init__(self, n_features: int = cfg.N_FEATURES, n_extra: int = 5):
        super().__init__()
        self.coarse_size = cfg.COARSE_SIZE
        self.fine_size = cfg.FINE_SIZE
        self.coarse_stage = ValuePropStage(n_extra, k=cfg.VALUE_PROP_K_COARSE)
        self.fine_stage = ValuePropStage(n_extra, k=cfg.VALUE_PROP_K_FINE)
        self.feature_projector = nn.Sequential(
            nn.Conv2d(n_features, n_extra, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, cost_surface: torch.Tensor, features: torch.Tensor,
                goal_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = cost_surface.shape[0]

        # Project features to extra channels
        extra_full = self.feature_projector(features)

        # Stage 1: Coarse
        cost_coarse = F.interpolate(cost_surface, self.coarse_size, mode='bilinear', align_corners=False)
        goal_coarse = F.interpolate(goal_mask, self.coarse_size, mode='nearest')
        extra_coarse = F.interpolate(extra_full, self.coarse_size, mode='bilinear', align_corners=False)
        v_coarse = self.coarse_stage(cost_coarse, goal_coarse, extra_coarse)

        # Stage 2: Fine
        cost_fine = F.interpolate(cost_surface, self.fine_size, mode='bilinear', align_corners=False)
        goal_fine = F.interpolate(goal_mask, self.fine_size, mode='nearest')
        extra_fine = F.interpolate(extra_full, self.fine_size, mode='bilinear', align_corners=False)
        v_fine_init = F.interpolate(v_coarse, self.fine_size, mode='bilinear', align_corners=False)
        v_fine = self.fine_stage(cost_fine, goal_fine, extra_fine, v_fine_init)

        return v_fine, v_coarse


class SelfSupervisedBellmanLoss(nn.Module):
    """
    自监督Bellman残差损失。
    L = MSE(V(s) - [cost(s) + gamma * softmin(V(s'))])^2
    无需Dijkstra标签 — 纯自监督。
    """

    def __init__(self, gamma: float = None):
        super().__init__()
        self.gamma = gamma or cfg.VALUE_PROP_GAMMA
        self.min_pool = DifferentiableMinPool()

    def forward(self, value: torch.Tensor, cost_surface: torch.Tensor,
                goal_mask: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        min_neighbor = self.min_pool(value)
        target = cost_surface + self.gamma * min_neighbor

        diff = (value - target) ** 2

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).float()
            diff = diff * mask
            n_valid = mask.sum() + 1e-8
            loss = diff.sum() / n_valid
        else:
            loss = diff.mean()

        # Goal consistency: V(goal) should equal cost(goal)
        goal_consistency = ((value * goal_mask) - (cost_surface * goal_mask)).abs().mean()

        return loss + 0.1 * goal_consistency


# ============================================================
# PathRefiner: 神经路径精炼器
# ============================================================
class PathRefiner(nn.Module):
    """
    坐标精炼网络。输入路径点序列, 输出精炼后的坐标。
    MLP + 1D Conv + MultiheadAttention。
    """

    def __init__(self, hidden_dims: List[int] = None, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        hd = hidden_dims or cfg.REFINER_HIDDEN
        self.input_proj = nn.Linear(2, hd[0])
        self.conv_layers = nn.ModuleList()
        in_ch = hd[0]
        for out_ch in [hd[0], hd[0] // 2]:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv1d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.attention = nn.MultiheadAttention(in_ch, n_heads, dropout=dropout, batch_first=True)
        self.output_proj = nn.Sequential(
            nn.Linear(in_ch, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )
        self.delta_scale = 0.1

    def forward(self, path_coords: torch.Tensor) -> torch.Tensor:
        B, L, _ = path_coords.shape
        x = self.input_proj(path_coords)  # (B, L, hidden)
        for conv in self.conv_layers:
            x_t = x.transpose(1, 2)  # (B, hidden, L)
            x_t = conv(x_t)
            x = x_t.transpose(1, 2)  # (B, L, hidden)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        delta = self.output_proj(x)
        return path_coords + delta * self.delta_scale

    def refine_path(self, coords: np.ndarray, n_iterations: int = None) -> np.ndarray:
        """批量路径精炼推断"""
        n_iter = n_iterations or cfg.REFINER_N_ITERATIONS
        self.eval()
        path_t = torch.from_numpy(coords).float().unsqueeze(0).to(cfg.DEVICE)
        with torch.no_grad():
            for _ in range(n_iter):
                path_t = self.forward(path_t)
        return path_t.squeeze(0).cpu().numpy()


# ============================================================
# 模型保存/加载
# ============================================================
def save_model(model: nn.Module, name: str):
    """保存PyTorch模型"""
    path = cfg.V3_MODELS_DIR / f"{name}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "class_name": model.__class__.__name__,
        "version": "v3.20260525",
    }, path)
    return path


def load_model(model: nn.Module, name: str) -> nn.Module:
    """加载PyTorch模型, 自动移至cfg.DEVICE。文件不存在时抛出FileNotFoundError。"""
    path = cfg.V3_MODELS_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    ckpt = torch.load(path, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(cfg.DEVICE)
    return model
