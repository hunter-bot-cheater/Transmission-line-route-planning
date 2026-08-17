# 代码风格与接口规范文档 v3

**版本**: v3.20260525
**适用范围**: 台湾输电线路智能路径规划系统 v3 深度学习版
**基于**: v2 CODING_STANDARDS.md (完全兼容)

---

## v3 新增规范

### 深度学习模块规范

1. **模型类命名**: 使用功能+架构后缀 (CostUNet, ValuePropNet, PathRefiner)
2. **模型文件**: `.pt` (PyTorch checkpoint), 保存在 `data/models/v3_dl/`
3. **设备管理**: 统一使用 `cfg.DEVICE`, 禁止硬编码 `cuda` 或 `cpu`
4. **模型输入输出**: 所有Tensor形状在docstring中标注 (B, C, H, W)
5. **训练循环**: 必须包含早停(patience=15)和学习率调度(ReduceLROnPlateau)
6. **梯度裁剪**: 最大梯度范数1.0
7. **分块预测**: 大图(>512x512)自动分块预测

### 神经路径规划规范

1. **零传统搜索**: 禁止使用 A*, Dijkstra, RRT, 贪心图搜索, 或任何优先队列式算法
2. **值传播**: MultiScaleValuePropNet — 可微Bellman迭代, softmin 8邻域pooling, 两级(256→1024), 自监督训练
3. **路径提取**: 神经梯度追踪 — 连续梯度下降, 亚像素精度, 自适应步长, Polyak动量, 目标引力偏置
4. **精炼**: PathRefiner作为可选后处理, 或通过RDP+等距重采样+滑动平均平滑

### PyTorch代码规范

1. **模型定义**: 继承 `nn.Module`, 在 `__init__` 中定义所有层
2. **前向传播**: 仅包含计算逻辑, 无副作用
3. **预测方法**: 实现 `predict_*` 系列方法封装numpy转换和分块
4. **保存加载**: 使用 `save_model()` / `load_model()` 统一接口

---

## 接口契约 (继承v2)

### 模块接口

```python
def process(input_data: dict, params: dict, verbose: bool = False) -> dict:
    """
    Returns:
        dict: {"status": "success"|"failure", "data": {...}, "metadata": {...}}
    """
```

### v3 特殊接口

```python
# 成本模型预测
def predict_cost_surface(model, feature_stack, hard_mask) -> np.ndarray:
    """返回 (H, W) float32"""

# 神经路径规划
def neural_path_planning(cost_surface, hard_mask, transform, start_rc, goal_rc, ...) -> list:
    """返回 [(lon, lat), ...] 路径坐标列表"""

# 质量门控
def quality_gate(coords, aligned, hard_mask, final_cost, transform, straight_km) -> dict:
    """返回 {"passed": bool, "checks": {...}}"""
```

---

## 配置分层 (v3扩展)

```
config.py
├── 路径配置 (BASE_DIR, V3_DIR, ...)
├── 多省份边界 (PROVINCE_BBOX)
├── 坐标参考系 (WGS84, PROJECTED_CRS)
├── 栅格参数 (BASE_RESOLUTION=90m)
├── 地形因子参数 (多尺度窗口)
├── 硬约束参数 (8项)
├── 伪标签权重 (12项因素)
├── 深度学习参数
│   ├── CostUNet (编码器通道, dropout)
│   ├── 训练 (batch_size, lr, epochs)
│   └── 值传播 (k=20)
├── 特征波段 (26维)
└── 测试用例 (10条)
```

---

## 测试规范 (v3扩展)

### DL特定测试

- 模型输出形状验证
- 梯度流动检查
- 过拟合小批量测试 (single-batch overfit)
- 分块预测一致性

### 路径规划测试

- 梯度流动检查: 值函数梯度方向正确指向低值区
- 障碍物绕行: U形障碍测试 (梯度追踪自然绕行, 无需显式几何推理)
- 起止点不可达: 最近有效点回退机制
- 收敛性: 目标引力偏置确保有限步内到达

---

*完整的v2规范适用于所有非DL部分代码。*
