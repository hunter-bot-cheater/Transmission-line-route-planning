# CHANGELOG — v2.20260525

## 最终参数 (经3轮验证调整)

### 硬约束参数

| 参数 | v1 值 | v2 初始值 | v2 最终值 | 说明 |
|------|-------|-----------|-----------|------|
| MAX_SLOPE | 35° | 28° | 45° | 仅阻断极端陡坡; 质量门控独立检测 |
| SLOPE_QUALITY_THRESHOLD | — | 35° | 45° | 质量门控坡度阈值(独立于硬约束) |
| WATER_BUFFER | 50m | 100m | 0m | 水域不纳入硬约束(台湾水系密集, 阻断路径) |
| WATER_BUFFER_QUALITY | — | 100m | 30m | 质量门控水域距离阈值 |
| MAX_ELEVATION | 3500m | 3000m | 3000m | 不变 |
| PROTECTED_BUFFER | 0 | 200m | 200m | 不变 |
| BUILDING_DENSITY_LIMIT | 无 | 500 | 500 | 不变 |
| MAX_TURN_ANGLE | — | 15° | 50° | 栅格路径经平滑后转角 |
| MAX_SINUOSITY | — | 2.0 | 3.0 | 允许绕行水域 |
| MAX_CONTINUOUS_CLIMB | — | 1000m | 2500m | 单调段高程变化(算法已修复) |
| COST_ANOMALY_RATIO | — | 3.0 | 3.0 | 不变 |

### 算法修复 (path_planning.py)

- **水域质量审查**: 从"任意点距水域<阈值→不通过"改为"≤25%采样点违规→通过"(台湾106,898条水系密集)
- **高程连贯性**: 修复重叠窗口累积bug(原算法将300m爬升报告为3200m), 改为单调段起止点实际高程差
- **端点处理**: `ASTAR_STRICT_ENDPOINTS=False`, 自动搜索最近可行点(半径50像素)
- **启发式权重**: 1.2 → 1.0 (标准A*)
- **对角线成本**: 1.5 → 1.414 (标准sqrt(2))

### 软约束强化
- LABEL_WEIGHTS: dist_existing 0.45→0.65, slope 0.16→0.15
- PSEUDO_LABEL_PARAMS 新增 roughness 因子
- dist_existing_decay: 400m→250m, water_decay: 150m→200m

### 路径平滑
- B样条 → 线性插值 + 5点滑动平均平滑(150m窗口)
- RDP epsilon: 50m → 90m

## 验证结果 (v2.20260525 最终)

| 指标 | 数值 |
|------|------|
| 测试线路 | 10/10 生成路径 |
| 质量门控通过 | 10/10 (100%) |
| 平均 Hausdorff | 5,995m |
| 中位 Hausdorff | 4,020m |
| 平均 500m 重叠率 | 33.6% |
| 平均长度误差 | 4.1% |

## 兼容性说明
- v1 输出格式不兼容, 需通过 validate_v2.py 重新生成
- shared/data_acquisition.py 接口保持一致
