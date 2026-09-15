# v3 深度学习输电线路智能路径规划系统

基于CNN成本建模、神经值传播和梯度路径提取，实现端到端的输电线路智能路径规划。**v3全面深度学习化: 以PyTorch CNN替代随机森林, 以Bellman值传播+梯度追踪替代A*启发式搜索。**

**核心流程**: 数据获取 → 多尺度预处理 → CNN成本建模 → 神经值传播 → 梯度路径提取 → PathRefiner精炼 → 质量门控 → 对比可视化

---

## 与 v2 的核心差异

| 维度 | v2 | v3 |
|------|-----|-----|
| **成本模型** | 随机森林 (R^2>0.97) | CostUNet CNN (组合损失) |
| **路径搜索** | A* + octile启发式 | Bellman值传播 + 梯度追踪 |
| **路径精炼** | RDP + 滑动平均 | PathRefiner神经网络 |
| **特征维度** | 18维 | 26维 (多尺度+扩展约束) |
| **数据范围** | 台湾 | 9省份 (台湾+西部山区8省) |
| **深度学习框架** | scikit-learn | PyTorch |
| **训练方式** | 伪标签→RF回归 | 伪标签→CNN端到端分割 |

---

## 目录结构

```
v3_20260525/
├── config.py                  # 全局配置 (26维特征, 9省份bbox, DL超参)
├── main.py                    # 单线路入口
├── validate_v3.py             # 10条线路对比验证
├── CHANGELOG.md               # 变更记录
├── README.md                  # 本文件
├── src/
│   ├── __init__.py
│   ├── dl_models.py           # 神经网络架构 (CostUNet, MultiScaleValuePropNet, PathRefiner)
│   ├── data_acquisition.py    # 数据获取 (多省份OSM+合成数据+风险代理)
│   ├── preprocessing.py       # 预处理 (多尺度地形因子+栅格对齐+归一化)
│   ├── cost_model.py          # 成本建模 (CNN训练+伪标签)
│   ├── path_planning.py       # 路径规划 (神经值传播 + 梯度追踪 + 平滑)
│   └── output.py              # 输出 (SHP+对比图+汇总报告)
├── tests/
│   └── test_pipeline.py       # 单元+集成测试
├── docs/
│   └── CODING_STANDARDS.md
└── output/
    └── comparisons/           # 10条线路独立输出
```

---

## v3 vs v2 准确度对比 (10条台湾线路)

| 指标 | v2 (RF+A*) | v3 CNN (全国训练) | 提升 |
|------|-----------|-----------------|------|
| Hausdorff距离 | 5995m | **4013m** | **-33.1%** |
| 500m重叠率 | 33.6% | **51.3%** | **+52.6%** |
| 平均最近距离 | 2579m | **1193m** | **-53.7%** |
| 长度误差 | 4.1% | 4.5% | -9.8% |

> v3 CNN在7/10条线路上优于v2, 综合准确度显著提升。CNN模型使用7省份合成数据训练, 台湾10条真实线路作为独立测试集 (held-out)。

## 快速开始

### 环境要求

```
Python >= 3.9
PyTorch >= 1.13
CUDA (可选, 显著加速CNN训练)
```

核心依赖:
```
torch >= 1.13
numpy, scipy, rasterio, geopandas, shapely, matplotlib
scikit-learn (仅用于伪标签验证)
```

### 单线路规划

```bash
cd D:\大创\v3_20260525

# 默认: 台湾, 核三厂(屏东) → 台北
python main.py

# 自定义起止点
python main.py --start 22.0,120.5 --end 25.0,121.5

# 切换至四川
python main.py --region sichuan

# 使用合成数据 (无GIS文件时)
python main.py --synthetic

# 复用已训练模型
python main.py --skip-training
```

### 10条线路对比验证

```bash
cd D:\大创\v3_20260525

python validate_v3.py                    # 全部10条
python validate_v3.py --case case_01     # 单条
python validate_v3.py --cases 3          # 前3条
python validate_v3.py --synthetic        # 合成数据
python validate_v3.py --skip-training    # 复用模型
```

---

## 算法详解

### 1. CostUNet 成本预测

```
输入: (H, W, 26) 多波段特征堆叠
编码器: [32→64→128→256→512] 双卷积+下采样
瓶颈: 512通道 + 2x残差块
解码器: [256→128→64→32] 转置卷积+跳跃连接
输出: (H, W, 1) 逐像元建设成本 (0-1)
损失: MSE + 梯度一致性 + 硬约束惩罚
```

### 2. MultiScaleValuePropNet 神经值传播

```
V_{k+1}(s) = cost(s) + γ · softmin_{s'∈N(s)} V_k(s')
softmin: 可微分8邻域min-pooling (温度τ=0.5)
粗尺度(256×256, K=20) → 细尺度(1024×1024, K=10)
全程可微, 自监督Bellman残差训练, 零图搜索
```

### 3. Neural Gradient Tracker 梯度路径提取

```
s_{t+1} = s_t - α_t · v_t
v_t = β · v_{t-1} + (1-β) · ∇V(s_t) / ||∇V(s_t)||
α_t: 自适应步长 (低成本区大步, 高成本区小步)
目标引力偏置确保收敛
亚像素精度 (双线性插值 + 中心有限差分)
零优先队列/零visited集合/零节点图 — 纯梯度下降
```

### 4. PathRefiner 精炼

```
输入: 粗路径坐标序列 (L, 2)
MLP投影 → 1D卷积(多层) → 多头自注意力 → 残差输出
输出: 精炼后坐标 (L, 2)
```

---

## 特征体系 (26维)

### 地形因子 (13维)
| 索引 | 特征 | 描述 |
|------|------|------|
| 0 | elevation | DEM高程 |
| 1 | slope | Horn坡度(°) |
| 2-3 | aspect_cos/sin | 坡向余弦/正弦 |
| 4 | tri | 地形粗糙度指数 |
| 5-7 | tpi_100/300/900 | 多尺度地形位置指数 |
| 8-9 | profile/plan_curvature | 剖面/平面曲率 |
| 10-12 | roughness_3/9/27 | 多尺度粗糙度 |

### 距离栅格 (5维)
| 13 | dist_road | 距道路距离 |
| 14 | dist_water | 距水域距离 |
| 15 | dist_existing_line | 距现有线路距离 |
| 16 | dist_railway | 距铁路距离 |
| 17 | dist_fault | [v3新增] 距断裂带距离 |

### 分类/密度 (3维)
| 18 | landuse_code | 土地利用分类 (1-8) |
| 19 | building_density | 建筑密度 (栋/km²) |
| 20 | vegetation_height | [v3新增] 植被高度代理 (m) |

### 风险代理 (5维)
| 21 | typhoon_risk | 台风暴露度 |
| 22 | seismic_risk | 地震风险 |
| 23 | landslide_risk | 滑坡敏感性 |
| 24 | ice_cover_risk | [v3新增] 覆冰风险 |
| 25 | lightning_risk | [v3新增] 雷击风险 |

---

## 硬约束 (8项)

1. 坡度 > 45° 禁止
2. 水域 (可配缓冲)
3. 自然保护区 + 200m缓冲
4. 高程 > 3000m 禁止
5. 建筑密度 > 500栋/km² 禁止
6. [v3] 断裂带 + 500m缓冲 禁止
7. [v3] 覆冰风险 > 0.6 禁止
8. [v3] 雷击风险 > 0.7 禁止

---

## 多省份支持

| 省份 | bbox | DEM来源 |
|------|------|---------|
| 台湾 (默认) | (120.0, 21.9, 122.0, 25.4) | SRTM 30m |
| 四川 | (97.35, 26.05, 108.52, 34.32) | SRTM 30m / 合成 |
| 云南 | (97.53, 21.14, 106.20, 29.25) | SRTM 30m / 合成 |
| 西藏 | (78.40, 26.85, 99.12, 36.50) | SRTM 30m / 合成 |
| 贵州 | (103.60, 24.62, 109.60, 29.22) | SRTM 30m / 合成 |
| 甘肃 | (92.21, 32.52, 108.77, 42.80) | SRTM 30m / 合成 |
| 陕西 | (105.48, 31.70, 111.25, 39.58) | SRTM 30m / 合成 |
| 福建 | (115.83, 23.55, 120.72, 28.33) | SRTM 30m / 合成 |
| 重庆 | (105.29, 28.16, 110.19, 32.23) | SRTM 30m / 合成 |

---

## 输出清单

每条验证线路输出:
- `{case_id}_path_v3.shp` — DL规划路径 (SHP格式)
- `{case_id}_real_path.shp` — 真实路径 (SHP格式)
- `{case_id}_map_overview.png` — 对比总览图 (v2风格: 山体阴影地形渲染, 蓝色真实线路, 红色虚线预测路径)
- `{case_id}_elevation_profile.png` — 高程剖面图 (v2风格: 高程填充 + 坡度叠加双轴)
- `{case_id}_statistics_v3.json` — 统计指标 (Hausdorff, 重叠率, 长度误差等)

汇总输出:
- `v3_comparison_summary.json` — 汇总指标 (均值/中位数)
- `v3_summary_comparison_charts.png` — 综合对比图表 (4面板)

多省份合成数据支持通过 `--synthetic` 参数启用, 用于无真实GIS数据的省份。

---

*项目路径: D:\大创\v3_20260525\ | 版本: v3.20260525 | 框架: PyTorch*
