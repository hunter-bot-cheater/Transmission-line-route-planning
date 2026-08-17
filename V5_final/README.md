# 输电线路智能路径规划系统

v1: 随机森林 + A* | v2: 严格质量门控 | **v3: CNN深度学习 + 神经路径规划(无A*)**

**v3核心流程**: 数据获取 → 多尺度预处理 → CostUNet成本建模 → Bellman值传播 → 梯度路径提取 → PathRefiner精炼 → 质量门控 → 对比可视化

---

## 版本对比

| 维度 | v1 | v2 | **v3** |
|------|-----|-----|-----|
| **成本模型** | 随机森林 | 随机森林 | **CostUNet CNN** |
| **路径搜索** | A* | A* (优化) | **Bellman值传播 + 梯度追踪** |
| **特征维度** | 17维 | 18维 | **26维 (多尺度+扩展约束)** |
| **数据范围** | 台湾 | 台湾 | **9省份** |
| **质量门控** | 无 | 7项 | **7项** |
| **深度学习框架** | — | — | **PyTorch** |
| **约束层** | 3项 | 5项 | **8项 (+断裂带/覆冰/雷击)** |

---

## 目录结构

```
D:\大创\
├── README.md                         # 本文件
├── requirements_extra.txt            # Python额外依赖
├── .gitignore
│
├── shared/                           # 公共模块 (v1/v2/v3共享)
│   ├── __init__.py
│   └── data_acquisition.py           # OSM下载、DEM加载、输电线过滤、风险代理层
│
├── data/                             # 共享数据 (自动生成)
│   ├── downloaded/                   #   OSM缓存 (.pkl, 7天有效期)
│   ├── processed/                    #   处理后栅格
│   └── models/                       #   训练好的模型及标准化器
│       └── v3_dl/                    #   v3 PyTorch模型
│
├── v1_20260525/                      # v1: 宽松约束版本 (历史基线)
│   ├── config.py
│   ├── main.py
│   ├── validate.py / validate_multi.py
│   └── src/
│
├── v2_20260525/                      # v2: 严格质量门控版本
│   ├── config.py
│   ├── validate_v2.py
│   ├── CHANGELOG.md
│   └── src/
│
└── v3_20260525/                      # v3: 深度学习全面重构 ★
    ├── config.py                     #   全局配置 (26维特征, 9省份, DL超参)
    ├── main.py                       #   单线路入口
    ├── validate_v3.py                #   10条线路对比验证
    ├── CHANGELOG.md                  #   变更记录
    ├── README.md                     #   v3文档
    ├── src/
    │   ├── __init__.py
    │   ├── dl_models.py              #   [新] CostUNet + ValuePropNet + PathRefiner
    │   ├── data_acquisition.py       #   多省份 + 合成数据 + 扩展风险层
    │   ├── preprocessing.py          #   多尺度地形因子 + 归一化
    │   ├── cost_model.py             #   CNN训练 + 伪标签
    │   ├── path_planning.py          #   神经值传播 + 梯度路径提取 (无A*)
    │   └── output.py                 #   对比可视化 + 汇总
    ├── docs/
    │   └── CODING_STANDARDS.md
    ├── tests/
    │   └── test_pipeline.py
    ├── run_comparison.py             #   快速对比验证 (无需训练)
    ├── generate_province_data.py     #   多省份合成数据生成
    └── output/
        └── comparisons/              #   10条线路对比输出

data/                                    # 共享数据
├── downloaded/                          #   OSM缓存 (.pkl)
├── processed/                           #   处理后栅格
├── models/                              #   训练好的模型
│   └── v3_dl/                           #   v3 PyTorch模型
└── province_data/                       #   8省份合成数据 (DEM + 输电线SHP)
```

---

## v1 vs v2 对比

| 维度 | v1 | v2 |
|------|-----|-----|
| **约束策略** | 宽松阈值, 单一硬约束 | 收紧阈值 + 7项质量门控 |
| **水域处理** | 50m硬约束缓冲 | 不纳入硬约束, 质量门控检测 |
| **验证规模** | 6条线路 | 10条标准线路 |
| **通过标准** | 路径生成即通过 | 7项质量审查全部通过 |
| **输出组织** | 单一输出目录 | 每条线路独立子目录 |
| **路径平滑** | B样条 | 线性插值 + 滑动平均 |
| **高程检查** | 无 | 单调段高程变化 ≤ 2500m |
| **曲率检查** | 无 | 最大转角 ≤ 50° |
| **水域检查** | 无 | ≤25%采样点距水 < 30m |
| **成本异常** | 无 | 路径均值/全图中位数 ≤ 3.0 |

---

## 快速开始

### v3 (最新 ★ 推荐, 准确度超v2)

```bash
cd D:\大创\v3_20260525
python main.py                              # 单线路: 台湾, 核三厂→台北
python main.py --region sichuan             # 切换至四川
python generate_province_data.py            # 生成8省份合成数据
python prepare_national_dataset.py          # 构建全国训练数据集 (train/val split)
python train_national_model.py              # 训练CostUNet (GPU推荐)
python test_trained_model.py                # 10条线路测试 + v2对比
python run_comparison.py                    # 快速对比验证 (启发式, 无需训练)
```

### v2

```bash
cd D:\大创\v2_20260525
python validate_v2.py          # 验证全部10条标准线路
```

### v1

```bash
cd D:\大创\v1_20260525
python main.py                 # 默认: 核三厂(屏东) → 台北
python main.py --start-lat 22.0 --start-lon 120.5 --end-lat 25.0 --end-lon 121.5
```

---

## 数据管线

### v3 (深度学习)

```
                ┌─────────────────────────────────┐
                │  data_acquisition.py            │
                │  DEM + OSM(9类) + 输电线          │
                │  风险代理(5层) + 合成数据          │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  preprocessing.py               │
                │  多尺度地形(13维) + 栅格对齐(90m)  │
                │  硬约束(8项) + 软约束 + 归一化     │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  cost_model.py (★CNN)           │
                │  26维特征堆叠 + 伪标签(12项因素)   │
                │  CostUNet训练(组合损失+早停)       │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  path_planning.py (★神经)        │
                │  Bellman值传播 + 梯度路径提取      │
                │  PathRefiner精炼 (无A*)          │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  output.py                      │
                │  SHP/GeoJSON + 对比可视化 + 汇总   │
                └─────────────────────────────────┘
```

### v2 (传统)
                │  shared/data_acquisition.py     │
                │  DEM + OSM + 输电线 + 风险代理    │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  preprocessing.py               │
                │  地形因子(7层) + 栅格对齐(90m)     │
                │  硬约束掩膜 + 软约束掩膜           │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  cost_model.py                  │
                │  17维特征堆叠 + 伪标签生成         │
                │  随机森林训练(R²>0.97)            │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  path_planning.py               │
                │  成本融合 + A*搜索 + 路径平滑      │
                │  + 7项质量门控(v2)               │
                └──────────────┬──────────────────┘
                               │
                ┌──────────────▼──────────────────┐
                │  output.py                      │
                │  SHP / GeoJSON / 统计 / 可视化    │
                └─────────────────────────────────┘
```

---

## 输入数据

| 数据 | 格式 | 路径 | 说明 |
|------|------|------|------|
| DEM | GeoTIFF 30m | `D:\地形数据\台湾省_DEM_30m分辨率_SRTM数据.tif` | SRTM 1-arcsec |
| 输电线 | Shapefile | `D:\输电线数据\示例数据-中国输电线路矢量.shp` | 1,795条 |
| OSM | Overpass API → pickle缓存 | `data/downloaded/*.pkl` | 自动下载, 7天缓存 |

---

## 特征体系 (17维)

| 维度 | 特征 | 类型 | 来源 |
|------|------|------|------|
| 0 | elevation | 连续 | DEM |
| 1 | slope | 连续 | 地形因子 |
| 2-3 | aspect_cos/sin | 连续 | 地形因子 |
| 4 | tri | 连续 | 地形粗糙度指数 |
| 5 | tpi | 连续 | 地形位置指数 |
| 6-7 | profile/plan_curvature | 连续 | 曲率 |
| 8 | roughness | 连续 | 局部标准差 |
| 9 | dist_road | 连续 | OSM道路EDT |
| 10 | dist_water | 连续 | OSM水域EDT |
| 11 | dist_existing_line | 连续 | 现有线路EDT |
| 12 | landuse_code | 离散 | OSM土地利用 |
| 13 | building_density | 连续 | OSM建筑密度 |
| 14 | typhoon_risk | 连续 | 台风暴露度代理 |
| 15 | seismic_risk | 连续 | 地震风险代理 |
| 16 | landslide_risk | 连续 | 滑坡敏感性代理 |

---

## 质量门控项 (v2, 7项)

1. **坡度**: 最大坡度 ≤ 45°
2. **水域**: ≤25%采样点距水域 < 30m (台湾水系密集)
3. **保护区**: 0点落入硬约束区
4. **曲率**: 相邻三点最大转角 ≤ 50°
5. **弯曲度**: 路径长度/直线距离 ≤ 3.0
6. **高程连贯性**: 单调段高程变化 ≤ 2500m (50m反转重置)
7. **成本异常**: 路径均值/全图中位数 ≤ 3.0

---

## 依赖环境

**核心库** (Python 3.13):
```
geopandas  1.1     # 矢量数据处理
rasterio   1.5     # 栅格读写与重投影
scikit-learn 1.7   # 随机森林
matplotlib 3.10    # 可视化
shapely    2.1     # 几何计算
scipy      1.16    # 滤波/插值/空间分析
numpy      2.3     # 数组计算
```

**额外依赖** (见 `requirements_extra.txt`):
```
folium     # 交互式地图
contextily # 在线底图
openpyxl   # Excel输出
```

---

## 10条标准测试线路 (v2)

| ID | 名称 | 电压 | 长度 | 区域 |
|----|------|------|------|------|
| case_01 | 屏东恒春线 | 345kV | 57km | 南台湾 |
| case_02 | 嘉义台中纵贯线 | 345kV | 110km | 中台湾 |
| case_03 | 新竹桃园线 | 161kV | 74km | 西北台湾 |
| case_04 | 台东花东纵谷线 | 161kV | 71km | 东台湾 |
| case_05 | 台中山线 | 345kV | 53km | 中台湾山区 |
| case_06 | 台中苗栗线 | 161kV | 56km | 中西部 |
| case_07 | 南台湾联络线 | 345kV | 32km | 南台湾 |
| case_08 | 中部横贯线 | 345kV | 42km | 中部山区 |
| case_09 | 北部沿海线 | 345kV | 34km | 北台湾 |
| case_10 | 东部联络线 | 161kV | 48km | 东台湾 |

---

## 技术参数

| 参数 | 值 |
|------|-----|
| 分析分辨率 | 90m (2401 × 4201) |
| DEM原始分辨率 | 30m SRTM |
| 坐标系统 | WGS84 (EPSG:4326) / UTM 51N (EPSG:32651) |
| A*邻域 | 8邻域 Moore |
| 启发式 | Octile距离 |
| OSM数据量 | 道路71k / 水域107k / 建筑566k / 土地利用128k |
| 台湾输电线 | 1,795条 |

---

*项目路径: D:\大创\ | 版本: v1/v2/v3.20260525 | v3: PyTorch深度学习 | Python 3.9+*
