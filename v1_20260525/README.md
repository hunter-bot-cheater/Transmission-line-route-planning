<<<<<<< HEAD
# 台湾输电线路智能路径规划系统 — 完整说明文档

## 项目概述

本项目基于GIS空间分析、机器学习和启发式搜索算法，实现台湾地区输电线路的智能路径规划。系统接收起点和终点坐标，自动获取并处理多源地理数据，构建建设成本预测模型，最终输出最优路径的矢量文件和可视化结果。

**核心流程**: 数据获取 → 空间预处理 → 成本建模(随机森林) → A*路径搜索 → 输出与可视化

**运行方式**:
```bash
cd D:\大创
python main.py                                          # 默认: 核三厂(屏东) → 台北
python main.py --start-lat 22.0 --start-lon 120.5 --end-lat 25.0 --end-lon 121.5
```

---

## 一、目录结构总览

```
D:\大创\
├── main.py                      # [入口] 5阶段流程编排器
├── config.py                    # [配置] 全部路径、参数、常量定义
├── requirements_extra.txt       # [依赖] 额外Python包列表
│
├── src/                         # 源代码模块
│   ├── data_acquisition.py      # 模块1: 多源GIS数据获取
│   ├── preprocessing.py         # 模块2: 空间数据预处理
│   ├── cost_model.py            # 模块3: 随机森林成本建模
│   ├── path_planning.py         # 模块4: A*路径搜索与平滑
│   └── output.py                # 模块5: 结果导出与可视化
│
├── data/                        # 中间产物 (自动生成)
│   ├── downloaded/              #   OSM下载缓存 (7天有效期)
│   ├── processed/               #   处理后栅格预留
│   └── models/                  #   训练好的RF模型及标准化器
│
└── output/                      # 最终成果
    ├── optimal_path.shp         # 最优路径矢量 (ESRI Shapefile)
    ├── optimal_path.geojson     # 最优路径矢量 (GeoJSON)
    ├── start_end_points.shp     # 起止点矢量
    ├── cost_surface.tif         # 随机森林预测成本表面
    ├── final_cost_surface.tif   # 融合约束后的最终成本表面
    ├── constraint_mask.tif      # 硬约束二值掩膜 (0=禁止, 1=允许)
    ├── statistics.json          # 统计指标 (JSON)
    ├── statistics.xlsx          # 统计指标 (Excel)
    ├── map_overview.png         # 概览图 (300dpi, 双面板)
    ├── map_detail.png           # 走廊详图 (600dpi, 2km缓冲区)
    ├── elevation_profile.png    # 高程剖面图 (含坡度分布)
    └── interactive_map.html     # 交互式Leaflet地图
```

---

## 二、各文件详细说明

### 2.1 `main.py` — 流程编排器 (入口)

**作用**: 按顺序调用5个模块，串联完整数据处理管线

**机理**:
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
  获取      预处理     建模      搜索      输出
```

- 解析命令行参数 (起止点坐标、跳过阶段)
- 支持从 `start_end.json` 读取配置
- 各阶段产物落盘，失败可从断点恢复
- 全程计时并打印进度

**关键代码逻辑**:
1. 调用 `acquire_all()` 获取全部原始数据
2. 调用 `derive_terrain_factors()` + `align_all_rasters()` 预处理
3. 调用 `build_feature_stack()` + `train_random_forest()` 建模
4. 调用 `fuse_cost_surface()` + `astar_search()` + `smooth_path()` 搜索
5. 调用 `export_path()` + `compute_statistics()` + 四个可视化函数输出

---

### 2.2 `config.py` — 全局配置中心

**作用**: 所有模块共享的参数、路径、常量定义

**关键参数**:

| 参数 | 值 | 作用 |
|------|-----|------|
| `TAIWAN_BBOX` | (120.0, 21.9, 122.0, 25.4) | 台湾WGS84边界 |
| `WGS84` | EPSG:4326 | 地理坐标系 |
| `PROJECTED_CRS` | EPSG:32651 | UTM 51N投影(米制) |
| `BASE_RESOLUTION` | 90m | 分析栅格分辨率 (原始DEM=30m) |
| `MAX_SLOPE` | 35° | 硬约束坡度阈值 |
| `WATER_BUFFER` | 50m | 水域缓冲区 |
| `RF_N_ESTIMATORS` | 200 | 随机森林树数量 |
| `RF_MAX_DEPTH` | 20 | 最大树深度 |
| `FEATURE_BANDS` | 17维 | 特征向量定义 |
| `LABEL_WEIGHTS` | 加权方案 | 伪标签生成权重 |

---

### 2.3 `src/data_acquisition.py` — 数据获取 (Phase 1)

**作用**: 加载DEM、过滤输电线、下载OSM数据、生成风险代理层

**数据源与处理**:

| 数据层 | 来源 | 处理方式 | 大小 |
|--------|------|----------|------|
| **DEM** | `D:\地形数据\台湾省_DEM_30m分辨率_SRTM数据.tif` | windowed读取, NaN处理 | 369MB源文件 |
| **输电线** | `D:\输电线数据\示例数据-中国输电线路矢量.shp` | 空间过滤至台湾bbox, WGS84确保 | 1795条线 |
| **OSM道路** | Overpass API → 本地pickle缓存 | 5类公路(motorway~tertiary) | 71,043条 |
| **OSM水域** | Overpass API → 本地pickle缓存 | 河流+湖泊+水库 | 106,898条 |
| **OSM保护区** | Overpass API → 本地pickle缓存 | 国家公园+自然保护区 | 20个面 |
| **OSM土地利用** | Overpass API → 本地pickle缓存 | landuse+natural标签 | 127,968个面 |
| **OSM建筑** | Overpass API → 本地pickle缓存 | building标签 | 565,547个面 |
| **台风风险** | DEM导出代理 | 东南坡向+低海拔+陡坡加权 | 12600×7200 |
| **地震风险** | DEM导出代理 | 基础风险0.6 + 坡度放大因子 | 同上 |
| **滑坡风险** | DEM导出代理 | 坡度+TRI+剖面曲率组合 | 同上 |

**缓存机制**: OSM数据以pickle格式保存到 `data/downloaded/`, 7天内有效。超期后自动重新下载，下载失败则使用过期缓存。

**关键函数**:
- `load_dem()` — 栅格窗口读取, 处理NoData
- `load_taiwan_lines()` — GeoDataFrame空间过滤
- `_overpass_query()` — 带速率限制(2s间隔)和超时(180s)的API调用
- `build_typhoon_risk()` — 地形暴露度模型
- `build_landslide_risk()` — 滑坡敏感性模型

---

### 2.4 `src/preprocessing.py` — 空间预处理 (Phase 2)

**作用**: 统一坐标系统、提取地形因子、生成约束掩膜、对齐所有栅格

#### 2.4.1 地形因子提取

从DEM导出7个地形因子，基于Horn(1981)公式和scipy滤波器：

| 因子 | 方法 | 窗口 | 物理意义 |
|------|------|------|----------|
| **slope** | Horn公式, np.gradient | 3×3 | 坡度(度) |
| **aspect** | arctan2(dy,dx) → cos/sin | 3×3 | 坡向(方位) |
| **TRI** | |中心−邻域均值| | 3×3 | 地形粗糙度指数 |
| **TPI** | 中心−大窗口均值 | ~300m | 山脊/山谷位置 |
| **profile_curvature** | 二阶导数沿坡向 | 3×3 | 水流加速/减速 |
| **plan_curvature** | 二阶导数沿等高线 | 3×3 | 水流汇聚/发散 |
| **roughness** | 局部标准差 | 9×9 | 微地形变异性 |

#### 2.4.2 约束掩膜

**硬约束掩膜** (0=禁止建设):
- 坡度 > 35° (基于滑坡/施工安全)
- 水域 50m 缓冲区
- 自然保护区 (OSM)
- 高程 > 3500m (台湾最高玉山3952m, 3500m以上几乎不可达)

**软约束掩膜** (0-1连续惩罚系数):
- OSM土地利用分类映射为成本系数
- 无OSM数据时自动回退到基于坡度的代理

#### 2.4.3 栅格对齐

所有图层统一到90m分辨率:
- SRTM原始: 0.000278°/px (≈30.9m)
- 目标: 0.000833°/px (≈90m)
- 统一网格: 2401×4201 像素
- 重采样: 连续变量用bilinear, 分类变量用nearest

**关键函数**:
- `compute_slope()` — 梯度计算(已做度→米单位转换)
- `derive_terrain_factors()` — 7因子一站式提取
- `generate_hard_mask()` — 多约束融合
- `align_all_rasters()` — 带距离栅格计算的统一对齐
- `compute_distance_raster()` — EDT欧氏距离变换

---

### 2.5 `src/cost_model.py` — 成本建模 (Phase 3)

**作用**: 构建17维特征、生成训练标签、训练随机森林、预测成本表面

#### 2.5.1 特征工程

17维特征向量 `(2401×4201×17)`:

| 维度 | 特征名 | 类型 | 来源 |
|------|--------|------|------|
| 0 | elevation | 连续 | DEM |
| 1 | slope | 连续 | 地形因子 |
| 2-3 | aspect_cos/sin | 连续 | 地形因子 |
| 4 | tri | 连续 | 地形因子 |
| 5 | tpi | 连续 | 地形因子 |
| 6-7 | profile/plan_curvature | 连续 | 地形因子 |
| 8 | roughness | 连续 | 地形因子 |
| 9 | dist_road | 连续 | OSM道路EDT |
| 10 | dist_water | 连续 | OSM水域EDT |
| 11 | dist_existing_line | 连续 | 输电线EDT |
| 12 | landuse_code | 离散 | OSM土地利用(暂用0) |
| 13 | building_density | 连续 | OSM建筑(暂用0) |
| 14-16 | typhoon/seismic/landslide | 连续 | 风险代理层 |

#### 2.5.2 伪标签生成

因无真实建设成本数据，使用专家知识加权组合生成训练标签：

```
cost = 0.30×(1-e^(-d_existing/800))  # 距现有线越远成本越高
     + 0.25×slope/35                  # 坡度惩罚
     + 0.15×roughness/100             # 地形粗糙度
     + 0.10×(1-e^(-d_road/1500))     # 道路可达性
     + 0.10×e^(-d_water/100)          # 水域穿越惩罚
```

标签值域 [0, 1], 0=低成本, 1=高成本。

#### 2.5.3 随机森林训练

| 参数 | 值 |
|------|-----|
| n_estimators | 200 |
| max_depth | 20 |
| min_samples_leaf | 10 |
| max_features | sqrt(17)≈4 |
| 训练样本 | 100,000 (从1000万有效像素中随机采样) |
| 训练/测试 | 80/20 |
| 输出 | R² > 0.8 (训练), 特征重要性排序 |

**关键函数**:
- `build_feature_stack()` — 键名映射+NaN填充+17维堆叠
- `generate_pseudo_labels()` — 6因素加权组合
- `train_random_forest()` — 标准化+RF训练+模型保存
- `predict_cost_surface()` — 分批预测(50k/批)+硬约束INF处理

---

### 2.6 `src/path_planning.py` — 路径搜索 (Phase 4)

**作用**: 成本表面融合、A*最优路径搜索、路径平滑与约束校正

#### 2.6.1 成本表面融合

```
final_cost = gaussian_smooth(rf_cost × soft_mask) 
final_cost[hard_mask == 0] = INF
```

- 平滑sigma=1.0, 去除栅格伪影
- 归一化补偿: 权重归一化防止平滑能量损失
- 软约束锚定在[0.01, 1.0]防止零值

#### 2.6.2 A*搜索算法

**图结构**: 8邻域Moore图 (可以走对角线)

**启发式函数**: Octile距离 (比曼哈顿更紧, 比欧式更准)
```
h = c_min × [max(dr,dc) + (√2-1)×min(dr,dc)]
```

**优化策略**:
- **惰性图**: 不物化全图(~1000万节点), 邻域按需查询
- **Tie-breaking**: 小扰动因子偏向目标方向, 减少探索
- **对角线检测**: 禁止穿越两侧均为INF的对角线
- **起点/终点修复**: 如在硬约束区自动搜索最近可行点

**搜索性能**: ~1000万节点, A*约探索10-50万节点, 耗时<30秒

#### 2.6.3 路径平滑

1. **栅格→地理**: 像元坐标转WGS84经纬度
2. **RDP简化**: Ramer-Douglas-Peucker算法 (epsilon=90m), 去除冗余点
3. **B样条拟合**: scipy.interpolate.splprep (k=3三次样条)
4. **等距重采样**: 30m间距重采样
5. **硬约束校正**: 自动检测并移动违规点至最近可行区域(搜索半径20像素)

#### 2.6.4 路径距离计算

Haversine球面距离公式:
```
a = sin²(Δlat/2) + cos(lat1)×cos(lat2)×sin²(Δlon/2)
d = 2×R×atan2(√a, √(1-a))    (R=6371km)
```

**关键函数**:
- `fuse_cost_surface()` — 成本融合+validity-weighted平滑
- `astar_search()` — 标准A* + octile启发式
- `smooth_path()` — 四步平滑管线
- `_validate_and_fix()` — 硬约束后处理校正
- `compute_path_length_km()` — Haversine距离

---

### 2.7 `src/output.py` — 结果输出 (Phase 5)

**作用**: SHP/GeoJSON导出、统计计算、4幅高质量可视化

#### 2.7.1 矢量输出

- **optimal_path.shp**: ESRI Shapefile (含path_id, length_km, vertices属性)
- **optimal_path.geojson**: GeoJSON格式 (Web兼容)
- **start_end_points.shp**: 起止点 (含名称属性)

#### 2.7.2 统计指标

| 指标类别 | 具体指标 |
|----------|----------|
| 几何 | path_length_km, straight_line_km, sinuosity(弯曲度), vertices |
| 高程 | elevation_min/max/mean, total_ascent/descent |
| 坡度 | slope_mean/max, slope_pct_gt15/25/35 |
| 成本 | cost_total/mean/max |
| 约束 | hard_constraint_crossings |

导出为 `statistics.json` 和 `statistics.xlsx`。

#### 2.7.3 可视化

**图1: map_overview.png (300dpi, 双面板)**
- 左: 成本表面 (RdYlGn_r色带) + 最优路径(红色) + 现有线路(蓝色) + 硬约束(红色半透明)
- 右: 山影地形 (315°方位角, 45°高度角) + 最优路径(金色) + 现有线路
- 图上含: 起止点标注、图例、经纬度网格

**图2: map_detail.png (600dpi, 2km走廊)**
- 路径走廊局部放大, 山影底图
- 显示路径在复杂地形的局部细节

**图3: elevation_profile.png**
- 上轴: 高程剖面 (坡度颜色编码散点 + 蓝色填充)
- 下轴: 坡度分布 (橙色填充 + 15°/35°警戒线)
- 起点/终点标注 (高程值)

**图4: interactive_map.html (独立Leaflet地图)**
- OpenStreetMap底图
- 最优路径(红色粗线) + 现有线路(蓝色细线)
- 起止点标注
- 路径长度实时计算 (JavaScript Haversine)
- 支持缩放/平移/点击

---

## 三、数据流图

```
输入数据                        中间处理                     最终输出
════════                      ════════                    ════════

D:\地形数据\                   ┌──────────┐
  台湾省_DEM.tif ─────────────→│  DEM加载  │──→ dem (12600×7200)
                               └──────────┘         │
                                                    ├──→ 地形因子(7层)
D:\输电线数据\                                       │
  中国输电线.shp ─────────────→│ 输电线过滤 │──→ lines (1795条)
                               └──────────┘         │
                                                    │
Overpass API ─────────────────→│ OSM下载   │──→ roads(71k)
                               └──────────┘    water(107k)
                                                protected(20)
                                                landuse(128k) ──→ 软约束掩膜
                                                buildings(566k)
                                                    │
                                                    ├──→ 硬约束掩膜 (5.6%)
                                                    │
                               ┌──────────┐         │
                               │ 栅格对齐  │←────────┘
                               │ 90m/2401² │──→ aligned_rasters
                               └──────────┘         │
                                                    │
                               ┌──────────────┐     │
                               │ 特征堆叠17维  │←────┘
                               └──────────────┘     │
                                      │             │
                               ┌──────↓──────┐     │
现有线路(60m缓冲) ───→│ 伪标签生成   │     │
                               └──────↓──────┘     │
                                      │             │
                               ┌──────↓──────┐     │
                               │ 随机森林训练 │     │
                               │ R² > 0.80   │     │
                               └──────↓──────┘     │
                                      │             │
                               ┌──────↓──────┐     │
                               │ 成本表面预测 │────→ cost_surface.tif
                               └──────↓──────┘
                                      │
                            硬约束 + 软约束
                                      │
                               ┌──────↓──────┐
                               │ 成本融合    │────→ final_cost_surface.tif
                               └──────↓──────┘
                                      │
                                A*搜索 (起点→终点)
                                      │
                               ┌──────↓──────┐
                               │ 路径平滑    │
                               │ RDP→B样条   │
                               │ →约束校正   │
                               └──────↓──────┘
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                    optimal_path   statistics   4幅可视化
                    .shp/.geojson  .json/.xlsx  .png/.html
```

---

## 四、关键技术决策

### 4.1 为什么用90m分辨率
- 原始DEM 30m → 台湾全岛约1.93亿像素
- 90m降采样 → 约1000万像素, A*搜索可高效完成
- 30m最终精度通过RRT*走廊细化或路径平滑实现

### 4.2 为什么用WGS84存储但米制计算
- WGS84(EPSG:4326) 是通用地理坐标, 方便数据交换
- UTM 51N(EPSG:32651) 用于距离敏感计算
- 所有地形因子中的cellsize均做了度→米转换

### 4.3 为什么用伪标签而不是真实成本
- 真实建设成本数据涉及商业机密, 无法获取
- 伪标签基于专家知识和现有线路逆向推导
- 随机森林能从多维度特征中学习更复杂的成本模式

### 4.4 为什么用A*而不是更复杂的算法
- A*在栅格图上保证最优解 (对给定离散化)
- 8邻域octile启发式是admissible的 (不大于真实代价)
- 复杂度 O(N log N), 对1000万节点可在30秒内完成
- 路径平滑(RDP+B样条)消除了栅格伪影

### 4.5 硬约束穿越=0的保证
- A*搜索本身不穿越硬约束 (INF代价)
- B样条平滑可能在边界处产生少量穿越
- `_validate_and_fix()` 自动检测并将违规点移至最近可行像元

---

## 五、依赖环境

### 已安装核心库 (Python 3.13)
```
geopandas  1.1.3    # 矢量数据处理
rasterio   1.5.0    # 栅格读写与重投影
scikit-learn 1.7.2  # 随机森林
matplotlib 3.10.7   # 静态可视化
networkx   3.6.1    # 图结构 (备用)
shapely    2.1.2    # 几何计算
scipy      1.16.3   # 滤波/插值/空间分析
numpy      2.3.4    # 数组计算
```

### 额外安装 (已完成)
```
pip install folium contextily openpyxl
```

---

## 六、输入数据规格

### DEM数据
- 格式: GeoTIFF (16-bit)
- 分辨率: 30m SRTM (1 arc-second)
- 坐标: EPSG:4326
- 覆盖: 台湾省及周边
- 文件: `D:\地形数据\台湾省_DEM_30m分辨率_SRTM数据.tif`

### 输电线数据
- 格式: ESRI Shapefile (PolyLine)
- 坐标: EPSG:4326
- 属性: voltage, operator, circuits等
- 记录数: 1,795条
- 文件: `D:\输电线数据\示例数据-中国输电线路矢量.shp`

---

## 七、输出成果使用指南

| 需求 | 使用文件 | 工具 |
|------|----------|------|
| GIS分析 | optimal_path.shp | QGIS / ArcGIS |
| Web展示 | optimal_path.geojson | Leaflet / Mapbox |
| 成本分析 | cost_surface.tif, statistics.xlsx | QGIS / Excel |
| 论文配图 | map_overview.png (300dpi) | 直接使用 |
| 汇报展示 | interactive_map.html | 浏览器打开 |
| 路径验证 | elevation_profile.png | 检查高程合理性 |
| 参数调优 | statistics.json | 修改config.py后重新运行 |

---

*项目路径: D:\大创\ | 总规模: ~519 MB | 运行耗时: ~4分钟 | Python 3.13*
=======
# -
>>>>>>> a20bd10cecf9394965cb4aa8d1f7034cb2cfb346
