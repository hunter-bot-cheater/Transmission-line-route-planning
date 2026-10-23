# 输电线路智能路径规划系统 V5.0

> 一套面向固定站点之间输电线路走廊初筛的规划软件。
> 路线只根据地形和经核验的环境约束生成。待评估的真实线路仅用于规划后的误差计算，不参与路线生成或候选选择。

---

## 一、软件定位

本软件以变电站/线路接入点任务表明确起终点，用数字高程模型（DEM）、地形因子与 OpenStreetMap（OSM）建筑数据进行走廊初筛。默认只运行确定性的 A*，不训练或依赖 VIN/PPO；加权 A* 可通过启发函数权重单独试跑。VIN/PPO 仅可选作诊断对照，永不进入最终选路。最终路线、硬约束及数据边界见下表和“站点间线路规划设计说明”。

| 方法 | 角色 | 原理 |
| --- | --- | --- |
| **A\*** | 默认规划器 | 8 邻域代价路径搜索；代价包含坡度、粗糙度、相对高脊、林地/湿地筛查代价和建成区代理，可计入相邻 DEM 像元地面高差软惩罚 |
| **加权 A\*** | 可选启发式实验 | 通过 `--astar-heuristic-weight` 加大终点启发权重；不保证实际加速或改善线路贴合 |
| **VIN-Grad / PPO** | 可选诊断对照 | 仅在 `--include-ai-diagnostics` 下运行，不会被选为最终路线 |

固定站点规划通过 `stations.csv` 和 `route_tasks.csv` 指定已核实的站端接入点。站点表可选填经核验的地面高程、测量来源和垂直基准，系统仅在两端测量来源齐全且垂直基准一致时报告实测高差，不据此自动判定可建性。可通过 `--constraint-geojson` 加入有来源记录的自定义禁入面或软代价面；`docs/公开数据与获取流程.md` 提供台湾官方自然保护区和饮用水保护区数据的下载、转换方法。此图层不会随仓库自动加载，建筑足迹仍作为硬障碍，OSM 建筑密度和住宅用地标记仍只是建成区代理软代价。规划结果记录 OSM 图层的缓存/覆盖状态；缺少图层时会提示，不能把空图层当成“无人居住”。目前尚无自动接入的人口统计区、敏感目标、逐档塔位及导线净空数据，因此输出属于早期走廊筛查，不是可行性批复或工程设计结论。`real_cases.csv` 端点是历史测试线路折线端点，不是核验过的变电站。

基础 A* 与加权 A* 使用同一 `1 + 4 × route_cost` 像元代价、可通行掩膜、米制 8 邻域移动规则和地面高差筛查；地形代价另含默认权重 0.10 的 300 米 TPI 正值惩罚，以弱偏好低位鞍部、减少暴露高脊。OSM 用地码 1/7（林地/湿地）可通过 `--landcover-penalty-weight` 加入软筛查代价，默认关闭；测试权重 0.10 在 12 条线路上的 F1@100m 均值从 3.29% 降至 3.07%，因此不作为默认项。这些标签不等同于法定保护范围，且数据覆盖不完整。以上代价均不读取线路几何。每条移动边按相邻两格代价的平均值乘以移动距离计价，避免正反向搜索产生方向偏置。基础 A* 的启发权重为 1，求解当前代价面的最优路径；加权 A* 仅在显式调整权重时运行，输出独立标记。VIN/PPO 默认不运行，也不会在启发式路线失败时被选作回退。真实线路误差在规划结束后计算，不改变候选和最终路线。

规划过程不读取输电线路网络、走廊、参考线或历史线路几何作为先验。我们审计发现，本机现有线路图层与当前评测线路存在同源重合，因此该图层不进入规划流程；真实线路仅在候选生成和选路完成后读取，用于独立计算相似度。测试结果用于衡量当前数据集，不代表对未见区域的泛化能力。

## 二、目录结构

```
输电线路智能路径规划系统V5.0/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── run_real_tw.bat                   # Windows 真实线路事后评估入口
├── transmission_path_planner/        # ★ 软件包（软著拟登记源代码）
│   ├── __init__.py                   #   包初始化、版本号
│   ├── __main__.py                   #   支持 python -m 运行
│   ├── config.py                     #   全局配置：依赖导入、常量、确定性种子
│   ├── geometry.py                   #   经纬度<->栅格、Haversine、路径长度
│   ├── io_export.py                  #   路径导出（SHP/GeoJSON）
│   ├── data.py                       #   DEM 读取、地形因子、山体阴影
│   ├── osm.py                        #   OSM 矢量栅格化、距离/密度/用地波段
│   ├── corridor.py                   #   旧走廊先验工具（当前 CLI 未接入）
│   ├── cost.py                       #   26 维特征栈、CostUNet、启发式代价
│   ├── methods.py                    #   A* / 加权 A* 与可选 VIN-Grad / PPO
│   ├── stations.py                   #   站点台账与站间任务校验
│   ├── smoothing.py                  #   路径平滑、硬约束吸附、高程采样
│   ├── metrics.py                    #   双向空间相似度、覆盖率、形状与长度差指标
│   ├── visualization.py              #   对比图、价值图、指标柱状图
│   └── cli.py                        #   命令行入口与批量编排
├── examples/                         # 站点和站间任务空模板
├── docs/                             # 规划设计边界与项目技术路线
├── scripts/                          # 官方开放数据获取脚本
└── 软著材料/                          # 软件著作权登记材料草稿
    ├── 软件说明书.md
    ├── 软著申请表填写说明.md
    ├── 申报资料清单.md
    └── 源程序.txt                     # 登记范围 Python 源码分页稿（提交前复核页数）
```

当前仓库不含专利申请材料目录。

## 三、运行环境

- 操作系统：Windows 10/11 或 Linux
- Python：3.9 及以上（开发验证 3.13）
- 算力：建议 NVIDIA GPU（CUDA 11.8+），CPU 亦可运行但较慢
- 关键第三方库：numpy、scipy、matplotlib、shapely、rasterio、geopandas、torch

## 四、安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 规划并对真实线路作事后误差评估（真实线不参与规划）
python -m transmission_path_planner \
    --dem ./data/taiwan_dem_35m.tif \
    --osm-cache-dir ./data/downloaded \
    --real-cases-csv ../data/real_cases_tw/real_cases.csv \
    --out-dir ./results/v5_full --no-ppp

# 3. 固定站点间规划（先复制模板并填入已核验的站端接入点）
python -m transmission_path_planner \
    --dem ./data/taiwan_dem_35m.tif \
    --osm-cache-dir ./data/downloaded \
    --stations-csv ./examples/stations.csv \
    --route-tasks-csv ./examples/route_tasks.csv \
    --constraint-geojson ./data/verified_constraints.geojson \
    --out-dir ./results/station_routes --no-ppp

# 可选：获取官方保护区筛查图层（权重需按项目情况选择）
python scripts/fetch_taiwan_constraints.py --soft-weight 1.0

# 4. 纯规划（省略评估线路；也可加 --real-cases-csv 只作事后评分）
python -m transmission_path_planner \
    --dem ./data/taiwan_dem_35m.tif \
    --osm-cache-dir ./data/downloaded \
    --out-dir ./results/v5_plan_only --no-ppp

# 5. 或直接调用包内入口
from transmission_path_planner.cli import main
main()
```

从本目录运行 `run_real_tw.bat` 可启动规划并在规划结束后对真实线路评分。脚本使用本目录内 DEM 和上一级工作区 `data/real_cases_tw/` 中的 CSV/SHP；将软件包单独复制到其他位置时，需按实际位置修改数据路径。

`config.py` 按软件目录位置解析默认数据与输出根目录，并优先查找随包 DEM。可通过 `--osm-cache-dir` 指定 OSM 环境图层缓存目录；路线规划不读取线路网络或历史线路几何。真实站点数据必须由项目方核实后填写模板。

真实评测线路只用于规划后的空间误差对照。规划输入限于地形数据、站点和有来源记录的环境约束；线路网络、走廊及任何真实路线几何均不作为规划输入。命令行参数以 `python -m transmission_path_planner --help` 为准。

官方约束数据来源、更新时间、许可、下载方式及使用边界见 `docs/公开数据与获取流程.md`。生成的数据保存在 `data/` 下并由 `.gitignore` 排除，需按数据许可和来源说明管理；仓库包含下载脚本不表示图层已核验或自动生效。

规划网格默认最长边为 4096，优先保留输入 DEM 的原始像元；运行日志会报告输入网格、实际像元尺度及降采样倍率。内存不足时可调低 `--max-side`，但会损失地形细节。当前本地 35 米 DEM 的来源与高程基准尚未核实。旧版 v1 指标曾报告第三方“20 米来源”栅格组 2/12、35 米基线组 1/12 不超过内部 8% 线；该口径已停止用于当前结果，且不构成严格分辨率对照。v2 双向空间相似度复算见 `docs/路线视觉相似度复测记录_20261022.md`。官方 2026 DTM 图幅索引仍需取得并核验源文件类型、坐标系和高程基准后才能替换。官方建物矢量可按流程转换为约束 GeoJSON；6 米在线地图图砖不能当成可分析的 DEM 数值。

## 五、历史实验记录

- 本节指标为既有项目记录，不代表本次整理后的代码已在同一数据和环境下复测；申报材料不据此保证性能。
- 旧版本文稿曾引用 `batch50/` 和 `交付_50线评估/评估报告_无泄漏泛化.md`，但这两项材料不在当前仓库，无法在此复核输入数据、参数和结果。本轮不把其中的历史指标作为当前代码的验证证据。
- `docs/精度复测记录_20261022.md` 中的 8% 结果是旧版 v1 历史口径，不是当前 v2 指标；引用时应同时写明测试集、候选生成/选优规则和误差定义。

## 六、知识产权状态

本目录包含软件著作权申请材料草稿：

- `transmission_path_planner/` 为本次登记拟采用的模块化源代码范围；
- `软著材料/` 含软件说明书、申请表字段草稿、源代码鉴别材料及申报资料清单；
- 当前仓库不含专利申请材料。

> 版本号：V5.0｜开发完成日期、首次发表和著作权人信息待申请人据实补充。

---

*本软件由大创项目「复杂山区输电线路 AI 路径规划」整理产出。*
