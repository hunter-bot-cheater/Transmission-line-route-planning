# 输电线路智能路径规划系统 V5.0

> 一套面向固定站点之间输电线路走廊初筛的规划软件。
> 路线只根据地形和项目方提供的环境约束生成。`source`、`verified` 等字段是项目方填写的来源/核验声明，程序不独立核验其真实性。待评估的真实线路仅用于规划后的误差计算，不参与路线生成或候选选择。

---

## 一、软件定位

本软件以变电站/线路接入点任务表明确起终点，用数字高程模型（DEM）、地形因子与 OpenStreetMap（OSM）建筑数据进行走廊初筛。默认只运行确定性的 A*，不训练或依赖 VIN/PPO；加权 A* 可通过启发函数权重单独试跑。VIN/PPO 仅可选作诊断对照，永不进入最终选路。最终路线、硬约束及数据边界见下表和“站点间线路规划设计说明”。

| 方法 | 角色 | 原理 |
| --- | --- | --- |
| **A\*** | 默认筛查规划器 | 启发权重 1.0；对当前离散代价面求最优路线。可用大于 1 的权重换取更快的近似搜索 |
| **基础 A\*** | 可选最优路径对照 | `--astar-heuristic-weight 1.0` 求解当前代价面的最优路线 |
| **转弯代价 A\*** | 可选实验模式 | `--turn-penalty-weight` 启用方向状态搜索；内存占用较高，默认关闭 |
| **塔位图** | 实验筛选候选 | `--experimental-tower-graph` 显式提供档距参数；考虑候选塔位及跨距，不自动进入最终选路 |
| **VIN-Grad / PPO** | 可选诊断对照 | 仅在 `--include-ai-diagnostics` 下运行，不会被选为最终路线 |

固定站间规划通过 `stations.csv` 和 `route_tasks.csv` 指定站端接入点。正式站间任务的两端都必须声明 `point_role=line_connection_point`、`verification_status=verified`，并填写坐标来源 `source` 与可追溯凭证 `verification_reference`；默认拒绝缺少这些声明的任务。旧式或未核验坐标若只用于探索，可显式加 `--allow-unverified-endpoints`，输出会标记 `exploratory_only_unverified_endpoints`，不得解释为正式站间候选。程序只校验声明字段，不独立认证坐标、站点身份或凭证真实性；`coordinate_crs` 仍须声明 `EPSG:4326`，且不会自动转换坐标。端点核验声明不会放宽 DEM NoData、OSM 建筑或自定义硬约束检查。实测站点高程仅在净空筛查中、且两端高程来源齐全、站点间垂直基准相同并与显式 `--dem-vertical-datum` 声明一致时用于首末跨距端点；否则该塔位净空候选标记为 unknown。DEM 垂直基准本身仍是调用者声明，程序不独立核验。输出会逐跨记录采样最小净空；DEM-only 筛查通过不等于工程净空认证。该实验模型对站端和中间塔均使用同一个固定挂点高度，不代表真实站内出线构架或导线接续高度。若在 `--constraint-geojson` 中提供约束，`source` 是必填的来源声明；程序只检查必要字段、WGS84 坐标及几何格式，不核验来源真伪、数据日期、法规效力、边界精度或覆盖完整性。

农业部 115 年土石流影响范围、大规模崩塌潜势区及影响范围 MVT 仅在本地预置于默认数据目录时读取，作为联合软代价，不把潜势区当禁区。MVT 为 z10 简化瓦片，任务结果记录清单覆盖状态和沿线路径的栅格暴露比例。整个 `data/` 目录由仓库根目录 `.gitignore` 排除，DEM、MVT 瓦片及其 `manifest.json`/校验文件、OSM 缓存等数据不随 Git 提交；新克隆后须按 `docs/公开数据与获取流程.md` 预置所需数据，并核对 MVT 的 `manifest.json`、`tiles.sha256` 及运行结果中的任务覆盖状态。OSM `features_observed` 表示缓存中观测到与请求范围相交的要素，不证明 OSM 对该区域覆盖完整；OSM 建筑足迹只在已有数据范围内作为硬障碍，建筑密度与住宅用地仅是建成区代理。可选 WorldPop 100 米人口栅格默认关闭；官方最小统计区人口边界当前 worktree 未取得，不能把村里边界或 WorldPop 栅格冒充为最小统计区统计。敏感目标、真实逐档塔位及导线工程净空数据仍未自动接入，因此输出属于早期走廊筛查，不是可行性批复或工程设计结论。`real_cases.csv` 端点是历史测试线路折线端点，不是核验过的变电站。

默认软代价由地形坡度/粗糙度、TPI、OSM 建成区代理、官方灾害掩膜联合构成；可选人口层仅在显式设置 `--population-raster` 和正的 `--population-weight` 时参与，默认关闭。人口栅格按源像元面积换算为 people/km²，再用固定对数尺度归一化；NoData 不当作零人口，而是按保守的最大代理代价搜索并记录覆盖缺口。WorldPop 是约 100 米模型估算，重采样到更细 DEM 不会提高其真实精度。所有权重均是可审计的筛查参数，不是已校准的工程造价或法规阈值。OSM 建筑层使用约 500 米平滑的“建筑触及像元比例”代理，它不等于真实建筑占地率或人口暴露量；OSM 林地/湿地代价默认关闭。普通 A* 按相邻像元代价均值与米制移动距离积分，参考线路几何不参与规划。方向状态 A* 的正转弯权重通过 `--turn-penalty-weight` 显式开启，权重 0 保持旧模式；高分辨率网格的额外内存需求会在日志中提示。实验塔位图需要显式给出最小/最大档距，可选输入固定挂点、弧垂和净空值；其塔位成本目前为零，结果仅作候选筛选，不能宣称满足杆塔/导线规范。历史 12 条线路目前冻结为事后诊断集，路线相似度不再作为调参或选路目标；更重要的验收项是数据覆盖可追溯、路径无已配置硬约束违规、风险暴露可见且结果对权重变化可解释。VIN/PPO 默认不运行，也不会作为最终路线回退。

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

# 3. 固定站点间规划（先按项目资料核实并填写站端接入点）
python -m transmission_path_planner \
    --dem ./data/taiwan_dem_35m.tif \
    --osm-cache-dir ./data/downloaded \
    --stations-csv ./examples/stations.csv \
    --route-tasks-csv ./examples/route_tasks.csv \
    --constraint-geojson ./data/project_constraints.geojson \
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

`config.py` 按软件目录位置解析默认数据与输出根目录；只有在本地预置 DEM 时才会从默认路径读取。可通过 `--dem`、`--osm-cache-dir` 指定其他本地文件。路线规划不读取线路网络或历史线路几何。真实站点数据须由项目方核实并填写模板，程序不替项目方核验。

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
