# 输电线路智能路径规划系统 V5.0

> 一套面向固定站点之间输电线路走廊初筛的规划软件。
> 8% 是参考线对照模式的目标阈值；无引导规划的泛化能力须按独立测试集单独报告。

---

## 一、软件定位

本软件以变电站/线路接入点任务表明确起终点，用数字高程模型（DEM）、地形因子与 OpenStreetMap（OSM）建筑数据进行走廊初筛。默认代价由地形和建成区代理构成；CostUNet 仅在显式启用时作为可选先验。软件保留 VIN-Grad、PPO 候选生成和 A* 基线，但最终路线、硬约束及数据边界见下表和“站点间线路规划设计说明”。

| 方法 | 角色 | 原理 |
| --- | --- | --- |
| **VIN-Grad** | 近似 AI 候选 | 带硬掩膜和米制方向权重的 Bellman 值场 + 沿 −∇V 的连续梯度追踪 |
| **PPO** | 辅助候选 | 近端策略优化（Clipped Surrogate + GAE）策略梯度生成器 |
| **A\*** | 确定性基线 | 8 邻域代价路径搜索；可计入相邻 DEM 像元地面高差软惩罚 |

固定站点规划通过 `stations.csv` 和 `route_tasks.csv` 指定已核实的站端接入点。站点表可选填经核验的地面高程、测量来源和垂直基准，系统仅在两端测量来源齐全且垂直基准一致时报告实测高差，不据此自动判定可建性。可通过 `--constraint-geojson` 加入有来源记录的自定义禁入面或软代价面；`docs/公开数据与获取流程.md` 提供台湾官方自然保护区和饮用水保护区数据的下载、转换方法。此图层不会随仓库自动加载，建筑足迹仍作为硬障碍，OSM 建筑密度和住宅用地标记仍只是建成区代理软代价。规划结果记录 OSM 图层的缓存/覆盖状态；缺少图层时会提示，不能把空图层当成“无人居住”。目前尚无自动接入的人口统计区、敏感目标、逐档塔位及导线净空数据，因此输出属于早期走廊筛查，不是可行性批复或工程设计结论。`real_cases.csv` 端点是历史测试线路折线端点，不是核验过的变电站。

VIN-Grad 与 A* 共用 `1 + 4 × route_cost` 像元代价、可通行掩膜和 8 邻域移动规则；VIN 的方向步长按行列实际地面间距比例计价，目标值固定为零，并禁止穿越硬禁区或对角切角。VIN 仍由神经网络近似值场并作梯度追踪，不含 A* 的相邻地面高差惩罚或地面坡度边限制，也不保证找到最短路；生成的候选会经过硬约束和地面坡度复核，未引导模式以 A* 为最终基线。

默认 `--guide-mode none` 将规划与验收分开：真实线路只在路径生成后计算指标，规划期间不用于代价、可通行区、随机重试或选路；最终路线采用确定性的 A* 代价基线。无真实线路时也会导出 A* 最终路线。`--guide-mode auto` 是兼容旧行为的显式选项：提供真实线路时用它引导规划并按参考误差选候选。此开关隔离的是推理期参考线；要报告严格无泄漏成绩，还须确认训练数据和已加载模型权重也未包含测试线路。参考线引导结果不能作为无泄漏泛化成绩。

## 二、目录结构

```
输电线路智能路径规划系统V5.0/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── run_real_tw.bat                   # Windows 参考线引导批处理入口
├── transmission_path_planner/        # ★ 软件包（软著/专利对应源代码）
│   ├── __init__.py                   #   包初始化、版本号
│   ├── __main__.py                   #   支持 python -m 运行
│   ├── config.py                     #   全局配置：依赖导入、常量、确定性种子
│   ├── geometry.py                   #   经纬度<->栅格、Haversine、路径长度
│   ├── io_export.py                  #   路径导出（SHP/GeoJSON）
│   ├── data.py                       #   DEM 读取、地形因子、山体阴影
│   ├── osm.py                        #   OSM 矢量栅格化、距离/密度/用地波段
│   ├── corridor.py                   #   既有输电线路走廊先验和缓存
│   ├── cost.py                       #   26 维特征栈、CostUNet、启发式代价
│   ├── methods.py                    #   三方法：VIN-Grad / PPO / A*
│   ├── stations.py                   #   站点台账与站间任务校验
│   ├── smoothing.py                  #   路径平滑、硬约束吸附、高程采样
│   ├── metrics.py                    #   偏差度量（对称 Hausdorff + 长度误差）与验收
│   ├── visualization.py              #   对比图、价值图、指标柱状图
│   └── cli.py                        #   命令行入口与批量编排
├── examples/                         # 站点和站间任务空模板
├── docs/                             # 规划设计边界与项目技术路线
├── scripts/                          # 官方开放数据获取脚本
├── 软著材料/                          # 软件著作权登记材料草稿
│   ├── 软件说明书.md
│   ├── 软著申请表填写说明.md
│   ├── 申报资料清单.md
│   └── 源程序.txt                     # 登记范围 Python 源码分页稿（提交前复核页数）
└── 专利材料/                          # 发明专利申请全套材料
    ├── 权利要求书.md
    ├── 说明书.md
    ├── 说明书摘要.md
    └── 附图说明.md
```

## 三、运行环境

- 操作系统：Windows 10/11 或 Linux
- Python：3.9 及以上（开发验证 3.13）
- 算力：建议 NVIDIA GPU（CUDA 11.8+），CPU 亦可运行但较慢
- 关键第三方库：numpy、scipy、matplotlib、shapely、rasterio、geopandas、torch

## 四、安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 参考线引导对比（真实线会参与规划和误差选优）
python -m transmission_path_planner \
    --dem ./data/taiwan_dem_35m.tif \
    --real-cases-csv ../data/real_cases_tw/real_cases.csv \
    --out-dir ./results/v5_full \
    --guide-mode reference \
    --real-attract 0.4 --corridor-w 2.5 --corridor-band 1200

# 3. 固定站点间规划（先复制模板并填入已核验的站端接入点）
python -m transmission_path_planner \
    --dem ./data/taiwan_dem_35m.tif \
    --stations-csv ./examples/stations.csv \
    --route-tasks-csv ./examples/route_tasks.csv \
    --constraint-geojson ./data/verified_constraints.geojson \
    --out-dir ./results/station_routes --no-ppp

# 可选：获取官方保护区筛查图层（权重需按项目情况选择）
python scripts/fetch_taiwan_constraints.py --soft-weight 1.0

# 4. 严格无引导评估（真实线只作事后评分；也可不传 --real-cases-csv）
python -m transmission_path_planner \
    --dem ./data/taiwan_dem_35m.tif \
    --real-cases-csv ../data/real_cases_tw/real_cases.csv \
    --out-dir ./results/v5_unguided \
    --guide-mode none --no-ppp

# 5. 或直接调用包内入口
from transmission_path_planner.cli import main
main()
```

从本目录运行 `run_real_tw.bat` 可启动参考线引导对比。脚本使用本目录内 DEM 和上一级工作区 `data/real_cases_tw/` 中的 CSV/SHP；将软件包单独复制到其他位置时，需按实际位置修改数据路径。

`config.py` 按软件目录位置解析默认数据与输出根目录，并优先查找随包 DEM；兼容保留旧机器的外部 DEM/SHP 路径作为回退。默认不加载 CostUNet；如要启用，须准备兼容权重并添加 `--use-costunet`。真实站点数据必须由项目方核实后填写模板。

`--guide-mode none` 是默认值。`auto` 是兼容旧行为的显式选项；`reference` 明确启用输入参考线引导。若显式设置 `--real-attract 0 --corridor-w 0`，这两项会真正关闭；未指定时，参考线模式才使用默认权重。命令行参数以 `python -m transmission_path_planner --help` 为准。

官方约束数据来源、更新时间、许可、下载方式及使用边界见 `docs/公开数据与获取流程.md`。生成的数据保存在 `data/` 下并由 `.gitignore` 排除，需按数据许可和来源说明管理；仓库包含下载脚本不表示图层已核验或自动生效。

## 五、历史实验记录

- 本节指标为既有项目记录，不代表本次整理后的代码已在同一数据和环境下复测；申报材料不据此保证性能。
- 现有无泄漏泛化报告（独立实验脚本 `batch50/`，非本 CLI 的参考线引导模式）记录：100 条测试线中，候选池 80/100 有满足逐段贴合阈值的方案；确定性选优为 42/100。该选优使用测试 CSV 的线路长度先验，部署时若没有该字段，须按实际输入重新评估。该结果支持“多方案供工程师复核”，不支持“全自动单条路线稳定达标”的表述。见 `交付_50线评估/评估报告_无泄漏泛化.md`。
- 历史报告中的 8%/10% 指标口径和测试流程并不完全相同；引用时应同时写明测试集、候选生成/选优规则和误差定义。

## 六、知识产权状态

本目录包含软件著作权材料草稿及专利相关文稿：

- `transmission_path_planner/` 为本次登记拟采用的模块化源代码范围；
- `软著材料/` 含软件说明书、申请表字段草稿、源代码鉴别材料及申报资料清单；
- `专利材料/` 含方法类发明专利的权利要求书、说明书、摘要、附图说明。

> 版本号：V5.0｜开发完成日期、首次发表和著作权人信息待申请人据实补充。

---

*本软件由大创项目「复杂山区输电线路 AI 路径规划」整理产出。*
