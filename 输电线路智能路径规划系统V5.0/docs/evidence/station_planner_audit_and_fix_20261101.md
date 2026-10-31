# 站间规划审查与修正记录（2026-11-01）

## 审查范围

对站间规划的端点掩膜、塔位图候选状态、约束数据接入、线路对照数据独立性和服务器复跑条件做了并行只读审查。规划仍禁止读取历史线路几何作为生成或选路先验；参考线仅用于生成候选冻结后的事后评分。

## 已修正的程序问题

1. 固定站端此前在普通 A* 模式下使用 `tower_site_mask` 校验。该掩膜含有 `--max-surface-slope-deg` 塔位坡度阈值，可能拒绝已有接入点，即使其 DEM、建筑和项目硬约束均允许架空走廊通过。现统一使用 `tower_span_mask` 校验站端；坡度和平台筛查仍仅限制实验塔位候选。回归用例验证陡坡站端可进入走廊搜索，而该像元不会成为内部塔位。
2. `tower_candidate_validation_status` 以前可能在 `clearance_profile_status=not_requested` 时仍输出 `survived_configured_screening`。现在未请求净空检查会记为 `clearance_not_screened`；配置了模型时，必须有非空逐跨样本且每跨均为 `sampled_clearance_ok` 才会记为 `survived_configured_screening`。

## 数据输入改进

- 新增可选 `--official-building-geojson`，加载已获授权的 NLSC e-Map BUILD 建筑面，与 OSM 建筑共同参与硬掩膜和精确交叉复核；查询按规划范围截取并考虑配置的建筑净距缓冲。结果仅能证明未与已加载要素相交，不能证明区域覆盖完整。
- 新增 NLSC 官方建筑数据准备脚本、SHA-256 溯源，以及读取官方农业部灾害风险 SHP 的可选输入支持；风险面作为软代价，不解释为法定禁区。本机是否持有这些经授权数据仍须逐目录核验，数据许可、年份和覆盖状态均需按具体输入核验。
- 对台电全台电桿 CSV 只做 HTTP HEAD 和 4 KiB Range 读取。文件约 320 MB，字段仅有县市/行政区/乡里、形式、桿号和 TWD97 坐标，没有电压、回路或塔序字段；没有下载完整文件，也没有将其用于线路先验或评分。

## 线路真值与复跑限制

- 既有 Deepmei–Dongshan 参考折线与 OSM 同源几何几乎完全相同；端点也是代理位置，部分候选端点几乎落在参考线上。既有评分只能称为 OSM 事后对照，不能证明独立竣工线路精度。
- 当前工作站到指定服务器 `10.166.33.23:22` 的只读 SSH 连接超时。本轮没有在本机执行真实线路规划，也没有在服务器执行任何写入或规划任务。
- 高价值真值仍是经授权、版本明确的 NLSC 1:5,000 高压线及铁塔矢量和线路侧站端接入点。B5000 WMS 可用于人工视觉对照，但图层不可查询，不能替代可审计的线路中心线矢量。

## 可复现验证

在 `输电线路智能路径规划系统V5.0/` 下运行：

```powershell
D:\python\python.exe -m pytest tests/test_official_buildings.py tests/test_station_planning.py tests/test_official_hazards.py tests/test_run_provenance.py tests/test_osm_settlement_metadata.py tests/test_geospatial_spacing.py tests/test_engineering_assessment.py -q
D:\python\python.exe -m pytest tests/test_cli_reference_independence.py tests/test_experimental_tower_graph.py -q
```

结果分别为 **156 passed, 103 subtests passed** 和 **131 passed, 25 subtests passed**。`git diff --check` 通过。以上都是自动化用例，不是实际线路误差复跑。

代码提交：`1cec1a2 Strengthen station route constraint screening`，已推送至 `codex/station-route-hardening`。
