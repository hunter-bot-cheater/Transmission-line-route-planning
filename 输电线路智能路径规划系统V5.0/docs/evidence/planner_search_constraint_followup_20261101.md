# 搜索代价与硬约束跟进审计（2026-11-01）

## 范围

本记录涵盖 A* 搜索代价一致性、最大地面坡度硬限制的 API 参数校验、塔位场地平整度筛查状态，以及相关合成测试。没有运行台湾真实线路规划；本机仅运行小型合成数据单元测试。

## 修复与依据

1. `transmission_path_planner/experimental_astar_16.py` 中实验 A* 坡度惩罚原先将 `grade / 0.30` 封顶为 1，而生产 `astar_search` 与 `astar_path_objective` 使用未封顶公式 `weight × (grade / 0.30) × edge_length`。在坡度超过 30% 的随机小栅格上，两者搜索代价相差 381.83。实验实现现已与生产目标统一；独立代码审查确认最大地面坡度限制仍由 `max_ground_grade` 单独执行。
2. `astar_search` 在设置 `max_ground_grade` 时，原先可能因缺少 elevation 或米制像元间距而跳过边坡限制。入口现要求同时提供高程和 `pixel_spacing` 或 `pixel_m`，并拒绝非有限或非正阈值，避免把请求的硬约束静默变为未启用。
3. 7×7 全零 DEM 的塔位 pad 筛查测试包含 DEM 边缘候选；边缘候选无法获得完整 pad 覆盖，汇总状态应为 `screened_with_unknown_candidates`。实现会保留有完整覆盖的中心候选，同时披露未知候选。测试期望已按此语义修正。

## 验证

在 `输电线路智能路径规划系统V5.0` 目录执行：

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest -q tests/test_experimental_astar_16.py tests/test_smoothing_direct_shortcut.py tests/test_metrics_precision.py tests/test_engineering_assessment.py tests/test_station_planning.py
```

结果：**146 passed, 75 subtests passed**。`git diff --check` 对本轮修改文件通过。

## 独立审查摘要

- A* 离散边权和 `astar_path_objective` 的边成本、坡度项、turn 角度及非等方像元方向缩放一致。
- 路线后验距离统计按弧长加权；对称 P95 取双向较坏值，F1 使用双向长度覆盖率调和平均。
- `path_cost_integral` 是沿几何线段的软成本积分，坡度和转弯项为分段估计；它不是离散 A* 目标的精确重算。
- `cost_meter` 表示加权代价乘距离，不等于实际货币或工程造价。

## 尚未验证

真实站间线路误差、正式工程可建性和线路比较图必须通过服务器规划任务验证。只读 BatchMode SSH 到 `benke@10.166.33.23` 返回 `Permission denied (publickey,password)`；没有传递密码、没有登录服务器，也没有改动服务器数据。
