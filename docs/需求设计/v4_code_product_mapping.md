# v4 代码 ↔ 产物对应关系 (Code–Product Mapping)

> 目标脚本：`D:\大创\scripts\ai_path_planning_v4.py`
> 输出根目录：`D:\大创\outputs\`（由 `--out-dir` 指定子目录）
> 生成日期：2026-08-10 ｜ 配套任务：优化/审核扩展到 v4 + 接 `shared/data_acquisition.py` 补齐波段

---

## 1. 概述

`ai_path_planning_v4.py` 是"端到端深度学习选线"版本，相比 v3 集成脚本的核心差异：

- **三种路径算法**同场对比：`VIN-Grad`（神经值传播+梯度追踪，主 AI 方法）、`PPO`（RL 策略梯度）、`A*(baseline)`（传统对照）。
- **26 维特征栈**（`compute_osm_bands` + `build_feature_stack`），但 **v4 自带 `compute_osm_bands`，不调用 `shared/data_acquisition.py`**（这是与 v3 集成脚本的关键分叉，参见第 7 节）。
- **真实线对比 / 走廊引导**：通过 `--real-cases`（R01–R04 带地面真值）或 `--real-shp`（全区线网）启用，叠加"走廊代价惩罚 + 真实线方向吸引力"把规划路径拉向真实走廊。
- 全部依赖 torch（CostUNet / VIN / PPO）；无 torch 时退回启发式代价面且跳过 VIN/PPO。

---

## 2. 运行模式矩阵（CLI → 行为 → 产物目录）

| 触发方式 | 用例来源 | `use_real` | 走廊引导 | 典型产物目录 |
|---|---|---|---|---|
| 默认（无真实参数） | `DEFAULT_CASES` → T01–T04 | False | 关闭 | `v4_ppo_fixed_final8` |
| `--real-cases` | `REAL_CASES` → R01–R04（含真值） | True | 默认开（w=1.2, band=1500） | `v4_real_compare` / `v4_real_optimized` / `v4_real_optimized_ppo` |
| `--real-shp <shp>` | `DEFAULT_CASES`(T0X) + 全局线网参考 | True | 按参数 | `v4_t01_network_full` / `v4_t01_network_smoke` |
| `--with-real` | 需配合上面两者之一 | True | 按参数 | （与上述组合） |
| `--no-ppp` | 任意 | 同 | 同 | 跳过 PPO |
| `--start/--end lat,lon` | 单条 `Custom` 用例 | 取决于上面 | 同 | 自定义 |
| `--cases N` | `DEFAULT_CASES[:N]` | False | 关闭 | 限制默认用例数 |

**走廊引导实测差异**（解释 `real_compare` vs `real_optimized`）：

| 目录 | `--corridor-w` | `--corridor-band` | VIN-Grad 对真实线 Hausdorff(km) | 解读 |
|---|---|---|---|---|
| `v4_real_compare` | 0.90 | 4000 m（宽） | 3.0 / 6.5 / 1.2 / 7.0 | 弱引导 → VIN 自由游走，偏差大（更"自然"） |
| `v4_real_optimized` | 1.20 | 1500 m（窄） | 0.89 / 0.67 / 0.90 / 0.61 | 强/紧引导 → 路径贴合真实走廊（"优化"） |

---

## 3. 算法 / 方法映射（函数 → 方法名 → 何时运行）

| 方法名（写入 `methods` 字典） | 核心函数（行号） | 运行条件 |
|---|---|---|
| `VIN-Grad` | `train_vin`(844) / `gradient_track_path` / `build_vin_extra`(1890) | `HAS_TORCH` 为真 |
| `PPO` | `train_ppo`(947) / `build_ppo_context`(899) / `ppo_rollout_path`(1157) | `HAS_TORCH` 且 **非** `--no-ppp` |
| `A*(baseline)` | `astar_search`(1381) | **始终运行**（传统对照） |

> 导出矢量文件名由方法名推导（`main()` 行 2204）：
> `VIN-Grad`→`planned_route_vingrad.*`，`PPO`→`planned_route_ppo.*`，`A*(baseline)`→`planned_route_aastar.*`。

---

## 4. 产物映射表（代码写出点 → 文件 → 含义 → 条件）

### 4.1 逐走廊目录 `OUT/<corridor>/`
| 文件 | 写出点 | 含义 | 条件 |
|---|---|---|---|
| `path_comparison_<safe>.png` | `plot_comparison`(2199) | 该走廊多算法路径叠加对比图 | 每走廊必有 |
| `planned_route_vingrad.shp/.geojson` | `export_route`(2205) | VIN-Grad 路径矢量 | `HAS_TORCH` |
| `planned_route_ppo.shp/.geojson` | `export_route`(2205) | PPO 路径矢量 | torch + 非 `--no-ppp` |
| `planned_route_aastar.shp/.geojson` | `export_route`(2205) | A* 路径矢量 | 始终 |
| `real_route.shp/.geojson` | `export_route`(2209) | 真实/参考线路矢量 | 仅 `use_real` 且该走廊有真实线 |
| `paths.json` | (2212) | 全部方法经纬度路径数组 | 每走廊必有 |

### 4.2 首走廊聚合（仅第一个成功用例，写 `OUT/` 根）
| 文件 | 写出点 | 含义 | 条件 |
|---|---|---|---|
| `path_comparison.png` | `plot_comparison`(2218) | 首走廊对比图（聚合） | 首用例 |
| `value_map.png` | `plot_value_map`(1835, 2223) | VIN 值传播概率图 | torch + VIN-Grad 且 `plot_value_map` 成功 |
| `metrics.json` | (2225) | 首走廊 VIN-Grad 指标（length_km/cum_relief/mean_cost/sinuosity/hard_violations/time_s 等） | VIN-Grad 在 `case_metrics` 中 |

### 4.3 全局聚合（所有走廊处理完后，写 `OUT/` 根）
| 文件 | 写出点 | 含义 | 条件 |
|---|---|---|---|
| `algo_comparison.png` | `plot_algo_comparison`(1857, 2235) | 各走廊/算法指标对比条形图 | 始终（⚠️ **已知崩溃点**：PermissionError，见第 6 节） |
| `comparison_metrics.json` | (2236) | 全部用例逐算法指标 | 始终 |
| `comparison_metrics.csv` | (2249) | 扁平指标表 | 有结果时 |
| `real_comparison_metrics.json` | (2270) | 规划路径 vs 真实线偏差（Hausdorff/mean/p90/max）+ 基线 VIN 无引导记录 | 仅 `use_real` |

### 4.4 副作用（非 `--out-dir` 控制）
| 文件 | 写出点 | 含义 | 风险 |
|---|---|---|---|
| `D:\大创\outputs\_last_V.npz` | `np.savez`(2109) | 调试用 V/cost/hard/起终点，便于免重训秒级重测 PPO | ⚠️ **硬编码绝对路径**，破坏可移植性（同 v3 Finding 6 家族） |

---

## 5. 各产物目录实测解读

| 目录 | 模式 | 走廊 | 算法产物 | 关键状态 |
|---|---|---|---|---|
| `v4_ppo_fixed_final8` | 默认 T01–T04，`use_real=False` | T01–T04 | vingrad/ppo/aastar 三套矢量齐全；`algo_comparison.png`+`comparison_metrics` 齐全 | 完成。缺 `value_map.png`（疑似当时 `plot_value_map` 异常被用例级 `except` 吞掉或代码快照差异）。历史曾因 `plot_algo_comparison` PermissionError 崩溃，后重跑成功 |
| `v4_real_compare` | `--real-cases` R01–R04，弱引导(w=0.9,band=4000) | R01–R04 | 齐全 + `real_comparison_metrics.json` + `value_map.png` | 完成。VIN 偏差大（对比基线） |
| `v4_real_optimized` | `--real-cases` R01–R04，强引导(w=1.2,band=1500) | R01–R04 | 齐全 + `real_comparison_metrics.json` + `value_map.png` | 完成。VIN 偏差显著减小（贴合真实走廊） |
| `v4_real_optimized_ppo` | `--real-cases` + PPO | R01 起 | **空目录，仅 `run.log`** | ❌ 失败：R01 在 `[PPO] 训练策略网络 ...` 后进程死亡（CUDA OOM/段错误），崩溃发生在逐走廊导出（2200）之前，故无任何产物 |
| `v4_t01_network_smoke` | `--real-shp` 全局线网 + T01（冒烟） | T01 | 齐全 + `real_comparison_metrics.json` + `algo_comparison.png` | 完成（冒烟级） |
| `v4_t01_network_full` | `--real-shp` 全局线网 + T01/T02（全量） | T01/T02 | 有 `value_map.png`/`path_comparison.png`/metrics，但**缺 `algo_comparison.png` 与 `real_comparison_metrics.json`**，`run.log` 0 字节 | ⚠️ 不完整：聚合阶段前进程被杀（疑似 OOM），故缺聚合产物 |

---

## 6. 跨版本一致缺陷（v3 集成脚本 & v4 共有）

1. **`土地利用=0`（系统性空波段）**
   - v4 两条真实运行日志均打印 `OSM 几何数: ... 土地利用0 ...` —— 土地利用多边形始终为 0。
   - v3 集成脚本 `landuse_code` 默认哨兵原为常数 `8`（与真实类别混淆），本次审核已改为中性哨兵 `0`（见 Task #12 修复 Finding 3）。
   - 根因待查：`taiwan_landuse.pkl` 是否为空 / Overpass `fetch_osm_landuse` 是否漏查 tag。需在 Task #14 中验证。

2. **`plot_algo_comparison` 写出健壮性（PermissionError）**
   - 写出 `algo_comparison.png`(2235) 时若目录无写权限 / 文件被占用，会抛 `PermissionError: [Errno 13]`。当前仅被外层 `except` 捕获并打印 `[WARN]`，但已写出的逐走廊结果不受影响。建议加 try + 备用文件名 / 重试。

3. **`plot_value_map` 未单独 try 包裹**
   - 行 2223 `plot_value_map` 在首走廊块内、未被独立 try 保护；一旦抛异常会被行 2228 的用例级 `except` 吞掉，导致该走廊被标记为"处理失败"且 `main_done` 不置位（后续走廊重复进入首走廊块）。轻微健壮性 bug。

4. **硬编码 `outputs/_last_V.npz` 绝对路径**（行 2109）—— 可移植性问题，应改到 `OUT/` 下。

---

## 7. v4 与 v3 集成脚本的关系（成本面 & 算法分叉）

| 维度 | v3 集成脚本 `scripts/ai_path_planning.py` | v4 `scripts/ai_path_planning_v4.py` |
|---|---|---|
| OSM/风险数据源 | **复用 `shared/data_acquisition.py`**（config shim + 离线缓存） | 自带 `compute_osm_bands`，**不调用 shared** |
| 成本面 | CostUNet(可选) 或 启发式；OSM 波段 13–25 由 `compute_osm_feature_bands` 补齐 | `costunet.predict` 或 `heuristic_cost`；OSM 由自带函数 |
| 规划算法 | A* / Dijkstra | VIN-Grad + PPO + A*（三者对比） |
| 真实线接入 | 无（独立 AI 规划） | `--real-cases`/`--real-shp` + 走廊引导 |
| 26 维特征 | `FEATURE_BANDS`（来自 v3 config） | 自带相同 26 维布局 |

> **统一方向**：把 v4 的 `compute_osm_bands` 也改为调用 `shared/data_acquisition.py`（与 v3 集成脚本对齐），可一并修复两版的"土地利用=0"数据缺陷，并消除重复的 OSM/风险抓取逻辑。

---

## 8. Task #14 待办（v4 代码优化 / 审核）

- [ ] **修复 `plot_algo_comparison` 写出健壮性**：try + 备用文件名 / 重试，避免 PermissionError 中断聚合。
- [ ] **修复 `plot_value_map` 未独立 try**：拆分首走廊块，失败不影响 `main_done` 与后续走廊。
- [ ] **调查 `土地利用=0`**：确认 `taiwan_landuse.pkl` 内容与 `fetch_osm_landuse` 查询；必要时统一到 shared 模块。
- [ ] **`v4_real_optimized_ppo` 崩溃**：PPO 训练 CUDA OOM/进程死亡 → 减小显存占用（梯度累积 / 更小 batch / `torch.cuda.empty_cache()`）/ 捕获进程级异常，保证至少写出已完成走廊产物。
- [ ] **硬编码路径**：`outputs/_last_V.npz` 改到 `OUT/` 下；顺带核对 v4 内其他 `D:\大创` 硬编码（如 `REAL_PATH_DIR`、`BASE_DIR`）。
- [ ] **算法一致性**：核对 v4 的 A*（1381）与 v3 集成脚本 A* 在成本面/不可通行判定上是否一致，避免两版结论不可比。
