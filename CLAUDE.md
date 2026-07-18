# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

台湾输电线路智能路径规划系统 — GIS + Random Forest cost modeling + A* heuristic search for optimal transmission line routing across Taiwan. Current version is v2 (strict quality-gated). v1 is the looser-constraint prototype kept for baseline comparison.

## Commands

```bash
cd v2_20260525
python validate_v2.py       # Run all 10 test cases (v2, recommended)

cd v1_20260525
python main.py              # Single-line: Pingtung → Taipei (default)
python main.py --start-lat 22.0 --start-lon 120.5 --end-lat 25.0 --end-lon 121.5
```

No test runner, no linter, no build step. This is a computational research pipeline — each run produces GIS output files (SHP, GeoJSON, PNG, JSON) under `output/`.

## Architecture

### 5-phase pipeline

```
Phase 1: Data Acquisition    shared/data_acquisition.py
          ↓                  DEM (SRTM) + OSM (Overpass API) + power-line shapefile + risk proxies
Phase 2: Preprocessing       v2_20260525/src/preprocessing.py
          ↓                  Terrain factors (7 layers) → aligned 90m grid → hard/soft constraint masks
Phase 3: Cost Modeling       v2_20260525/src/cost_model.py
          ↓                  18-dim feature stack → pseudo-label generation → Random Forest (R² > 0.97)
Phase 4: Path Planning       v2_20260525/src/path_planning.py
          ↓                  Cost fusion → A* search (8-neighbor, octile heuristic) → smoothing → 7 quality gates
Phase 5: Output              v2_20260525/src/output.py
                             SHP / GeoJSON / statistics JSON / map PNG / interactive HTML
```

### Key design decisions

- **Why Random Forest?** With no real cost data, 17-dim terrain+infrastructure+risk features feed a pseudo-label formula weighted by distance to existing lines. RF is interpretable (feature importance) and avoids the convergence risk of deep RL on a 10M-pixel search space.
- **Why A\*, not metaheuristics?** The cost surface is a deterministic, known grid after Phase 3. A\* with octile heuristic guarantees optimality. GWO, WOA, and IPSO-SA are planned as comparison baselines to validate this choice.
- **Pseudo-label generation** (cost_model.py:78-182) is the core innovation — an expert-knowledge formula encodes construction cost without real cost data, using exponential decay from existing lines, slope penalties, water proximity, and roughness.
- **7 quality gates** (path_planning.py:413-643) are the engineering differentiator from v1: slope ≤ 45°, ≤25% water-proximity violations, zero hard-constraint intrusions, max turn ≤ 50°, sinuosity ≤ 3.0, elevation continuity ≤ 2500m, cost anomaly ratio ≤ 3.0.

### v1 ↔ v2 relationship

v2 is v1 with tightened parameters and quality gates added. The pipeline structure is identical. `shared/data_acquisition.py` is used by both. v1 source (`v1_20260525/src/`) contains older copies of preprocessing, cost_model, path_planning, and output — when fixing a bug that exists in both versions, apply the fix to v2 first, then backport to v1 only if needed. Duplicated utility functions (`geo_to_grid`, `compute_path_length_km`, RDP fallback, etc.) should eventually be extracted to `shared/path_utils.py`.

### External data dependencies

Two files must exist locally (paths configured in `config.py`):
- DEM: GeoTIFF elevation data (30m or 12.5m), placed at `DATA_DIR / "dem" / <filename>.tif`
- Power lines: Shapefile with .shp/.shx/.dbf/.prj companions, placed at `DATA_DIR / "shp" / <filename>.shp`

OSM data is auto-downloaded via Overpass API and cached for 7 days in `data/downloaded/`.

## Configuration

`config.py` is the single source of truth for all parameters. Every module imports it as `import config as cfg`. Never hardcode numeric thresholds or paths — add them to config.py.

Key config sections:
- `BASE_DIR`, `DATA_DIR`, `DEM_PATH`, `SHP_PATH` — environment-specific paths
- `MAX_SLOPE`, `WATER_BUFFER`, `MAX_ELEVATION`, `PROTECTED_BUFFER`, `BUILDING_DENSITY_LIMIT` — hard constraints
- `SLOPE_QUALITY_THRESHOLD`, `WATER_BUFFER_QUALITY`, `MAX_TURN_ANGLE`, `MAX_SINUOSITY`, `MAX_CONTINUOUS_CLIMB`, `COST_ANOMALY_RATIO` — quality gates
- `LABEL_WEIGHTS`, `PSEUDO_LABEL_PARAMS` — pseudo-label formula
- `ASTAR_NEIGHBORHOOD`, `ASTAR_HEURISTIC_WEIGHT`, `ASTAR_DIAGONAL_COST` — A* behavior

## Coding conventions

From `v2_20260525/docs/CODING_STANDARDS.md`:

- **Naming**: modules/functions/variables = `snake_case`, classes = `CapWords`, constants = `UPPER_CASE`, private = `_prefix`
- **Type annotations**: required on all function parameters and return values
- **Imports**: 3 groups (stdlib → third-party → local), each separated by blank line
- **Docstrings**: Google style with `Args:`, `Returns:` sections required for all public functions
- **Indentation**: 4 spaces, no tabs. Max line width 100 chars.
- **Error handling**: custom exception classes defined in the standard (`PathPlanningError`, `HardConstraintViolationError`, `NoPathFoundError`, `InvalidEndpointError`) should be used rather than returning `None` with print statements

## Notes

- `data/` and `output/` are gitignored. Large raster files (.tif) and OSM caches (.pkl) should never be committed.
- The analysis resolution is 90m (downsampled from 30m or 12.5m source). Grid size is approximately 2401 × 4201 pixels for Taiwan.
- Coordinate systems: WGS84 (EPSG:4326) for geographic, UTM 51N (EPSG:32651) for projected (meters).
- `geo_to_grid(lat, lon, transform)` takes lat FIRST, then lon — swapping them is a common bug.

## 已知问题（待修复）

### 伪标签权重和软约束系数缺乏严格依据

`LABEL_WEIGHTS` 中的 7 个权重（dist_existing=0.65, slope=0.15, water=0.08, landuse=0.04, protected=0.04, road_access=0.02, railway=0.02）和 `LANDUSE_SOFT_COST`（8 种土地利用类型的软约束系数，0.1-0.9）是经验设定的，未经过：
- 参数敏感性分析（每个权重±10%对结果的影响）
- 文献定量支撑
- 真实造价数据校准

当前通过 10 条真实线路的 Hausdorff/重叠率间接验证了合理性，但缺乏严格论证。建议补充用 RF 特征重要性反证权重方向正确性，并对关键权重做敏感性分析。

### IPSO-SA 和 DBO 收敛到相同直线路径

v3 对比实验发现 IPSO-SA 和 DBO 在 10 条线路上产生几乎相同的直线路径，且与 A\*（栅格锯齿弯曲）不同。根因分析：

- **连续路标点编码**：IPSO-SA/DBO 使用 M 个 (lat,lon) 路标点 + 线性插值，路径不受栅格限制，直线在连续空间中就是最优解
- **A\* 热启动**：两个算法种群初始化时个体 1 均为 A\* 路径降采样，提供了相同的搜索起点
- **相同适应度函数**：两者共享 `_evaluate_fitness()`，同一成本表面 + 同一打分 → 导向同一局部最优
- **走廊宽度 15km 过窄**：种群被限制在起止点 ±15km 范围内，搜索空间不足以产生差异化路径

注意：验证了 corridor_bonus 不是根因——削弱它后 Hausdorff 反而恶化（6419m→10938m），说明 corridor_bonus 正确地引导路径贴近已有线路走廊。

建议修复：取消 A\* 热启动、增加路标点密度（1.5→3.0/km）、扩大走廊（15→30km）、给 DBO 单独使用不同的初始化策略。

### 12.5m DEM 重采样导致路径质量下降（已解决）

`shared/data_acquisition.py` 中的 `_resample_dem_to_standard` 在管线开始前将 DEM 重采样到 90m，导致所有地形因子在粗网格上计算，微地形被抹平。改用 30m SRTM DEM 原生分辨率计算地形后再降采样，V2 Hausdorff 从 10954m 改善至 6419m，质量通过率从 1/10 提升至 10/10。注意：shared/data_acquisition.py 不应包含预重采样逻辑，V3 如需适配高分辨率 DEM 应使用自己的 `data_acquisition_v3.py`。

## 版本说明

| 版本 | 目录 | 内容 |
|------|------|------|
| V1 | `v1_20260525/` | 宽松约束原型 |
| V2 | `v2_20260525/` | A\* + 7项质量门控 (当前稳定版) |
| V3 | `v3_20260710/` | 纯群体智能框架 (A\*/IPSO-SA/DBO + RF成本表面) |
| V4 | `v4_20260718/` | CNN增强版 (U-Net成本预测 + MLP启发式 + 完整AI管线) |

**版本间关系**: V4 从 V3 演进而来，新增 `cnn_model/` 目录和 CNN 成本表面功能。V3 回退为纯群体智能版本作为对照基线。V2 是稳定的 A\* 单算法版本。

---

## 必做事项

**0. 版本变更时必须同步更新的文件：**
- `CLAUDE.md` — 版本说明 + 已知问题
- `README.md` — 版本介绍 + 对比结果
- `docs/changelog/` — 修改记录
- 桌面 `C:\Users\86133\Desktop\大创项目更改说明袁润熙\` — 个人备份

以上文件在每次重大改动后**必须检查是否需要更新**，不得遗漏。

**1. 每次 `git commit` 并推送后，必须更新修改记录文件：**

**1. 每次 `git commit` 并推送后，必须更新修改记录文件：**

```
docs/changelog/<YYYY-MM-DD>_项目修改记录.txt
```

记录内容包括：
- 提交哈希和说明
- 修改了哪些文件
- 新增/删除/重命名了什么
- 修改原因（为什么改、解决什么问题）

按日期命名，同一天多次提交合并在一个文件中。

**2. 修改代码前必须确认范围：**
- 用户说"本地修复" → 只在本地修改，不 commit，不 push
- 用户说"提交"或"推送" → 才执行 git commit + push
- 用户没有明确指示时 → 先询问是否推送到 GitHub

**3. 推送后同时更新两份修改记录：**
- 仓库内 `docs/changelog/` （队友可见）
- 桌面 `C:\Users\86133\Desktop\大创项目更改说明袁润熙\` （个人备份）
- 提交哈希和说明
- 修改了哪些文件
- 新增/删除/重命名了什么
- 修改原因（为什么改、解决什么问题）

按日期命名，同一天多次提交合并在一个文件中。文件模板参考已有的 `2026-07-06_项目修改记录.txt`。
