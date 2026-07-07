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

---

## 必做事项

**每次 `git commit` 并推送后，必须更新修改记录文件：**

```
docs/changelog/<YYYY-MM-DD>_项目修改记录.txt
```

记录内容包括：
- 提交哈希和说明
- 修改了哪些文件
- 新增/删除/重命名了什么
- 修改原因（为什么改、解决什么问题）

按日期命名，同一天多次提交合并在一个文件中。文件模板参考已有的 `2026-07-06_项目修改记录.txt`。
