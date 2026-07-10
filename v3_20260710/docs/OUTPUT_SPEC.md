# V3 输出文件规范

**版本**: v3.20260710
**适用范围**: v3 及后续版本

---

## 命名格式

```
{case_id}_{algorithm}_{content}_{version}.{ext}
```

| 字段 | 说明 | 示例 |
|------|------|------|
| case_id | 案例编号 (case_01 ~ case_10) | case_01 |
| algorithm | 算法标识 (astar / ipso-sa / dbo) | ipso-sa |
| content | 内容描述 | optimal_path, quality_report, statistics |
| version | 版本号 (v3) | v3 |
| ext | 文件格式 | shp, geojson, json, png, tif |

---

## 每条线路输出清单 (存在于 output/{case_id}/)

### 矢量路径
| 文件名 | 说明 |
|--------|------|
| `{case_id}_{algorithm}_path_v3.shp` | ESRI Shapefile 路径矢量 |
| `{case_id}_{algorithm}_path_v3.geojson` | GeoJSON 路径矢量 |

### 统计与报告
| 文件名 | 说明 |
|--------|------|
| `{case_id}_{algorithm}_statistics_v3.json` | 路径统计 (长度/高程/坡度/违规) |
| `{case_id}_{algorithm}_quality_report_v3.json` | 7 项质量门控审查结果 |
| `{case_id}_{algorithm}_convergence_v3.json` | 收敛曲线数据 |

### 可视化
| 文件名 | 说明 |
|--------|------|
| `{case_id}_{algorithm}_convergence_v3.png` | 收敛曲线图 (群体智能算法) |

### 成本表面 (全算法共用)
| 文件名 | 说明 |
|--------|------|
| `{case_id}_cost_surface_v3.tif` | 最终融合成本表面 GeoTIFF |

### 对比汇总 (output/ 根目录)
| 文件名 | 说明 |
|--------|------|
| `validation_v3_summary.json` | 全量对比汇总 |
| `validation_v3_summary.png` | 对比图表 (Hausdorff/重叠率/耗时) |

## 文件内容规范

### statistics.json
```json
{
  "case_id": "case_01",
  "algorithm": "ipso-sa",
  "version": "v3.20260710",
  "length_km": 55.3,
  "elevation_min_m": 12.0,
  "elevation_max_m": 1205.0,
  "elevation_mean_m": 350.2,
  "slope_max_deg": 38.5,
  "slope_mean_deg": 8.3,
  "hard_constraint_violations": 0
}
```

### quality_report.json
```json
{
  "case_id": "case_01",
  "algorithm": "ipso-sa",
  "passed": true,
  "n_passed": 7,
  "n_total": 7,
  "checks": {
    "1_slope": {"passed": true, "value": 38.5, "threshold": 45, "detail": "..."},
    "2_water": {"passed": true, "value": 45.0, "threshold": 30, "violation_rate": "12.5%", "detail": "..."},
    "3_protected": {"passed": true, "value": 0, "threshold": 0, "detail": "..."},
    "4_curvature": {"passed": true, "value": 32.1, "threshold": 50, "detail": "..."},
    "5_sinuosity": {"passed": true, "value": 1.8, "threshold": 3.0, "detail": "..."},
    "6_elevation": {"passed": true, "value": 1200, "threshold": 2500, "detail": "..."},
    "7_cost": {"passed": true, "value": 1.2, "threshold": 3.0, "detail": "..."}
  }
}
```

### convergence.json
```json
{
  "case_id": "case_01",
  "algorithm": "ipso-sa",
  "iterations": 150,
  "best_fitness": 0.1391,
  "curve": [0.25, 0.22, 0.19, ...]
}
```
