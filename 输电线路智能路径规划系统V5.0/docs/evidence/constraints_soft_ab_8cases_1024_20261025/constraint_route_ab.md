# Soft constraint paired route screening

- Cases: 8 total; 7 paired routes; 1 with at least one unsuccessful result.
- Raster cap: `1024`; search margin: `0.3` degrees.
- Median paired length change: `+0.000 km` (treatment minus baseline).
- Median symmetric P95 distance: `5.395 km` to `6.449 km`.
- Median F1@100m: `1.844%` to `1.836%`.
- Sum of per-route exact vector overlap with soft polygons: `10.123 km` to `0.000 km`.
- Soft features: `140`; by category: `{"drinking_water_protection": 134, "marine_wildlife_habitat": 6}`; each feature weight is `1.0`.
- DEM SHA-256: `dc963ff6047f911275e9d2b3be678539f09ee8457939511c094721be6a2b4025`
- Constraint GeoJSON SHA-256: `5967d659856b925cd42909a6b5e1cf4ac09ae808d6d1c8e7ecd851003fa530b5`

| Case | Status base / treatment | Length base / treatment (km) | Δ length (km) | Raster soft cells base / treatment | Exact vector overlap base / treatment (km) | Symmetric P95 base / treatment (km) | F1@100m base / treatment | Failure detail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TW102 | route / route | 48.95 / 48.95 | +0.00 | 0 / 0 | 0.00 / 0.00 | 3.03 / 3.03 | 2.68% / 2.68% | — |
| TW173 | route / route | 52.40 / 52.89 | +0.48 | 38 / 1 | 2.63 / 0.00 | 5.39 / 6.45 | 1.84% / 1.84% | — |
| TW252 | failed / failed | — | — | — | — | — | — | 起点位于不可通行像元（OSM建筑足迹）；请核对接入点坐标或约束 |
| TW345 | route / route | 52.17 / 52.17 | +0.00 | 0 / 0 | 0.00 / 0.00 | 5.24 / 5.24 | 1.49% / 1.49% | — |
| TW59 | route / route | 58.66 / 58.66 | +0.00 | 0 / 0 | 0.00 / 0.00 | 8.95 / 8.95 | 2.37% / 2.37% | — |
| 天輪~龍潭線 | route / route | 95.10 / 100.88 | +5.78 | 71 / 0 | 7.49 / 0.00 | 9.34 / 10.80 | 1.47% / 1.42% | — |
| 明潭-鳳林線 | route / route | 66.13 / 66.13 | +0.00 | 0 / 0 | 0.00 / 0.00 | 2.91 / 2.91 | 7.97% / 7.97% | — |
| 深美～冬山線 | route / route | 45.85 / 45.85 | +0.00 | 0 / 0 | 0.00 / 0.00 | 6.49 / 6.49 | 1.58% / 1.58% | — |

## Interpretation limits

This is a sensitivity comparison, not a weight calibration. The reference line geometries were used only by the planner's post-hoc scorer; they did not enter route search or candidate selection. The CSV endpoints are still historical-line endpoints, not independently verified station terminals.
The 1024-cell cap produced roughly 102–134 m cells across these windows, so this comparison cannot support high-resolution engineering claims. OSM building and land-use coverage is observed-feature coverage, not proof of completeness. The official protection polygons are screening layers; overlap or avoidance does not establish legal status, risk reduction, or route feasibility.
The soft layer removed about 10.1 km of summed route-polygon overlap across the seven successful pairs, while median symmetric P95 distance worsened. That is an avoidance-versus-reference-fit tradeoff, not an overall accuracy gain; the weight is uncalibrated.
Vector overlap is measured after projecting the route and polygons to EPSG:3826. Raster exposure is all-touched cell contact on the exact planning grid; the two measures need not agree at coarse resolution.

## Reproduction

The paired commands used identical DEM, OSM cache, endpoints, A* settings, official hazard MVT, and `--max-side 1024`. The treatment additionally supplied `--constraint-geojson`; the baseline omitted it. Outputs remain under the ignored `outputs/constraints_soft_ab_8cases_1024_20261025/` directory.
The adjacent `constraint_route_ab.json` preserves each run's exact command and working directory, planner source hashes, runtime flags, planning-input hashes, and run-summary hash. The run provenance reported a dirty source tree; compare its recorded source-file hashes before attempting to reproduce it from Git alone.
