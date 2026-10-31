# Taiwan 115-year specific soil and water conservation zones

## Source and validation

- Official dataset: [115年度66區特定水土保持區範圍圖](https://data.gov.tw/dataset/176847), provided by Taiwan's Ministry of Agriculture.
- Download endpoint: `https://data.moa.gov.tw/GetOpenDataFile.aspx?FileType=ZIP&RID=87575&id=J80`
- Dataset page metadata was checked on 2026-10-30. It states 66 designated areas, SHP download, irregular updates, free use, and Open Government Data License v1.
- Downloaded archive: `data/official_sources/moi_soil_water_conservation_zones_115_20261030/source_1150922.zip`
- Archive SHA-256: `7d98b9aa05391f9649c312ac6792a5072539d569d156e3937e703cffd4827e66`
- Server-side data inspection confirmed 66 polygon features, EPSG:3826, 61 Polygon and 5 MultiPolygon features, and `DRSWC_Land=劃定` for all features. The category bounds are 120.3682–121.9348°E, 22.1007–25.1405°N; this is not a nationwide uniform layer.
- One source feature (zero-based row 48) was invalid. `shapely.make_valid` repaired it as a Polygon with relative area change about `1.53e-15`; the script records this operation in the manifest and refuses repairs that change area by more than 1%.
- The official resource contains 10 attributes. The preparation script checks schema for address-like field names before reading only the five listed non-address columns; it emits geometry and aggregate counts, not attribute records.
- Server structural verification passed on a 46×70 sample grid: the base/treatment had 140/141 features, hard masks were identical, and the SWC layer added a non-negative soft cost to 88 cells (summed weight 106).

## Planning role

The layer is a soft environmental and regulatory review signal only. It is not treated as a legal construction ban, a permit decision, a hard route exclusion, or engineering approval. The overlay dissolves the 66 designated polygons to avoid double-counting overlaps and assigns a configurable experimental soft penalty. The weight is a sensitivity-test setting, not an empirically calibrated engineering value.

Current benchmark endpoints and route references are unverified and share their source. Any route comparison using them remains exploratory and cannot establish independent real-world accuracy. No historical line geometry is used to create the overlay or plan candidate routes.

## Reproduction

`prepare_swc_overlay.py` copies the original constraint FeatureCollection, appends one dissolved RFC 7946 WGS84 soft-only polygon, and writes a manifest with base/source/output hashes. It refuses to reuse an output directory and never modifies the base constraints. The A/B candidate run is configured to compare this merged file with the original base file while keeping the other inputs and parameters fixed; the run provenance must confirm that before interpreting the result. Do not tune the weight against route-reference similarity.

After both server runs finish, `freeze_swc_candidates.py` validates the 12 route outputs per arm and hashes each final GeoJSON before any reference line geometry is opened. Then `score_swc_ab_posthoc.py` validates the freezes, provenance, argument equality and input manifest before reading reference geometries. The scorer normalizes only the expected `--constraint-geojson` path and output directory difference; all other arguments and all non-constraint planning input hashes must match. It refuses to score if a required input is unhashed, a shared input differs, or the treatment constraint hash is unchanged. It produces symmetric, sampled distance metrics and maps, including a 100 m threshold view. These are descriptive post-hoc comparisons, not pass/fail gates.

The intended server commands are:

```bash
python freeze_swc_candidates.py --experiment-root /data1/B24041626/pp/moi_swc_zones_115_20261030_01/ab_run_20261030_01
python score_swc_ab_posthoc.py \
  --experiment-root /data1/B24041626/pp/moi_swc_zones_115_20261030_01/ab_run_20261030_01 \
  --input-root /data1/B24041626/pp/official_hazard_vector_sensitivity_20261030_01 \
  --reference-dir /data1/B24041626/pp/work/benchmark_12 \
  --planner-root /data1/B24041626/pp/official_hazard_vector_sensitivity_20261030_01/bundle \
  --out-dir /data1/B24041626/pp/moi_swc_zones_115_20261030_01/posthoc_swc_20261030_01
```

The reference endpoints remain unverified and are shared with the benchmark source. Therefore the resulting geometry scores cannot establish independent accuracy or generalization. The planning runs also do not establish tower-span, conductor sag, clearance, foundation, right-of-way, permit, or grid-operability feasibility.
