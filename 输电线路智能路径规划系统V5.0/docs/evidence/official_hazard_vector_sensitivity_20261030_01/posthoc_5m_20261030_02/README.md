# Official hazard SHP/MVT route sensitivity: post-hoc geometry comparison

- Candidates: 12 per arm, 24 route/reference comparisons.
- Metric version: 7; arc-length sampling: 5 m.
- The two directional 100 m coverage values and their harmonic-mean F1 are descriptive spatial metrics; this report applies no coverage pass/fail gate.
- Both arms' non-source planning arguments match: True.

## Median route geometry results

| Arm | planned→reference within 100 m | reference→planned within 100 m | F1 within 100 m | symmetric P95 | symmetric P99 |
|---|---:|---:|---:|---:|---:|
| MVT | 3.08% | 3.31% | 3.19% | 5.11 km | 5.40 km |
| SHP | 3.08% | 3.31% | 3.19% | 5.13 km | 5.40 km |

- Median MVT↔SHP pairwise symmetric P95: 0.00 km.
- All route generation occurred on the server. Reference geometries were opened only after both candidate freezes and every route hash were verified.
- The two arms use the same fixed, historical benchmark endpoints; those endpoints are unverified and share their source with the references.
- All 24 routes are exploratory terrain-screening centerlines. Engineering feasibility is not evaluated; no tower-span, sag, clearance, or permit review is implied.
- SHP/MVT is a hazard-source representation sensitivity test. It can show whether encoding/rasterization changes the chosen geometry; it does not by itself show an accuracy improvement.
- The 5 m measurements are dense sampled approximations, not exact continuous Hausdorff bounds. Fréchet and endpoint metrics are unavailable for some disconnected geometries.

## Files

- `12_route_overlay_mvt_vs_shp.png`: all cases, reference with both planned routes.
- `B12-*_error_map.png`: two-way spatial error maps for each case.
- `route_similarity_posthoc.json` / `.csv`: full per-route metrics.
- `mvt_shp_pairwise_dispersion.csv`: route-to-route changes between arms.
- `reference_freeze_manifest.json` and `posthoc_manifest.json`: input hashes and run provenance.
