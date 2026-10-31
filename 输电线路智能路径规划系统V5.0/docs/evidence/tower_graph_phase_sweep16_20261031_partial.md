# Candidate-grid phase sweep: partial pairwise result

## Run identity

- Server experiment: `/data1/B24041626/pp/tower_graph_phase_sweep16_20261031_01`
- Source commit recorded in the server manifest: `aa470a1dda4f24e925ac9f5fda613b6ce780961d`
- DEM: `/data1/B24041626/pp/dtm_2026_planner_input_20261028_01/dtm_2026_20m_geographic_4326.tif`
- Case: `SHENMEI_DONGSHAN_EXPLORATORY`; endpoints are explicitly unverified station proxies.
- Planner reference geometry: not read during route search. The manifest says all 16 phases will be reported; no phase is selected using post-hoc route accuracy.
- At the partial capture, phases `(0,0)` and `(0,1)` had completed; phase `(0,2)` had started. The capture does not establish whether the server job later completed. A read-only SSH check on 2026-11-01 timed out, so current job and manifest status remain unverified.

## Pairwise route stability, phase (0,0) versus (0,1)

The two generated TowerGraph GeoJSON routes were projected to EPSG:3826. Each line was sampled at approximately 10 m intervals. For each sample, the nearest distance to the other route was computed in both directions; the metrics below summarize the pooled bidirectional distances.

| Metric | Result |
| --- | ---: |
| Projected route length, phase (0,0) | 47.573 km |
| Projected route length, phase (0,1) | 47.583 km |
| Absolute length difference | 10.2 m |
| Mean bidirectional sampled distance | 23.16 m |
| P95 / P99 bidirectional sampled distance | 59.54 m / 113.16 m |
| Maximum sampled distance | 172.08 m |
| Sampled route length within 100 m | 98.62% |
| Sampled route length within 500 m | 100.00% |

This is a **candidate-to-candidate grid-phase sensitivity result**, not a real-line accuracy score. It does not establish that either route resembles the as-built line. The real-route evaluation was not requested in either planning run, and engineering feasibility remains `not_evaluated`.

The planner configuration is exploratory: minimum span 0 m, maximum span 500 m, stride 4, no tower-site slope/pad screen, and no conductor-clearance profile. These values cannot be used as a construction-feasibility finding. A separate optional pad-relief screen was added locally after this server source snapshot; it is not part of these two runs.

## Phase interpretation and follow-up

For stride `s`, the union of candidate vertices across all `s²` phases equals the eligible `stride=1` vertex set. The phase-specific edge sets do not: each phase only connects vertices in its own row/column residue class. Rebuilding all legal edges after combining vertices is equivalent to a `stride=1` graph, but increases candidate vertices by about `s²` (16× for this experiment) and may sharply increase search cost. Therefore, do not treat the union of completed phase routes as a dense-graph solution.

Any no-reference phase comparison must use frozen common objective and constraints. Candidate-to-candidate distance clusters, site-use frequency, common objective cost, and fresh hard-constraint checks can describe stability; post-hoc real-line similarity must not select a phase. The two-phase result above is insufficient for a full sweep conclusion. Once server access returns, read this run's existing manifest before deciding whether to resume it; do not launch a duplicate sweep while its status is unknown.

Local synthetic regression: `pytest -q tests/test_experimental_tower_graph.py` completed with 37 passed on 2026-11-01. A separate synthetic phase diagnostic ran all 16 phases on the existing 51×61 illustrative surface with the same stride-4 graph and sampled-clearance configuration: 9 phases found routes, yielding 36 candidate pairs. The corrected metric samples segment midpoints and weights each distance by its represented arc length; pairwise bidirectional P95 distance has a 302.932 m median and 456.088 m maximum, while the median route-length fraction within 100 m is 53.01%. The earlier v1 output used unweighted samples and is retained only as a preliminary run. Both results show the previous two-phase pair was unusually stable for this synthetic case; neither measures Taiwan route accuracy. Reproduce with `D:\python\python.exe scripts/benchmark_tower_graph_phase_stability.py --output-dir docs\evidence\tower_graph_phase_synthetic_20261101_v3`. Raw phase and pair CSVs plus the input/config manifest are in that v3 directory.

## Per-phase evidence

- Phase `(0,0)` finished successfully in 1,545.9 s; its A* baseline was 50.455 km and TowerGraph route was 47.685 km in the planner's reported length metric.
- Phase `(0,1)` finished successfully in 1,467.2 s; its A* baseline was 50.455 km and TowerGraph route was 47.695 km in the planner's reported length metric. TowerGraph reported 119 intermediate tower sites, zero configured hard-mask violations, and span status `pass` only against the exploratory 0–500 m limits.
- Projected GeoJSON lengths above differ slightly from the planner's reported lengths because they are measured after reprojection to EPSG:3826; do not mix the two length methods in one comparison.

## Files and reproducibility

The server keeps `run_manifest.json`, `run_sweep.py`, per-phase `comparison_metrics.json`, `run_provenance.json`, and route GeoJSON/PNG outputs under the experiment directory. The server manifest is authoritative for the remaining phases. Do not interpret this partial pair as the sweep result or use its real-line similarity to select a phase.
