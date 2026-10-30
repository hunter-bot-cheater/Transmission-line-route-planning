# Route engineering assessment output

Each `最终路线` record now includes `engineering_assessment` in the route metrics JSON.

- `status` remains `not_evaluated` until a source-verified line engineering profile and the required crossing/structure checks are implemented.
- `configured_screens` reports only checks present on the selected candidate. The DEM parabolic sag result is labeled `screen_pass`, `screen_fail`, `screen_incomplete`, or `not_run`; it is not a clearance approval.
- `not_evaluated_requirements` lists unresolved engineering evidence, including voltage/tower/conductor/load case, crossing-specific clearance, tower foundation, right-of-way, and power-system studies.
- For station tasks, `station_connection_points` reflects whether endpoints were declared as verified line-side connection points. A declared status is still an input assertion, not an independent audit of the source.
- Optional `--engineering-profile-json` inputs are reported with local source-file hash status and parameter fields applied. This does not verify issuer authority or line applicability. The profile contract is described in [engineering_profile_v1.md](engineering_profile_v1.md).
- `crossing_data_layers` reports OSM layer loading and observed coverage status. An observed feature extent is not proof of completeness, and missing object elevations keep crossing clearances `not_evaluated`.

The summary is deliberately diagnostic. The optional profile can supply tower-graph span, site, and terrain-screen values; its non-terrain crossing rules do not change route selection. No profile or screen changes engineering feasibility to `pass`.
