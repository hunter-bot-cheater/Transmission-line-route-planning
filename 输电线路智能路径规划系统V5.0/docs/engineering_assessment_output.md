# Route engineering assessment output

Each `最终路线` record now includes `engineering_assessment` in the route metrics JSON.

- `status` remains `not_evaluated` until a source-verified line engineering profile and the required crossing/structure checks are implemented.
- `configured_screens` reports only checks present on the selected candidate. The DEM parabolic sag result is labeled `screen_pass`, `screen_fail`, `screen_incomplete`, or `not_run`; it is not a clearance approval.
- `not_evaluated_requirements` lists unresolved engineering evidence, including voltage/tower/conductor/load case, crossing-specific clearance, tower foundation, right-of-way, and power-system studies.
- For station tasks, `station_connection_points` reflects whether endpoints were declared as verified line-side connection points. A declared status is still an input assertion, not an independent audit of the source.

The summary is deliberately diagnostic. It does not change the route objective, candidate selection, or any engineering threshold.
