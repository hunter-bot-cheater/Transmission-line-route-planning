# Engineering profile v1

Use `--engineering-profile-json profile.json` with `--experimental-tower-graph` to provide auditable inputs for a tower-graph candidate. The file is JSON with `schema_version: 1` and these fields:

- `profile_id`; `line` with `voltage_kv`, `tower_family`, and `conductor_id`, each paired with a `*_source_id`; `load_cases` is a list of unique `{id, source_id}` records.
- `vertical_datum` with `name` and `source_id`.
- `span_limits` with `min_m`, `max_m`, and `source_id`.
- `tower_site` with `max_slope_deg`, `pad_radius_m`, and `source_id`.
- `sag_cases`, one per load case, each with `load_case`, `attachment_height_m`, `max_sag_m`, and `source_id`.
- `clearance_rules` with `terrain`, `road`, `rail`, `water`, and `building`; each rule has `minimum_m` and `source_id`.
- `sources`, each with unique `id`, `issuer`, `title`, `revision`, `section`, `page`, `url`, `local_file`, and `sha256`.

Each `local_file` is a relative path to a local copy of the cited source document, kept under the profile file's directory. The loader verifies the file hash and each parameter's source ID. It rejects missing crossing classes, invalid numbers, unresolved source IDs, path escapes, and mismatched hashes.

For tower-graph runs, the profile supplies span limits, candidate-site slope and pad radius, and a conservative scalar DEM screen using the lowest attachment height, highest sag, and terrain clearance across listed load cases. Explicit matching CLI values are accepted; conflicting values fail before routing. Road, rail, water, and building clearance values are recorded in the profile but are not applied because current route data do not provide independently complete crossing coverage and object elevations. OSM feature observations are included in output as coverage metadata only.

The audit status `profile_record_complete_local_source_hashes_match` proves input completeness and cited local file integrity only. It does not verify issuer authority, design applicability, utility approval, conductor catenary, structure/foundation capacity, permits, or power-system reliability. Those items remain `not_evaluated`; the route must not be described as engineered or approved.
