#!/usr/bin/env bash
set -euo pipefail

JOB_ROOT=/data1/B24041626/pp/moi_swc_zones_115_20261030_01/ab_run_20261030_01
SOURCE_ROOT=/data1/B24041626/pp/official_hazard_vector_sensitivity_20261030_01
BUNDLE="$SOURCE_ROOT/bundle"
STATIONS="$SOURCE_ROOT/inputs_benchmark12_v2/stations.csv"
TASKS="$SOURCE_ROOT/inputs_benchmark12_v2/route_tasks.csv"
DEM=/data1/B24041626/pp/dtm_2026_planner_input_20261028_01/dtm_2026_20m_geographic_4326.tif
BASE_CONSTRAINTS=/data1/B24041626/pp/work/data/official_constraints_20261023.geojson
SWC_CONSTRAINTS=/data1/B24041626/pp/moi_swc_zones_115_20261030_01/overlay_20261030_03/constraints_with_swc_zones.geojson
HAZARDS=/data1/B24041626/pp/work/data/official_sources/ardswc_115_mvt_z10_20261023
LAND_MASK=/data1/B24041626/pp/landmask_route_fix_20261026/inputs/county_boundaries_1140318.geojson
POPULATION=/data1/B24041626/pp/terrain_worldpop_sensitivity_4096_20261028/inputs/twn_pop_2025_CN_100m_R2025A_v1.tif

cd "$(dirname "$0")"
mkdir -p "$JOB_ROOT"
CONTROL_OUT="$JOB_ROOT/results_control_20261030_01"
SWC_OUT="$JOB_ROOT/results_swc_soft_20261030_01"
if [[ -e "$CONTROL_OUT" || -e "$SWC_OUT" || -e "$JOB_ROOT/pre_run_freeze.json" \
      || -e "$JOB_ROOT/pre_run_inputs.sha256" ]]; then
  printf 'refusing to overwrite an existing candidate, output, or freeze file\n' >&2
  exit 2
fi
for path in "$BUNDLE/bundle_manifest.json" "$STATIONS" "$TASKS" "$DEM" \
            "$BASE_CONSTRAINTS" "$SWC_CONSTRAINTS" "$HAZARDS/manifest.json" \
            "$LAND_MASK" "$POPULATION"; do
  [[ -f "$path" ]] || { printf 'required input missing: %s\n' "$path" >&2; exit 3; }
done
mkdir -p "$JOB_ROOT/cache/torch" "$JOB_ROOT/cache/matplotlib" "$JOB_ROOT/tmp"

sha256sum "$STATIONS" "$TASKS" "$BASE_CONSTRAINTS" "$SWC_CONSTRAINTS" \
  "$BUNDLE/bundle_manifest.json" > "$JOB_ROOT/pre_run_inputs.sha256"
cat > "$JOB_ROOT/pre_run_freeze.json" <<EOF
{
  "schema_version": 1,
  "status": "frozen_before_route_generation",
  "created_at": "$(date -Is)",
  "case_count": 12,
  "endpoint_qualification": "unverified; reference-derived or shared; exploratory only",
  "reference_geometry_read_before_freeze": false,
  "planning_prior": "no historical route geometry passed to route search",
  "baseline_constraint": "$BASE_CONSTRAINTS",
  "treatment_constraint": "$SWC_CONSTRAINTS",
  "treatment_delta": "one dissolved official 115-year designated SWC polygon, mode=soft, weight=1.0; no hard-mask changes",
  "planning_source": "$BUNDLE",
  "dem": "$DEM",
  "hazards": "$HAZARDS",
  "gpu_policy": "CUDA_VISIBLE_DEVICES empty; CPU run",
  "common_arguments": ["--max-side", "6400", "--no-ppp", "--official-hazard-weight", "1.0", "--settlement-weight", "2.0", "--ridge-penalty-weight", "0.1", "--landcover-penalty-weight", "0.0", "--grade-penalty-weight", "1.0", "--terrain-slope-weight", "0.75", "--roughness-weight", "0.25", "--astar-heuristic-weight", "1.0", "--building-clearance-m", "0", "--window-margin-deg", "0.3", "--allow-unverified-endpoints", "--population-weight", "1.0", "--population-nodata-policy", "maximum"]
}
EOF

common=(
  --dem "$DEM"
  --stations-csv "$STATIONS"
  --route-tasks-csv "$TASKS"
  --max-side 6400
  --no-ppp
  --official-hazard-weight 1.0
  --settlement-weight 2.0
  --ridge-penalty-weight 0.1
  --landcover-penalty-weight 0.0
  --grade-penalty-weight 1.0
  --terrain-slope-weight 0.75
  --roughness-weight 0.25
  --astar-heuristic-weight 1.0
  --building-clearance-m 0
  --window-margin-deg 0.3
  --allow-unverified-endpoints
  --osm-cache-dir /data1/B24041626/pp/work/osm
  --land-mask-geojson "$LAND_MASK"
  --population-raster "$POPULATION"
  --population-weight 1.0
  --population-nodata-policy maximum
)

run_candidate() {
  local label="$1" constraint="$2" out="$3"
  if [[ -e "$out" ]]; then
    printf 'refusing to overwrite existing output directory: %s\n' "$out" >&2
    return 2
  fi
  mkdir "$out"
  set +e
  env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME="$JOB_ROOT/cache" TORCH_HOME="$JOB_ROOT/cache/torch" \
    MPLCONFIGDIR="$JOB_ROOT/cache/matplotlib" TMPDIR="$JOB_ROOT/tmp" \
    PYTHONPATH="$BUNDLE" /usr/bin/time -v /data1/B24041626/pp/venv/bin/python \
      -m transmission_path_planner "${common[@]}" \
      --constraint-geojson "$constraint" \
      --official-hazards-dir "$HAZARDS" \
      --out-dir "$out" > "$out/stdout.log" 2> "$out/stderr_time.log"
  local status=$?
  set -e
  printf '%s\n' "$status" > "$out/exit_status.txt"
  printf '%s exit=%s output=%s\n' "$label" "$status" "$out"
  return "$status"
}

run_candidate control "$BASE_CONSTRAINTS" "$CONTROL_OUT"
run_candidate swc_soft "$SWC_CONSTRAINTS" "$SWC_OUT"
