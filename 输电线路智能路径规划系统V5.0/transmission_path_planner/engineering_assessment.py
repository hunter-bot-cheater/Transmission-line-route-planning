"""Honest engineering-screening summaries for exploratory route candidates."""


def engineering_assessment_record(route_metrics, station_task=None,
                                  profile_audit=None, osm_layer_metadata=None):
    """Separate configured raster screens from engineering checks not performed."""
    metrics = route_metrics or {}
    spans = metrics.get("clearance_profile_spans") or []
    profile_status = metrics.get("clearance_profile_status", "not_requested")
    profile_screen = "not_run"
    if profile_status in {
            "dem_only_profile_screening",
            "station_elevation_anchored_profile_screening"}:
        span_statuses = [span.get("status", "unknown") for span in spans]
        if span_statuses and all(status == "sampled_clearance_ok"
                                 for status in span_statuses):
            profile_screen = "screen_pass"
        elif any(status == "fail" for status in span_statuses):
            profile_screen = "screen_fail"
        else:
            profile_screen = "screen_incomplete"
    elif profile_status != "not_requested":
        profile_screen = "screen_incomplete"

    endpoint_qualification = (
        station_task.get("endpoint_qualification") if station_task else None
    ) or metrics.get("station_endpoint_qualification")
    if station_task:
        endpoint_status = (
            "declared_verified" if endpoint_qualification ==
            "declared_verified_line_connection_points" else "unverified_or_exploratory")
    else:
        endpoint_status = "not_a_station_task"

    if profile_audit is None:
        profile_audit = {
            "status": "not_provided",
            "engineering_feasibility": "not_evaluated",
        }
    osm_layer_metadata = osm_layer_metadata or {}
    crossing_layers = {}
    for crossing_class, layer_name in (
            ("road", "roads"), ("rail", "railways"),
            ("water", "water"), ("building", "buildings")):
        layer = osm_layer_metadata.get(layer_name, {})
        crossing_layers[crossing_class] = {
            "source_status": layer.get("status", "not_loaded"),
            "coverage_status": layer.get("coverage_status", "not_checked"),
            "crossing_assessment": "not_evaluated_no_verified_object_elevations",
        }

    profile_status = profile_audit.get("status", "not_provided")
    if profile_status == "not_provided":
        source_profile_status = "not_provided"
    else:
        source_profile_status = "hashes_checked_authority_and_applicability_unverified"

    profile_complete = (
        profile_status == "profile_record_complete_local_source_hashes_match")
    not_evaluated = [
        "conductor_specific_catenary_and_load_case_integration",
        "road_rail_water_building_clearance_by_crossing_class",
        "tower_structure_foundation_and_geotechnical_capacity",
        "right_of_way_land_rights_and_permits",
        "power_flow_and_contingency_study",
    ]
    if not profile_complete:
        not_evaluated.insert(0, "voltage_tower_conductor_and_load_case_profile")
    else:
        not_evaluated.append("profile_source_authority_and_line_applicability_review")

    return {
        "status": "not_evaluated",
        "engineering_profile_source_integrity": source_profile_status,
        "engineering_profile_audit": profile_audit,
        "station_connection_points": endpoint_status,
        "crossing_data_layers": crossing_layers,
        "configured_screens": {
            "hard_constraints": metrics.get("hard_constraint_status", "not_evaluated"),
            "tower_span_length": metrics.get("tower_span_status", "not_evaluated"),
            "tower_site_slope": metrics.get("tower_site_slope_status", "not_evaluated"),
            "tower_site_pad": metrics.get("tower_site_pad_status", "not_evaluated"),
            "tower_site_pad_relief": metrics.get(
                "tower_site_pad_relief_status", "not_evaluated"),
            "building_geometry": metrics.get("building_geometry_status", "not_evaluated"),
            "dem_parabolic_sag_clearance": profile_screen,
        },
        "not_evaluated_requirements": not_evaluated + (
            ["utility_verified_line_side_station_connections"]
             if station_task and endpoint_status != "declared_verified" else []),
        "interpretation": (
            "Configured raster screens describe exploratory candidate screening only; "
            "they do not establish engineering feasibility."),
    }
