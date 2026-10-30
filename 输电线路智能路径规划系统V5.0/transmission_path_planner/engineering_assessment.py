"""Honest engineering-screening summaries for exploratory route candidates."""


def engineering_assessment_record(route_metrics, station_task=None):
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

    return {
        "status": "not_evaluated",
        "source_verified_engineering_profile": "missing",
        "station_connection_points": endpoint_status,
        "configured_screens": {
            "hard_constraints": metrics.get("hard_constraint_status", "not_evaluated"),
            "tower_span_length": metrics.get("tower_span_status", "not_evaluated"),
            "tower_site_slope": metrics.get("tower_site_slope_status", "not_evaluated"),
            "tower_site_pad": metrics.get("tower_site_pad_status", "not_evaluated"),
            "building_geometry": metrics.get("building_geometry_status", "not_evaluated"),
            "dem_parabolic_sag_clearance": profile_screen,
        },
        "not_evaluated_requirements": [
            "voltage_tower_conductor_and_load_case_profile",
            "conductor_specific_catenary_and_sag_envelope",
            "road_rail_water_building_clearance_by_crossing_class",
            "tower_structure_foundation_and_geotechnical_capacity",
            "right_of_way_land_rights_and_permits",
            "power_flow_and_contingency_study",
        ] + (["utility_verified_line_side_station_connections"]
             if station_task and endpoint_status != "declared_verified" else []),
        "interpretation": (
            "Configured raster screens describe exploratory candidate screening only; "
            "they do not establish engineering feasibility."),
    }
