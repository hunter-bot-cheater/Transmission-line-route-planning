from transmission_path_planner.engineering_assessment import (
    engineering_assessment_record,
)


def test_assessment_keeps_unverified_engineering_requirements_explicit():
    report = engineering_assessment_record({
        "hard_constraint_status": "pass",
        "tower_span_status": "pass",
        "tower_site_slope_status": "interior_candidates_screened",
        "tower_site_pad_status": "site_pad_footprint_screened",
        "building_geometry_status": "pass",
        "clearance_profile_status": "dem_only_profile_screening",
        "clearance_profile_spans": [{"status": "sampled_clearance_ok"}],
        "station_endpoint_qualification": "exploratory_only_unverified_endpoints",
    }, station_task={"endpoint_qualification":
                     "exploratory_only_unverified_endpoints"})

    assert report["status"] == "not_evaluated"
    assert report["engineering_profile_source_integrity"] == "not_provided"
    assert report["configured_screens"]["dem_parabolic_sag_clearance"] == "screen_pass"
    assert "road_rail_water_building_clearance_by_crossing_class" in report[
        "not_evaluated_requirements"]
    assert "utility_verified_line_side_station_connections" in report[
        "not_evaluated_requirements"]


def test_observed_crossing_layers_never_imply_complete_assessment():
    report = engineering_assessment_record(
        {}, osm_layer_metadata={
            "roads": {"status": "available", "coverage_status": "features_observed"},
            "railways": {"status": "available", "coverage_status": "features_observed"},
        })

    assert report["crossing_data_layers"]["road"]["coverage_status"] == "features_observed"
    assert report["crossing_data_layers"]["road"]["crossing_assessment"].startswith(
        "not_evaluated")
    assert report["crossing_data_layers"]["water"]["source_status"] == "not_loaded"


def test_assessment_distinguishes_failed_incomplete_and_unrun_clearance_screens():
    failed = engineering_assessment_record({
        "clearance_profile_status": "dem_only_profile_screening",
        "clearance_profile_spans": [{"status": "fail"}],
    })
    incomplete = engineering_assessment_record({
        "clearance_profile_status": "unknown_station_elevation_datum_mismatch",
        "clearance_profile_spans": [],
    })
    unrun = engineering_assessment_record({"clearance_profile_status": "not_requested"})

    assert failed["configured_screens"]["dem_parabolic_sag_clearance"] == "screen_fail"
    assert incomplete["configured_screens"]["dem_parabolic_sag_clearance"] == "screen_incomplete"
    assert unrun["configured_screens"]["dem_parabolic_sag_clearance"] == "not_run"


def test_non_station_route_does_not_claim_missing_station_connection_as_a_failure():
    report = engineering_assessment_record({}, station_task=None)

    assert report["station_connection_points"] == "not_a_station_task"
    assert "utility_verified_line_side_station_connections" not in report[
        "not_evaluated_requirements"]
