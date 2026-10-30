import hashlib
import json

import pytest

from transmission_path_planner.engineering_profile import (
    CROSSING_CLASSES,
    load_engineering_profile,
)


def _write_profile(root):
    evidence = root / "utility-standard.txt"
    evidence.write_text("test evidence only", encoding="utf-8")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    source = {
        "id": "owner-standard",
        "issuer": "utility",
        "title": "Test evidence",
        "revision": "rev-a",
        "section": "section 1",
        "page": "1",
        "url": "https://example.invalid/test",
        "local_file": evidence.name,
        "sha256": digest,
    }
    profile = {
        "schema_version": 1,
        "profile_id": "fixture-profile",
        "line": {
            "voltage_kv": 161,
            "voltage_source_id": source["id"],
            "tower_family": "fixture-tower",
            "tower_family_source_id": source["id"],
            "conductor_id": "fixture-conductor",
            "conductor_source_id": source["id"],
            "load_cases": [
                {"id": "maximum_temperature", "source_id": source["id"]},
                {"id": "wind_ice", "source_id": source["id"]},
            ],
        },
        "vertical_datum": {"name": "fixture-datum", "source_id": source["id"]},
        "span_limits": {"min_m": 80, "max_m": 500, "source_id": source["id"]},
        "tower_site": {"max_slope_deg": 25, "pad_radius_m": 12,
                       "source_id": source["id"]},
        "sag_cases": [
            {"load_case": "maximum_temperature", "attachment_height_m": 22,
             "max_sag_m": 12, "source_id": source["id"]},
            {"load_case": "wind_ice", "attachment_height_m": 23,
             "max_sag_m": 10, "source_id": source["id"]},
        ],
        "clearance_rules": {
            name: {"minimum_m": 10, "source_id": source["id"]}
            for name in CROSSING_CLASSES
        },
        "sources": [source],
    }
    path = root / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path, profile


def test_complete_profile_records_hash_integrity_without_claiming_authority(tmp_path):
    path, _ = _write_profile(tmp_path)

    profile, audit = load_engineering_profile(path)

    assert profile["profile_id"] == "fixture-profile"
    assert audit["status"] == "profile_record_complete_local_source_hashes_match"
    assert audit["source_files"][0]["integrity_status"] == "match"
    assert audit["authority_and_applicability"] == "not_independently_verified"
    assert audit["engineering_feasibility"] == "not_evaluated"
    assert audit["route_parameters_applied"] is False


def test_profile_requires_every_crossing_class(tmp_path):
    path, profile = _write_profile(tmp_path)
    del profile["clearance_rules"]["rail"]
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="missing classes: rail"):
        load_engineering_profile(path)


def test_profile_rejects_source_hash_mismatch(tmp_path):
    path, profile = _write_profile(tmp_path)
    profile["sources"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_engineering_profile(path)


def test_profile_rejects_unresolved_source_reference(tmp_path):
    path, profile = _write_profile(tmp_path)
    profile["span_limits"]["source_id"] = "missing-source"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown source"):
        load_engineering_profile(path)


def test_profile_rejects_non_finite_engineering_parameter(tmp_path):
    path, profile = _write_profile(tmp_path)
    profile["sag_cases"][0]["max_sag_m"] = float("nan")
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="must be finite"):
        load_engineering_profile(path)


def test_profile_requires_sources_for_line_identity_and_load_cases(tmp_path):
    path, profile = _write_profile(tmp_path)
    del profile["line"]["conductor_source_id"]
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="line.conductor_source_id"):
        load_engineering_profile(path)

    profile["line"]["conductor_source_id"] = "owner-standard"
    del profile["line"]["load_cases"][0]["source_id"]
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match=r"line.load_cases\[0\].source_id"):
        load_engineering_profile(path)


def test_profile_rejects_source_path_escape(tmp_path):
    path, profile = _write_profile(tmp_path)
    profile["sources"][0]["local_file"] = "../outside.txt"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="stay under the profile directory"):
        load_engineering_profile(path)
