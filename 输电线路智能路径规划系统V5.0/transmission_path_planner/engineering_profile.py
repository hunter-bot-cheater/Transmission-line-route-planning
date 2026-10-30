"""Validate traceable engineering inputs without claiming design approval."""

import hashlib
import json
import math
from pathlib import Path


CROSSING_CLASSES = ("terrain", "road", "rail", "water", "building")


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"engineering profile {name} must be an object")
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"engineering profile {name} must be a non-empty string")
    return value.strip()


def _number(value, name, *, minimum=0.0, exclusive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"engineering profile {name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (value <= minimum if exclusive else value < minimum):
        operator = ">" if exclusive else ">="
        raise ValueError(f"engineering profile {name} must be finite and {operator} {minimum}")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_engineering_profile(path):
    """Load v1 profile inputs and verify references to local source documents.

    Hash verification establishes file integrity only. It does not establish
    source authority, engineering applicability, or route feasibility.
    """
    profile_path = Path(path).resolve()
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read engineering profile: {exc}") from exc
    profile = _object(profile, "root")
    if profile.get("schema_version") != 1:
        raise ValueError("engineering profile schema_version must be 1")
    profile_id = _text(profile.get("profile_id"), "profile_id")
    line = _object(profile.get("line"), "line")
    voltage_kv = _number(line.get("voltage_kv"), "line.voltage_kv", exclusive=True)
    _text(line.get("tower_family"), "line.tower_family")
    _text(line.get("conductor_id"), "line.conductor_id")
    line_source_ids = {
        field: _text(line.get(field), f"line.{field}")
        for field in ("voltage_source_id", "tower_family_source_id",
                      "conductor_source_id")
    }
    load_cases = line.get("load_cases")
    if not isinstance(load_cases, list) or not load_cases:
        raise ValueError("engineering profile line.load_cases must be a non-empty list")
    load_case_ids = []
    load_case_source_ids = []
    for index, item in enumerate(load_cases):
        case = _object(item, f"line.load_cases[{index}]")
        load_case_ids.append(_text(case.get("id"), f"line.load_cases[{index}].id"))
        load_case_source_ids.append(_text(
            case.get("source_id"), f"line.load_cases[{index}].source_id"))
    if len(set(load_case_ids)) != len(load_case_ids):
        raise ValueError("engineering profile line.load_cases must be unique")

    sources = profile.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("engineering profile sources must be a non-empty list")
    source_ids = set()
    source_records = []
    profile_root = profile_path.parent
    for index, value in enumerate(sources):
        source = _object(value, f"sources[{index}]")
        source_id = _text(source.get("id"), f"sources[{index}].id")
        if source_id in source_ids:
            raise ValueError(f"duplicate engineering profile source id: {source_id}")
        source_ids.add(source_id)
        for field in ("issuer", "title", "revision", "section", "page", "url"):
            _text(source.get(field), f"sources[{index}].{field}")
        local_file = Path(_text(source.get("local_file"), f"sources[{index}].local_file"))
        if local_file.is_absolute() or ".." in local_file.parts:
            raise ValueError("engineering profile source local_file must stay under the profile directory")
        evidence_path = profile_root / local_file
        try:
            resolved_evidence_path = evidence_path.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"engineering profile source file unavailable: {local_file}") from exc
        if (evidence_path.is_symlink() or not evidence_path.is_file()
                or not resolved_evidence_path.is_relative_to(profile_root)):
            raise ValueError(f"engineering profile source file missing or linked: {local_file}")
        expected_hash = _text(source.get("sha256"), f"sources[{index}].sha256").lower()
        if (len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)):
            raise ValueError(f"engineering profile source {source_id} has invalid sha256")
        actual_hash = _sha256(evidence_path)
        if actual_hash != expected_hash:
            raise ValueError(f"engineering profile source hash mismatch: {source_id}")
        source_records.append({"id": source_id, "local_file": str(local_file),
                               "sha256": actual_hash, "integrity_status": "match"})

    def require_source(value, name):
        source_id = _text(value, name)
        if source_id not in source_ids:
            raise ValueError(f"engineering profile {name} references unknown source: {source_id}")

    for field, source_id in line_source_ids.items():
        require_source(source_id, f"line.{field}")
    for index, source_id in enumerate(load_case_source_ids):
        require_source(source_id, f"line.load_cases[{index}].source_id")

    datum = _object(profile.get("vertical_datum"), "vertical_datum")
    _text(datum.get("name"), "vertical_datum.name")
    require_source(datum.get("source_id"), "vertical_datum.source_id")

    span = _object(profile.get("span_limits"), "span_limits")
    minimum_span = _number(span.get("min_m"), "span_limits.min_m")
    maximum_span = _number(span.get("max_m"), "span_limits.max_m", exclusive=True)
    if maximum_span <= minimum_span:
        raise ValueError("engineering profile span_limits.max_m must exceed min_m")
    require_source(span.get("source_id"), "span_limits.source_id")

    site = _object(profile.get("tower_site"), "tower_site")
    maximum_slope = _number(site.get("max_slope_deg"), "tower_site.max_slope_deg", exclusive=True)
    if maximum_slope > 90.0:
        raise ValueError("engineering profile tower_site.max_slope_deg must be <= 90")
    _number(site.get("pad_radius_m"), "tower_site.pad_radius_m", exclusive=True)
    require_source(site.get("source_id"), "tower_site.source_id")

    sag_cases = profile.get("sag_cases")
    if not isinstance(sag_cases, list) or not sag_cases:
        raise ValueError("engineering profile sag_cases must be a non-empty list")
    sag_by_case = {}
    for index, value in enumerate(sag_cases):
        case = _object(value, f"sag_cases[{index}]")
        case_id = _text(case.get("load_case"), f"sag_cases[{index}].load_case")
        if case_id in sag_by_case:
            raise ValueError(f"duplicate engineering profile sag case: {case_id}")
        _number(case.get("attachment_height_m"), f"sag_cases[{index}].attachment_height_m")
        _number(case.get("max_sag_m"), f"sag_cases[{index}].max_sag_m")
        require_source(case.get("source_id"), f"sag_cases[{index}].source_id")
        sag_by_case[case_id] = case
    if set(sag_by_case) != set(load_case_ids):
        raise ValueError("engineering profile sag_cases must match line.load_cases exactly")

    clearances = _object(profile.get("clearance_rules"), "clearance_rules")
    missing_classes = sorted(set(CROSSING_CLASSES) - set(clearances))
    if missing_classes:
        raise ValueError("engineering profile clearance_rules missing classes: "
                         + ", ".join(missing_classes))
    clearance_summary = {}
    for crossing_class in CROSSING_CLASSES:
        rule = _object(clearances[crossing_class], f"clearance_rules.{crossing_class}")
        clearance_summary[crossing_class] = _number(
            rule.get("minimum_m"), f"clearance_rules.{crossing_class}.minimum_m")
        require_source(rule.get("source_id"),
                       f"clearance_rules.{crossing_class}.source_id")

    return profile, {
        "status": "profile_record_complete_local_source_hashes_match",
        "profile_id": profile_id,
        "profile_path": str(profile_path),
        "profile_sha256": _sha256(profile_path),
        "voltage_kv": voltage_kv,
        "vertical_datum": datum["name"],
        "source_files": source_records,
        "crossing_classes_defined": list(CROSSING_CLASSES),
        "authority_and_applicability": "not_independently_verified",
        "route_parameters_applied": False,
        "engineering_feasibility": "not_evaluated",
    }
