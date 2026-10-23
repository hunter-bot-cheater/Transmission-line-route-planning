"""Synthetic 8-neighbor direction and smoothing-tolerance benchmark.

No historical routes or external data are read. Run from the repository root:
    D:/python/python.exe tools/benchmark_uniform_grid_direction_bias.py
Add ``--csv outputs/uniform_grid_direction_bias.csv`` for per-case results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None

from transmission_path_planner.cli import attach_input_endpoints
from transmission_path_planner.geometry import rowcol_to_geo
from transmission_path_planner.metrics import astar_path_objective, path_cost_integral
from transmission_path_planner.methods import astar_search
from transmission_path_planner import smoothing

PIXEL_M, LENGTH_PX = 35.0, 72.0
SHAPE, PHASES, ANGLES = (180, 180), (0.05, 0.25, 0.50, 0.75, 0.95), range(46)
# RDP-only profiles reproduce the previous post-processing path. The final
# profile invokes the current production constraint-guarded direct shortcut.
PROFILES = (("rdp45", 45.0, 90.0), ("rdp75", 75.0, 150.0),
            ("rdp150", 150.0, 300.0), ("rdp300", 300.0, 600.0),
            ("rdp600", 600.0, 1200.0), ("rdp1200", 1200.0, 2400.0),
            ("shortcut", 45.0, 90.0))


def _length_m(route, geod):
    return sum(geod.inv(*a, *b)[2] for a, b in zip(route, route[1:]))


def _cost_m(route, dem, transform, cost, spacing):
    result = path_cost_integral(route, dem, transform, cost, 0.0, 0.0, spacing)
    if result["status"] != "measured":
        raise RuntimeError(f"route objective unavailable: {result['status']}")
    return result["total_cost_meter"]


def _smooth_profile(raw, transform, hard, cost, dem, spacing, tolerance,
                    sample_spacing, use_shortcut=False):
    """Compare the prior RDP-only pipeline with the current guarded shortcut."""
    if use_shortcut:
        chosen = smoothing.smooth_path_with_constraints(
            raw, transform, hard, cost, dem, 0.0, 0.0, spacing)
        proposed = chosen
        proposed_hard = smoothing.path_hard_violations(proposed, transform, hard)
        proposed_cost = (_cost_m(proposed, dem, transform, cost, spacing)
                         if proposed_hard == 0 else None)
        return chosen, proposed, proposed_hard, proposed_cost, "accepted"

    original_rdp, original_smooth = smoothing._rdp, smoothing.smooth_path

    def rdp(points, _production_epsilon):
        return original_rdp(points, tolerance)

    with patch.object(smoothing, "_rdp", side_effect=rdp):
        proposed = original_smooth(raw, spacing_m=sample_spacing)
    snapped = smoothing.snap_to_valid(proposed, transform, hard)
    proposed_hard = (None if snapped is None else
                     smoothing.path_hard_violations(snapped, transform, hard))
    proposed_cost = (_cost_m(snapped, dem, transform, cost, spacing)
                     if snapped is not None and proposed_hard == 0 else None)
    if snapped is None:
        chosen = raw
        reason = "snap_failed"
    elif proposed_hard:
        chosen = raw
        reason = "hard_rejected"
    elif proposed_cost is not None:
        raw_cost = _cost_m(raw, dem, transform, cost, spacing)
        if proposed_cost <= raw_cost + max(abs(raw_cost), 1.0) * 1e-9:
            chosen = snapped
            reason = "accepted"
        else:
            chosen = raw
            reason = "cost_rejected"
    else:
        chosen = raw
        reason = "objective_unavailable"
    return chosen, proposed, proposed_hard, proposed_cost, reason


def _case(angle, row_phase, col_phase, transform, geod, cost, dem, hard, spacing):
    theta = math.radians(angle)
    r0, c0 = 35.0 + row_phase, 20.0 + col_phase
    r1, c1 = r0 + LENGTH_PX * math.sin(theta), c0 + LENGTH_PX * math.cos(theta)
    start, end = (math.floor(r0), math.floor(c0)), (math.floor(r1), math.floor(c1))
    slon, slat = transform * (c0, r0)
    elon, elat = transform * (c1, r1)
    start_ll, end_ll = (slat, slon), (elat, elon)
    path = astar_search(cost, start, end, pixel_spacing=spacing)
    if not path:
        raise RuntimeError(f"no route at angle={angle}, phase={(row_phase, col_phase)}")
    raw = [rowcol_to_geo(r, c, transform) for r, c in path]
    raw_cost = _cost_m(raw, dem, transform, cost, spacing)
    raw_obj = astar_path_objective(path, cost, dem, spacing, grade_penalty_weight=0.0)
    if raw_obj["status"] != "measured":
        raise RuntimeError(f"raw objective unavailable: {raw_obj['status']}")

    input_direct = geod.inv(slon, slat, elon, elat)[2]
    grid_direct = geod.inv(*raw[0], *raw[-1])[2]
    raw_final = attach_input_endpoints(raw, start_ll, end_ll)
    raw_inflation = 100 * (_length_m(raw_final, geod) / input_direct - 1)
    rows = []
    for name, tolerance, sample_spacing in PROFILES:
        chosen, proposed, proposed_hard, proposed_cost, reason = _smooth_profile(
            raw, transform, hard, cost, dem, spacing, tolerance, sample_spacing,
            use_shortcut=(name == "shortcut"))
        final = attach_input_endpoints(chosen, start_ll, end_ll)
        final_hard = smoothing.path_hard_violations(final, transform, hard)
        rows.append({
            "profile": name, "tolerance_m": tolerance, "sample_spacing_m": sample_spacing,
            "angle_deg": angle, "row_phase": row_phase, "col_phase": col_phase,
            "raw_inflation_pct": raw_inflation,
            "chosen_inflation_pct": 100 * (_length_m(final, geod) / input_direct - 1),
            "raw_center_inflation_pct": 100 * (_length_m(raw, geod) / grid_direct - 1),
            "proposed_center_inflation_pct": 100 * (_length_m(proposed, geod) / grid_direct - 1),
            "raw_astar_objective": raw_obj["total_cost"],
            "raw_continuous_objective": raw_cost,
            "proposed_continuous_objective": proposed_cost,
            "objective_delta": (None if proposed_cost is None else proposed_cost - raw_cost),
            "accepted": reason == "accepted", "rejection_reason": reason,
            "proposed_hard_violations": proposed_hard,
            "final_hard_violations": final_hard,
            "raw_points": len(raw), "proposed_points": len(proposed),
        })
    return rows


def _stats(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {label: round(float(fn(values)), 6) for label, fn in (
        ("mean_pct", np.mean), ("median_pct", lambda x: np.percentile(x, 50)),
        ("p95_pct", lambda x: np.percentile(x, 95)), ("max_pct", np.max),
    )}


def run():
    from affine import Affine
    from pyproj import Geod

    lat = 25.0
    dcol = PIXEL_M / (111_320 * math.cos(math.radians(lat)))
    drow = PIXEL_M / 110_574
    transform = Affine(dcol, 0, 120 - SHAPE[1] * dcol / 2, 0, -drow,
                       lat + SHAPE[0] * drow / 2)
    cost, dem = np.ones(SHAPE), np.zeros(SHAPE)
    hard, spacing, geod = np.ones(SHAPE, np.uint8), (PIXEL_M, PIXEL_M), Geod(ellps="WGS84")
    rows = [row for angle in ANGLES for rp in PHASES for cp in PHASES
            for row in _case(angle, rp, cp, transform, geod, cost, dem, hard, spacing)]
    summaries = {}
    for name, tolerance, sample_spacing in PROFILES:
        group = [row for row in rows if row["profile"] == name]
        worst = max(group, key=lambda row: row["chosen_inflation_pct"])
        summaries[name] = {
            "rdp_tolerance_m": tolerance, "resample_spacing_m": sample_spacing,
            "raw_inflation_pct": _stats(group, "raw_inflation_pct"),
            "chosen_inflation_pct": _stats(group, "chosen_inflation_pct"),
            "worst_angle_phase": {key: worst[key] for key in (
                "angle_deg", "row_phase", "col_phase", "chosen_inflation_pct")},
            "accepted_cases": sum(row["accepted"] for row in group),
            "rejected_cases": [
                {key: row[key] for key in ("angle_deg", "row_phase", "col_phase", "rejection_reason")}
                for row in group if not row["accepted"]
            ],
            "max_proposed_hard_violations": max(
                (row["proposed_hard_violations"] or 0) for row in group),
            "max_final_hard_violations": max(row["final_hard_violations"] for row in group),
            "cost_increases_over_0_001": sum(
                row["objective_delta"] is not None and row["objective_delta"] > 0.001
                for row in group
            ),
            "max_objective_delta": max(
                (row["objective_delta"] or 0.0) for row in group),
            "cases_over_5pct": sum(row["chosen_inflation_pct"] > 5 for row in group),
        }
    return {
        "synthetic_only": True, "angles_deg": [0, 45, 1],
        "phase_values_per_axis": PHASES, "cases_per_profile": len(ANGLES) * len(PHASES) ** 2,
        "grid": f"{SHAPE[0]}x{SHAPE[1]} at {PIXEL_M:g} m",
        "uniform_cost": 1.0, "flat_dem": True, "all_cells_traversable": True,
        "profiles": summaries,
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="optional per-case CSV output")
    args = parser.parse_args()
    summary, rows = run()
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.csv:
        print(f"\ncase_csv={args.csv.resolve()}")


if __name__ == "__main__":
    main()
