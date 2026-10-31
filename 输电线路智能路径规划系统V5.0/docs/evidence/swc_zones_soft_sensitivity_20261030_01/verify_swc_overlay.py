#!/usr/bin/env python3
"""Verify the SWC treatment only adds a rasterized soft cost, never hard masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rasterio.transform import from_bounds

from transmission_path_planner.osm import load_user_constraint_overlays


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-constraints", type=Path, required=True)
    parser.add_argument("--treatment-constraints", type=Path, required=True)
    args = parser.parse_args()

    base = read_json(args.base_constraints)
    treatment = read_json(args.treatment_constraints)
    assert len(treatment["features"]) == len(base["features"]) + 1
    assert treatment["features"][:-1] == base["features"]
    swc_feature = treatment["features"][-1]
    assert swc_feature["properties"]["mode"] == "soft"
    assert float(swc_feature["properties"]["weight"]) == 1.0
    assert swc_feature["properties"]["category"] == "designated_conservation_review"
    assert treatment["metadata"]["official_swc_zone_overlay"]["hard_exclusion_allowed"] is False

    shape = (46, 70)
    transform = from_bounds(119.9, 21.8, 122.2, 25.3, shape[1], shape[0])
    base_layers = load_user_constraint_overlays(args.base_constraints, transform, shape)
    treatment_layers = load_user_constraint_overlays(args.treatment_constraints, transform, shape)
    assert np.array_equal(base_layers["hard_mask"], treatment_layers["hard_mask"])
    delta = treatment_layers["soft_cost"] - base_layers["soft_cost"]
    assert np.all(delta >= -1e-6)
    assert np.any(delta > 0)

    print(json.dumps({
        "status": "pass",
        "base_feature_count": len(base["features"]),
        "treatment_feature_count": len(treatment["features"]),
        "hard_mask_changed": False,
        "soft_cost_added_cell_count": int(np.count_nonzero(delta > 0)),
        "soft_cost_added_sum": float(delta.sum()),
        "sample_grid_shape": list(shape),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
