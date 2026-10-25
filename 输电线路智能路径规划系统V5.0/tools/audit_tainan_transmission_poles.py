"""Audit Tainan's public 110-year transmission-pole SHP without planning use."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(shapefile, source_archive=None):
    layer = gpd.read_file(shapefile, encoding="UTF-8")
    bounds = layer.total_bounds.tolist() if not layer.empty else []
    notes = layer.get("Notes")
    tower_note = (notes.fillna("").astype(str).str.contains("電塔", regex=False)
                  if notes is not None else None)
    fields = {str(name).casefold() for name in layer.columns}
    return {
        "feature_count": len(layer),
        "geometry_type_counts": dict(Counter(
            layer.geometry.geom_type.fillna("<null>"))),
        "crs": layer.crs.to_string() if layer.crs else None,
        "coordinate_bounds_raw": bounds,
        "field_names": [str(name) for name in layer.columns],
        "category_code_counts": ({str(key): int(value) for key, value in
                                  layer["CatCode"].value_counts(dropna=False).items()}
                                 if "CatCode" in layer else {}),
        "notes_counts": ({str(key): int(value) for key, value in
                          layer["Notes"].fillna("<null>").value_counts().items()}
                         if "Notes" in layer else {}),
        "features_with_tower_note_count": int(tower_note.sum()) if tower_note is not None else 0,
        "unique_pole_numbers": (int(layer["PoleNo"].dropna().nunique())
                                if "PoleNo" in layer else None),
        "material_counts": ({str(key): int(value) for key, value in
                             layer["Material"].fillna("<null>").value_counts().items()}
                            if "Material" in layer else {}),
        "use_status_counts": ({str(key): int(value) for key, value in
                               layer["UseStat"].fillna("<null>").value_counts().items()}
                              if "UseStat" in layer else {}),
        "data_status_counts": ({str(key): int(value) for key, value in
                                layer["Status"].fillna("<null>").value_counts().items()}
                               if "Status" in layer else {}),
        "time_position_counts": ({str(key): int(value) for key, value in
                                  layer["TimePos"].fillna("<null>").value_counts().items()}
                                 if "TimePos" in layer else {}),
        "voltage_circuit_or_line_id_fields_present": any(
            any(token in name for token in ("voltage", "circuit", "line_id", "线路"))
            for name in fields),
        "source_archive_sha256": sha256(source_archive) if source_archive else None,
        "planning_use": "reference_only_not_a_verified_tower_candidate_layer",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shapefile", type=Path)
    parser.add_argument("--source-archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.shapefile, args.source_archive),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
