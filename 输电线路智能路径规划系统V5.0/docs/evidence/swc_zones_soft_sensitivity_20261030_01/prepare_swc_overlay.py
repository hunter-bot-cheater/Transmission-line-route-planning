#!/usr/bin/env python3
"""Create a traceable, soft-only overlay from Taiwan's 115-year SWC zones."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pyogrio
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping
from shapely.ops import unary_union


DATASET_ID = "176847"
DATASET_URL = f"https://data.gov.tw/dataset/{DATASET_ID}"
SOURCE_URL = "https://data.moa.gov.tw/GetOpenDataFile.aspx?FileType=ZIP&RID=87575&id=J80"
LICENSE = "Open Government Data License, version 1.0"
WEIGHT = 1.0
EXPECTED_FEATURE_COUNT = 66
READ_COLUMNS = ["DRSWC_Land", "Desig_Code", "County", "Anno_Area", "Anno_Date"]
MAX_REPAIR_AREA_CHANGE_RATIO = 0.01


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--base-constraints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.source_zip.is_file():
        raise FileNotFoundError(args.source_zip)
    if not args.base_constraints.is_file():
        raise FileNotFoundError(args.base_constraints)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to reuse output directory: {args.output_dir}")

    archive = args.source_zip.resolve()
    vector_path = f"/vsizip/{archive.as_posix()}"
    info = pyogrio.read_info(vector_path)
    fields = {str(name).strip().casefold() for name in info.get("fields", ())}
    if fields.intersection({"address", "地址", "住址"}):
        raise ValueError("The open-data layer contains an address-like attribute")
    if int(info.get("features", -1)) != EXPECTED_FEATURE_COUNT:
        raise ValueError("The SHP feature count differs from the published 66-zone snapshot")
    if str(info.get("geometry_type", "")).upper() not in {"POLYGON", "MULTIPOLYGON"}:
        raise ValueError("Expected polygon geometries")

    frame = gpd.read_file(vector_path, columns=READ_COLUMNS, engine="pyogrio")
    if frame.crs is None or frame.crs.to_epsg() != 3826:
        raise ValueError("Expected TWD97 / TM2 zone 121 (EPSG:3826)")
    if len(frame) != EXPECTED_FEATURE_COUNT:
        raise ValueError("Feature count differs from the published 66-zone snapshot")
    if set(frame["DRSWC_Land"].astype(str)) != {"劃定"}:
        raise ValueError("Unexpected designation status; review before planning use")

    repairs = []
    repaired_geometries = []
    for index, geometry in enumerate(frame.geometry):
        if geometry.is_valid:
            repaired_geometries.append(geometry)
            continue
        fixed = make_valid(geometry)
        polygon_parts = []
        stack = [fixed]
        while stack:
            part = stack.pop()
            if isinstance(part, Polygon):
                polygon_parts.append(part)
            elif isinstance(part, (MultiPolygon, GeometryCollection)):
                stack.extend(part.geoms)
        if not polygon_parts:
            raise ValueError(f"Invalid feature {index} did not repair to polygon geometry")
        fixed = unary_union(polygon_parts)
        if fixed.is_empty or not fixed.is_valid or fixed.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"Invalid feature {index} could not be safely repaired")
        area_change_ratio = abs(fixed.area - geometry.area) / max(geometry.area, 1e-12)
        if area_change_ratio > MAX_REPAIR_AREA_CHANGE_RATIO:
            raise ValueError(
                f"Invalid feature {index} repair changes polygon area by "
                f"{area_change_ratio:.6%}, above the 1% review limit")
        repairs.append({
            "source_row_zero_based": index,
            "repair_method": "shapely.make_valid; retain polygonal components only",
            "source_geometry_type": geometry.geom_type,
            "repaired_geometry_type": fixed.geom_type,
            "source_area_m2": float(geometry.area),
            "repaired_area_m2": float(fixed.area),
            "area_change_ratio": float(area_change_ratio),
        })
        repaired_geometries.append(fixed)
    frame = frame.set_geometry(gpd.GeoSeries(repaired_geometries, crs=frame.crs))

    wgs84_geometries = frame.to_crs("EPSG:4326").geometry
    dissolved = unary_union(list(wgs84_geometries))
    if dissolved.is_empty or not dissolved.is_valid or dissolved.geom_type not in {
            "Polygon", "MultiPolygon"}:
        raise ValueError("Dissolved SWC geometry is not a valid polygon")

    source_sha = sha256(archive)
    base_path = args.base_constraints.resolve()
    base_bytes = base_path.read_bytes()
    document = json.loads(base_bytes.decode("utf-8-sig"))
    if (not isinstance(document, dict) or document.get("type") != "FeatureCollection"
            or not isinstance(document.get("features"), list)):
        raise ValueError("Base constraints must be a GeoJSON FeatureCollection")
    if document.get("metadata") is not None and not isinstance(document["metadata"], dict):
        raise ValueError("Base constraints metadata must be an object")
    if document.get("data_sources") is not None and not isinstance(document["data_sources"], list):
        raise ValueError("Base constraints data_sources must be a list")
    if "crs" in document:
        crs = document["crs"]
        crs_properties = crs.get("properties") if isinstance(crs, dict) else None
        crs_name = crs_properties.get("name", "") if isinstance(crs_properties, dict) else ""
        if "4326" not in str(crs_name).upper() and "CRS84" not in str(crs_name).upper():
            raise ValueError("Base constraints must use WGS84 coordinates")

    source_record = {
        "agency": "Taiwan Ministry of Agriculture",
        "dataset_id": DATASET_ID,
        "dataset_name": "115年度66區特定水土保持區範圍圖",
        "dataset_url": DATASET_URL,
        "source_download_url": SOURCE_URL,
        "source_archive_sha256": source_sha,
        "license": LICENSE,
        "attribution": "資料來源：農業部，115年度66區特定水土保持區範圍圖；政府資料開放授權條款第1版。",
        "planning_role": "soft-only environmental and regulatory review screening",
    }
    overlay_metadata = {
            "dataset_id": DATASET_ID,
            "dataset_name": "115年度66區特定水土保持區範圍圖",
            "source_crs": "EPSG:3826",
            "output_crs": "EPSG:4326",
            "feature_count": EXPECTED_FEATURE_COUNT,
            "designation_status_values": ["劃定"],
            "invalid_source_feature_count": len(repairs),
            "geometry_repairs": repairs,
            "soft_only": True,
            "weight": WEIGHT,
            "hard_exclusion_allowed": False,
            "interpretation": "Regulatory and environmental review signal only; not a construction ban or permit determination.",
            "route_geometry_used": False,
    }
    document.setdefault("metadata", {})["official_swc_zone_overlay"] = overlay_metadata
    document.setdefault("data_sources", []).append(source_record)
    feature = {
        "type": "Feature",
        "properties": {
            "mode": "soft",
            "weight": WEIGHT,
            "source": "Taiwan MOA 115-year specific soil and water conservation zones",
            "category": "designated_conservation_review",
        },
        "geometry": mapping(dissolved),
    }
    document["features"].append(feature)

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "constraints_with_swc_zones.geojson"
    output.write_text(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n",
                      encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_url": DATASET_URL,
        "source_download_url": SOURCE_URL,
        "source_archive_path": str(archive),
        "source_archive_bytes": archive.stat().st_size,
        "source_archive_sha256": source_sha,
        "base_constraints_path": str(base_path),
        "base_constraints_sha256": hashlib.sha256(base_bytes).hexdigest(),
        "source_crs": "EPSG:3826",
        "output_crs": "EPSG:4326",
        "source_feature_count": len(frame),
        "invalid_source_feature_count": len(repairs),
        "geometry_repairs": repairs,
        "maximum_repair_area_change_ratio": max(
            (item["area_change_ratio"] for item in repairs), default=0.0),
        "maximum_allowed_repair_area_change_ratio": MAX_REPAIR_AREA_CHANGE_RATIO,
        "source_designation_status_counts": frame["DRSWC_Land"].astype(str).value_counts().to_dict(),
        "designation_code_counts": frame["Desig_Code"].astype(str).value_counts().to_dict(),
        "read_attribute_columns": READ_COLUMNS,
        "address_values_read_or_emitted": False,
        "route_geometry_used": False,
        "constraint_mode": "soft",
        "constraint_weight": WEIGHT,
        "hard_exclusion_allowed": False,
        "added_feature_count": 1,
        "feature_count_total": len(document["features"]),
        "output_path": str(output.resolve()),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
