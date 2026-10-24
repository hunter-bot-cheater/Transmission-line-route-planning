"""Download a privacy-minimized, endpoint-corridor subset of official ARDSWC tiles."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pyogrio
import requests
from pyproj import Transformer
from shapely.geometry import LineString, box
from shapely.ops import transform as transform_geometry


LAYERS = {
    "debris_flow_impact": ("DEBRIS1753_20260126", "debris_affect", "176526"),
    "landslide_potential": ("POTENTIAL_LANDSLIDE_20260106", "potential_landslide", "176527"),
    "landslide_impact": ("POTENTIAL_LANDSLIDE_AFFECT_20260106", "potential_landslide_affect", "176528"),
}
TILE_URL = "https://gis.ardswc.gov.tw/api/ardswc/vectortiles/shp/{}/{}/{}/{}.pbf"
HALF_WORLD = 20037508.342789244


def read_corridors(csv_path: Path, buffer_m: float):
    to_twd97 = Transformer.from_crs(4326, 3826, always_xy=True)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")))
    if len(rows) != 12:
        raise ValueError(f"expected 12 endpoint cases, got {len(rows)}")
    corridors = {}
    endpoints = []
    for row in rows:
        case_id = row["name"].strip()
        start = (float(row["start_lon"]), float(row["start_lat"]))
        end = (float(row["end_lon"]), float(row["end_lat"]))
        if not case_id or not all(math.isfinite(v) for v in (*start, *end)):
            raise ValueError("case IDs and endpoint coordinates must be valid")
        line = LineString([to_twd97.transform(*start), to_twd97.transform(*end)])
        corridors[case_id] = line.buffer(buffer_m, cap_style="round", join_style="round")
        endpoints.append({"case_id": case_id, "start_lon_lat": start, "end_lon_lat": end})
    return corridors, endpoints


def tiles_for_corridors(corridors, zoom):
    to_mercator = Transformer.from_crs(3826, 3857, always_xy=True)
    n = 1 << zoom
    size = 2 * HALF_WORLD / n
    tiles = set()
    for corridor in corridors.values():
        corridor = transform_geometry(to_mercator.transform, corridor)
        min_x, min_y, max_x, max_y = corridor.bounds
        x0 = max(0, math.floor((min_x + HALF_WORLD) / size))
        x1 = min(n - 1, math.floor((max_x + HALF_WORLD) / size))
        y0 = max(0, math.floor((HALF_WORLD - max_y) / size))
        y1 = min(n - 1, math.floor((HALF_WORLD - min_y) / size))
        for y in range(y0, y1 + 1):
            top = HALF_WORLD - y * size
            bottom = top - size
            for x in range(x0, x1 + 1):
                left = -HALF_WORLD + x * size
                tile = box(left, bottom, left + size, top)
                if corridor.intersects(tile):
                    tiles.add((x, y))
    return sorted(tiles, key=lambda p: (p[1], p[0]))


def fetch_one(task):
    layer_key, (layer_id, slug, dataset_id), zoom, x, y = task
    url = TILE_URL.format(slug, zoom, y, x)
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "route-planner-open-data-audit/1.0"})
        payload = response.content
        record = {
            "layer": layer_key, "layer_id": layer_id, "dataset_id": dataset_id,
            "z": zoom, "x": x, "y": y, "url": url, "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type"), "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(), "status": "download_failed",
            "feature_fragments": 0, "invalid_geometry_fragments": 0,
        }
        if response.status_code == 200 and not payload:
            record["status"] = "official_empty"
        elif response.status_code == 200:
            record["status"] = "downloaded"
        return record, payload
    except requests.RequestException as exc:
        return ({"layer": layer_key, "layer_id": layer_id, "dataset_id": dataset_id,
                 "z": zoom, "x": x, "y": y, "url": url, "http_status": None,
                 "content_type": None, "bytes": 0, "sha256": None,
                 "status": "request_error", "error": f"{type(exc).__name__}: {exc}",
                 "feature_fragments": 0, "invalid_geometry_fragments": 0}, b"")


def process_payload(record, payload, corridors, temporary_dir):
    if record["status"] != "downloaded":
        return []
    temp_path = temporary_dir / f"{record['layer']}_{record['z']}_{record['y']}_{record['x']}.pbf"
    temp_path.write_bytes(payload)
    try:
        frame = pyogrio.read_dataframe(
            temp_path, layer=0, X=record["x"], Y=record["y"], Z=record["z"]
        )
    except Exception as exc:
        record["status"] = "decode_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        temp_path.unlink(missing_ok=True)
        return []
    temp_path.unlink(missing_ok=True)
    if frame.crs is None or str(frame.crs).upper() != "EPSG:3857":
        record["status"] = "unexpected_crs"
        record["error"] = f"expected EPSG:3857, got {frame.crs}"
        return []
    record["feature_fragments"] = len(frame)
    record["invalid_geometry_fragments"] = int((~frame.geometry.is_valid).sum())
    if frame.empty:
        record["status"] = "decoded_empty"
        return []
    frame = frame.to_crs(3826)
    to_wgs84 = Transformer.from_crs(3826, 4326, always_xy=True)
    output = []
    for _, feature in frame.iterrows():
        geom = feature.geometry
        if geom is None or geom.is_empty:
            continue
        feature_id = next((feature.get(k) for k in
                           ("debrisno", "lslno", "islno", "id", "gid")
                           if feature.get(k) is not None), None)
        risk = feature.get("risk")
        for case_id, corridor in corridors.items():
            if not geom.intersects(corridor):
                continue
            clipped = geom.intersection(corridor)
            if clipped.is_empty:
                continue
            clipped_wgs84 = transform_geometry(to_wgs84.transform, clipped)
            if clipped_wgs84.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            output.append({
                "case_id": case_id,
                "source_feature_id": str(feature_id) if feature_id is not None else None,
                "risk": str(risk) if risk is not None else None,
                "source_tile": f"{record['z']}/{record['y']}/{record['x']}",
                "geometry": clipped_wgs84,
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoints-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zoom", type=int, default=13)
    parser.add_argument("--buffer-m", type=float, default=10000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_hash = hashlib.sha256(args.endpoints_csv.read_bytes()).hexdigest()
    corridors, endpoints = read_corridors(args.endpoints_csv, args.buffer_m)
    tiles = tiles_for_corridors(corridors, args.zoom)
    tasks = [(key, spec, args.zoom, x, y) for key, spec in LAYERS.items() for x, y in tiles]
    print(f"cases={len(endpoints)} corridor_buffer_m={args.buffer_m} zoom={args.zoom} spatial_tiles={len(tiles)} requests={len(tasks)}", flush=True)

    records = []
    payloads = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_one, task) for task in tasks]
        for count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record, payload = future.result()
            records.append(record)
            if payload and record["status"] == "downloaded":
                payloads[(record["layer"], record["x"], record["y"])] = payload
            if count % 150 == 0 or count == len(futures):
                print(f"fetched={count}/{len(futures)}", flush=True)

    feature_rows = {key: [] for key in LAYERS}
    with tempfile.TemporaryDirectory(prefix="ardswc_mvt_") as temp:
        temp_dir = Path(temp)
        for count, record in enumerate(records, start=1):
            key = record["layer"]
            payload = payloads.pop((key, record["x"], record["y"]), b"")
            feature_rows[key].extend(process_payload(record, payload, corridors, temp_dir))
            if count % 150 == 0 or count == len(records):
                print(f"decoded={count}/{len(records)}", flush=True)

    for key, rows in feature_rows.items():
        if not rows:
            continue
        frame = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        frame.to_file(args.output_dir / f"{key}_z{args.zoom}.gpkg", layer=key,
                      driver="GPKG", index=False)

    tile_path = args.output_dir / "tile_responses.jsonl"
    with tile_path.open("w", encoding="utf-8", newline="\n") as stream:
        for item in sorted(records, key=lambda r: (r["layer"], r["y"], r["x"])):
            stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_agency": "Agriculture and Rural Development and Soil and Water Conservation Agency (ARDSWC), Taiwan",
        "source_catalog": "https://data.ardswc.gov.tw/Data/OpenData/Api",
        "official_mvt_endpoint_template": TILE_URL,
        "official_shp_dataset_pages": {k: f"https://data.gov.tw/dataset/{v[2]}" for k, v in LAYERS.items()},
        "official_shp_resource_urls": {
            "176526": "https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?id=J73&FileType=SHP&RID=71085",
            "176527": "https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?id=J74&FileType=SHP&RID=66656",
            "176528": "https://data.moa.gov.tw/OpenData/GetOpenDataFile.aspx?id=J75&FileType=SHP&RID=66657",
        },
        "original_shp_download_limitations": {
            "result": "not_downloaded",
            "data_gov_resource_fetch": "timeout for all three dataset resource links",
            "local_head_probe": "J73/RID=71085 timed out after 20 seconds",
            "curl_probe": "Could not resolve host: data.moa.gov.tw",
            "original_crs_and_size": "unverified because the archive was not received",
        },
        "license": "Government Open Data License, Taiwan, version 1; free of charge, per data.gov.tw dataset metadata",
        "format": "Official Mapbox Vector Tiles (MVT/protobuf PBF), not original SHP; tile geometries are zoom-dependent, generalized, and tile-clipped.",
        "zoom": args.zoom,
        "cartographic_scale_denominator": None,
        "scale_note": "ARDSWC MVT endpoint does not publish a scale denominator or generalization tolerance; zoom is recorded without interpreting it as survey accuracy.",
        "original_shp_crs": None,
        "original_shp_crs_note": "The official resource names indicate TWD97, but the original archive and .prj were not downloadable for verification.",
        "tile_crs_verified": "EPSG:3857 via GDAL MVT driver",
        "tile_http_content_type_observed": "text/html; charset=utf-8",
        "tile_content_type_note": "The endpoint labels responses as text/html; nonempty payloads were independently decoded by GDAL's MVT driver before deriving features.",
        "corridor_crs": "EPSG:3826 (TWD97 / TM2 zone 121; metric buffering)",
        "derived_output_crs": "EPSG:4326",
        "scope_method": "12 endpoint-to-endpoint straight segments in EPSG:3826 (TWD97 / TM2 zone 121), each buffered by the stated ground-distance in meters; buffered corridors transformed to EPSG:3857 only for XYZ tile selection, then geometries clipped in EPSG:3826.",
        "corridor_buffer_m": args.buffer_m,
        "endpoint_csv_path": str(args.endpoints_csv.resolve()),
        "endpoint_csv_sha256": input_hash,
        "cases": endpoints,
        "spatial_tile_count": len(tiles),
        "request_count": len(tasks),
        "address_fields": "Dropped from derived GeoPackages; only case_id, source_feature_id, risk, source_tile, and geometry are retained.",
        "geometric_validation": "Per-tile fragments decoded with GDAL MVT; invalid counts are recorded in tile_responses.jsonl. Derived GeoPackages are read back for validation in the run report.",
        "layers": {},
    }
    for key, (layer_id, slug, dataset_id) in LAYERS.items():
        subset = [r for r in records if r["layer"] == key]
        manifest["layers"][key] = {
            "dataset_id": dataset_id, "mvt_layer_id": layer_id, "mvt_slug": slug,
            "request_status_counts": {status: sum(r["status"] == status for r in subset)
                                      for status in sorted({r["status"] for r in subset})},
            "downloaded_bytes": sum(r["bytes"] for r in subset),
            "decoded_feature_fragments": sum(r["feature_fragments"] for r in subset),
            "invalid_geometry_fragments": sum(r["invalid_geometry_fragments"] for r in subset),
            "corridor_clipped_output_features": len(feature_rows[key]),
        }
        gpkg = args.output_dir / f"{key}_z{args.zoom}.gpkg"
        if gpkg.is_file():
            check = pyogrio.read_dataframe(gpkg, layer=key)
            manifest["layers"][key]["gpkg_validation"] = {
                "feature_count": len(check), "crs": str(check.crs),
                "invalid_geometries": int((~check.geometry.is_valid).sum()),
                "bounds": list(map(float, check.total_bounds)),
                "sha256": hashlib.sha256(gpkg.read_bytes()).hexdigest(),
                "bytes": gpkg.stat().st_size,
            }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    assets = sorted(p for p in args.output_dir.iterdir()
                    if p.is_file() and p.name != "SHA256SUMS.txt")
    (args.output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
                for p in assets), encoding="utf-8"
    )
    print(json.dumps(manifest["layers"], ensure_ascii=False, indent=2), flush=True)
    print(f"manifest={args.output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
