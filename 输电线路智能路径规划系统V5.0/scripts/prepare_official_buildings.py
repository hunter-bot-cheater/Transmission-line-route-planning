"""Convert an authorized NLSC Taiwan e-Map BUILD vector layer to planner constraints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


SOURCE = "NLSC Taiwan e-Map BUILD"
DATASET_URL = "https://www.nlsc.gov.tw/cp.aspx?n=10702"


def convert_buildings(input_path, output_path, *, source_date="unknown", license_note="as supplied by NLSC"):
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("转换图层需要安装 requirements.txt 中的 geopandas") from exc

    frame = gpd.read_file(input_path)
    if frame.crs is None:
        raise ValueError("建筑图层缺少坐标系，不能安全重投影")
    source_crs = frame.crs.to_string()
    frame = frame.to_crs("EPSG:4326")
    frame.geometry = frame.geometry.make_valid()
    frame = frame[frame.geometry.geom_type.isin(("Polygon", "MultiPolygon"))]
    frame = frame[~frame.geometry.is_empty]
    if frame.empty:
        raise ValueError("输入图层中没有有效建筑面")

    min_lon, min_lat, max_lon, max_lat = frame.total_bounds
    if min_lon < -180 or max_lon > 180 or min_lat < -90 or max_lat > 90:
        raise ValueError("重投影后的建筑面超出 WGS84 坐标范围")

    source_label = f"{SOURCE}; source date {source_date}"
    features = []
    for geometry in frame.geometry:
        features.append({
            "type": "Feature",
            "geometry": geometry.__geo_interface__,
            "properties": {
                "mode": "hard",
                "source": source_label,
                "category": "building_footprint",
                "dataset_id": "NLSC_TAIWAN_EMAP_BUILD",
                "source_updated_at": source_date if source_date != "unknown" else None,
            },
        })

    document = {
        "type": "FeatureCollection",
        "features": features,
        "data_sources": [{
            "agency": "National Land Surveying and Mapping Center",
            "dataset_id": "Taiwan e-Map BUILD",
            "dataset_url": DATASET_URL,
            "source_updated_at": source_date if source_date != "unknown" else None,
            "source_crs": source_crs,
            "license": license_note,
        }],
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": "OGC:CRS84 (WGS84 longitude, latitude)",
            "constraint_mode": "hard",
            "feature_count": len(features),
            "important_note": (
                "Building footprints only; rasterized cells are blocked. Verify source coverage, "
                "currency, access terms, and project-specific clearance rules."
            ),
        },
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".part")
    partial.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    partial.replace(output_path)
    return len(features), source_crs


def main():
    parser = argparse.ArgumentParser(description="准备 NLSC 台湾通用電子地圖建物面约束")
    parser.add_argument("--input", required=True, help="已获授权的 BUILD 矢量文件，如 SHP/GPKG/GML")
    parser.add_argument("--output", required=True, help="输出 WGS84 约束 GeoJSON")
    parser.add_argument("--source-date", default="unknown", help="输入图层的测制或更新日期")
    parser.add_argument("--license-note", default="as supplied by NLSC",
                        help="保留随授权/申请取得的许可说明")
    args = parser.parse_args()
    count, source_crs = convert_buildings(
        args.input, args.output, source_date=args.source_date, license_note=args.license_note
    )
    print(f"已导出 {count} 个建筑面；源 CRS={source_crs}；输出={args.output}")


if __name__ == "__main__":
    main()
