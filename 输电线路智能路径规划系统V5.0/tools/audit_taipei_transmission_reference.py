"""Audit Taipei's public transmission-pipeline XML as a post-planning reference."""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


NS = {
    "gml": "http://www.opengis.net/gml",
    "utl": "https://standards.moi.gov.tw/schema/utilityex",
}


def read_line_features(path):
    root = ET.parse(path).getroot()
    rows = []
    missing = Counter()
    for member in root.findall(".//gml:featureMember", NS):
        feature = member.find("utl:UTL_管道", NS)
        if feature is None:
            continue
        geometry = feature.find(".//gml:LineString", NS)
        if geometry is None or geometry.get("srsName") != "EPSG:3826":
            missing["missing_or_unexpected_crs"] += 1
            continue
        pos_list = geometry.find("gml:posList", NS)
        values = [float(value) for value in (pos_list.text or "").split()] if pos_list is not None else []
        dimension = int(geometry.get("srsDimension", "3"))
        if len(values) < dimension * 2 or len(values) % dimension:
            missing["invalid_pos_list"] += 1
            continue
        points = [tuple(values[i:i + 2]) for i in range(0, len(values), dimension)]
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        start_id = feature.findtext("utl:起點編號", namespaces=NS) or ""
        end_id = feature.findtext("utl:終點編號", namespaces=NS) or ""
        feature_id = feature.findtext("utl:識別碼", namespaces=NS) or ""
        if not start_id:
            missing["missing_start_id"] += 1
        if not end_id:
            missing["missing_end_id"] += 1
        if not feature_id:
            missing["missing_feature_id"] += 1
        try:
            attribute_length = float(feature.findtext("utl:管線長度", namespaces=NS))
        except (TypeError, ValueError):
            attribute_length = None
            missing["invalid_attribute_length"] += 1
        rows.append({
            "feature_id": feature_id,
            "start_id": start_id,
            "end_id": end_id,
            "points": points,
            "geometry_length_m": length,
            "attribute_length_m": attribute_length,
        })
    return rows, missing


def read_station_features(path):
    root = ET.parse(path).getroot()
    rows = []
    for member in root.findall(".//gml:featureMember", NS):
        feature = member.find("utl:UTL_場站", NS)
        if feature is None:
            continue
        point = feature.find(".//gml:Point", NS)
        if point is None or point.get("srsName") != "EPSG:3826":
            continue
        coordinates = point.find("gml:coordinates", NS)
        values = [float(value) for value in (coordinates.text or "").split()] if coordinates is not None else []
        if len(values) < 2:
            continue
        rows.append({
            "feature_id": feature.findtext("utl:識別碼", namespaces=NS) or "",
            "name": feature.findtext("utl:場站名稱", namespaces=NS) or "",
            "point": tuple(values[:2]),
        })
    return rows


def distance_to_segment(point, start, end):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    fraction = 0 if denominator == 0 else max(
        0, min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator)
    )
    return math.dist(point, (start[0] + fraction * dx, start[1] + fraction * dy))


def audit(lines_path, stations_path):
    lines, missing = read_line_features(lines_path)
    stations = read_station_features(stations_path)
    segments = [
        (start, end)
        for line in lines
        for start, end in zip(line["points"], line["points"][1:])
    ]
    endpoint_ids = {
        node_id for line in lines for node_id in (line["start_id"], line["end_id"]) if node_id
    }
    station_ids = {station["feature_id"] for station in stations}
    station_checks = []
    for station in stations:
        point = station["point"]
        nearest = min((distance_to_segment(point, start, end) for start, end in segments), default=None)
        station_checks.append({
            "name": station["name"],
            "id_is_line_endpoint": station["feature_id"] in endpoint_ids,
            "nearest_line_geometry_m": None if nearest is None else round(nearest, 2),
        })
    xs = [point[0] for line in lines for point in line["points"]]
    ys = [point[1] for line in lines for point in line["points"]]
    attribute_lengths = [line["attribute_length_m"] for line in lines if line["attribute_length_m"] is not None]
    return {
        "line_feature_count": len(lines),
        "unique_line_feature_ids": len({line["feature_id"] for line in lines if line["feature_id"]}),
        "line_features_with_both_endpoint_ids": sum(bool(line["start_id"] and line["end_id"]) for line in lines),
        "unique_endpoint_ids": len(endpoint_ids),
        "geometry_length_sum_km": round(sum(line["geometry_length_m"] for line in lines) / 1000, 3),
        "attribute_length_sum_km": round(sum(attribute_lengths) / 1000, 3),
        "station_feature_count": len(stations),
        "station_ids_matching_line_endpoint_ids": len(station_ids & endpoint_ids),
        "coordinate_extent_epsg3826": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
        "missing_or_invalid": dict(missing),
        "station_checks": station_checks,
        "planning_use": "forbidden_reference_only_after_route_generation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lines_xml", type=Path)
    parser.add_argument("stations_xml", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.lines_xml, args.stations_xml), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
