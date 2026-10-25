"""Audit Tainan's public transmission-pipeline KML as a reference-only layer."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from pyproj import Geod


KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def audit(path, manholes_path=None):
    root = ET.parse(path).getroot()
    geod = Geod(ellps="WGS84")
    features = []
    field_counts = defaultdict(Counter)
    graph = defaultdict(list)
    missing = Counter()
    all_coords = []

    for placemark in root.findall(".//kml:Placemark", KML_NS):
        attrs = {
            item.get("name", ""): item.text or ""
            for item in placemark.findall(".//kml:SimpleData", KML_NS)
        }
        coordinates = placemark.find(".//kml:LineString/kml:coordinates", KML_NS)
        if coordinates is None or not coordinates.text:
            missing["missing_line_geometry"] += 1
            continue
        points = []
        for item in coordinates.text.split():
            values = item.split(",")
            if len(values) < 2:
                continue
            points.append((float(values[0]), float(values[1])))
        if len(points) < 2:
            missing["invalid_line_geometry"] += 1
            continue

        length_m = sum(
            abs(geod.inv(start[0], start[1], end[0], end[1])[2])
            for start, end in zip(points, points[1:])
        )
        for field in ("管理單位", "作業區分", "管線型態", "使用狀態", "資料狀態", "尺寸單位", "類別碼"):
            field_counts[field][attrs.get(field, "<blank>")] += 1
        for field in ("識別碼", "起點編號", "終點編號"):
            if not attrs.get(field):
                missing[f"missing_{field}"] += 1

        start_id = attrs.get("起點編號", "")
        end_id = attrs.get("終點編號", "")
        if start_id and end_id:
            graph[start_id].append(end_id)
            graph[end_id].append(start_id)
        try:
            attr_length_m = float(attrs["管線長度"])
        except (KeyError, ValueError):
            attr_length_m = None
            missing["invalid_attribute_length"] += 1
        features.append({
            "id": attrs.get("識別碼", ""),
            "start_id": start_id,
            "end_id": end_id,
            "length_m": length_m,
            "attribute_length_m": attr_length_m,
            "date": attrs.get("timePosition", ""),
            "points": points,
        })
        all_coords.extend(points)

    components = []
    visited = set()
    for node in graph:
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    endpoint_pairs = Counter(
        tuple(sorted((feature["start_id"], feature["end_id"])))
        for feature in features
        if feature["start_id"] and feature["end_id"]
    )
    lengths = [feature["length_m"] for feature in features]
    vertex_counts = [len(feature["points"]) for feature in features]
    attribute_lengths = [
        feature["attribute_length_m"]
        for feature in features
        if feature["attribute_length_m"] is not None
    ]
    dates = [feature["date"] for feature in features if feature["date"]]
    ids = [feature["id"] for feature in features if feature["id"]]
    result = {
        "placemark_count": len(root.findall(".//kml:Placemark", KML_NS)),
        "valid_line_feature_count": len(features),
        "unique_feature_ids": len(set(ids)),
        "geometry_length_sum_km": round(sum(lengths) / 1000, 3),
        "attribute_length_sum_km": round(sum(attribute_lengths) / 1000, 3),
        "max_geometry_length_m": round(max(lengths), 2) if lengths else None,
        "median_geometry_length_m": round(sorted(lengths)[len(lengths) // 2], 2) if lengths else None,
        "total_vertices": sum(vertex_counts),
        "vertex_count_min_median_max": [
            min(vertex_counts), sorted(vertex_counts)[len(vertex_counts) // 2], max(vertex_counts)
        ] if vertex_counts else [],
        "endpoint_node_count": len(graph),
        "endpoint_graph_component_count": len(components),
        "largest_component_node_counts": sorted((len(c) for c in components), reverse=True)[:10],
        "endpoint_pairs_reused": sum(count > 1 for count in endpoint_pairs.values()),
        "duplicate_endpoint_pair_edges": sum(count - 1 for count in endpoint_pairs.values() if count > 1),
        "time_position_min_max": [min(dates), max(dates)] if dates else [],
        "coordinate_bounds_lonlat": [
            min(point[0] for point in all_coords), min(point[1] for point in all_coords),
            max(point[0] for point in all_coords), max(point[1] for point in all_coords),
        ] if all_coords else [],
        "attribute_counts": {field: dict(values) for field, values in field_counts.items()},
        "missing_or_invalid": dict(missing),
        "station_features_present": False,
        "station_pairing_ready": False,
        "planning_use": "forbidden_reference_only_after_route_generation",
    }
    if manholes_path is not None:
        result["manhole_geometry_join"] = audit_manhole_geometry(features, manholes_path)
    return result


def audit_manhole_geometry(line_features, manholes_path):
    root = ET.parse(manholes_path).getroot()
    points = []
    for placemark in root.findall(".//kml:Placemark", KML_NS):
        attrs = {
            item.get("name", ""): item.text or ""
            for item in placemark.findall(".//kml:SimpleData", KML_NS)
        }
        coordinates = placemark.find(".//kml:Point/kml:coordinates", KML_NS)
        if coordinates is None or not coordinates.text:
            continue
        values = coordinates.text.split()[0].split(",")
        if len(values) < 2:
            continue
        points.append({
            "xy": (float(values[0]), float(values[1])),
            "id": attrs.get("識別碼", ""),
            "number": attrs.get("人手孔編號", ""),
            "date": attrs.get("timePosition", ""),
            "use_status": attrs.get("使用狀態", "<blank>"),
        })

    point_coordinates = {point["xy"] for point in points}
    line_endpoint_numbers = {
        endpoint
        for feature in line_features
        for endpoint in (feature["start_id"], feature["end_id"])
        if endpoint
    }
    manhole_numbers = {point["number"] for point in points if point["number"]}
    start_hits = sum(feature["points"][0] in point_coordinates for feature in line_features)
    end_hits = sum(feature["points"][-1] in point_coordinates for feature in line_features)
    both_hits = sum(
        feature["points"][0] in point_coordinates
        and feature["points"][-1] in point_coordinates
        for feature in line_features
    )
    unique_endpoints = {
        point
        for feature in line_features
        for point in (feature["points"][0], feature["points"][-1])
    }
    unique_vertices = {
        point for feature in line_features for point in feature["points"]
    }
    dates = [point["date"] for point in points if point["date"]]
    return {
        "point_feature_count": len(points),
        "unique_feature_ids": len({point["id"] for point in points if point["id"]}),
        "unique_manhole_numbers": len(manhole_numbers),
        "shared_line_endpoint_number_labels": len(line_endpoint_numbers & manhole_numbers),
        "unique_point_coordinates": len(point_coordinates),
        "line_feature_start_end_exact_coordinate_matches": {
            "start_count": start_hits,
            "end_count": end_hits,
            "both_ends_count": both_hits,
            "line_feature_count": len(line_features),
            "unique_endpoint_coordinates": len(unique_endpoints),
            "unique_endpoint_coordinates_matched": len(unique_endpoints & point_coordinates),
            "unique_line_vertices": len(unique_vertices),
            "unique_line_vertices_matched": len(unique_vertices & point_coordinates),
        },
        "time_position_min_max": [min(dates), max(dates)] if dates else [],
        "use_status_counts": dict(Counter(point["use_status"] for point in points)),
        "join_interpretation": "exact_geometry_only_not_station_or_network_topology",
        "planning_use": "post_route_geometry_qa_only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kml", type=Path)
    parser.add_argument("--manholes-kml", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.kml, args.manholes_kml), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
