"""站点台账与站间线路任务输入。"""

import csv
import hashlib
import math
import re


def route_output_name(route_id):
    """Return a filesystem-safe, collision-resistant directory name for a route ID."""
    route_id = str(route_id)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", route_id).strip("_-")[:64] or "route"
    suffix = hashlib.sha256(route_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{suffix}"


def station_coordinate_crs_properties(station_task):
    if not station_task:
        return {}
    return {
        "from_station_coordinate_crs": station_task["from_station"]["coordinate_crs"],
        "to_station_coordinate_crs": station_task["to_station"]["coordinate_crs"],
    }


def station_endpoint_properties(station_task):
    if not station_task:
        return {}
    source_independence = all(
        station_task[key]["reference_independence"] == "independent_of_reference"
        for key in ("from_station", "to_station")
    )
    result = {
        "station_endpoint_qualification": station_task["endpoint_qualification"],
        "station_endpoint_reference_independence": (
            "declared_independent_of_reference" if source_independence else "not_established"
        ),
    }
    for prefix, key in (("from", "from_station"), ("to", "to_station")):
        station = station_task[key]
        result.update({
            f"{prefix}_station_point_role": station["point_role"],
            f"{prefix}_station_verification_status": station["verification_status"],
            f"{prefix}_station_reference_independence": station["reference_independence"],
            f"{prefix}_station_source": station["source"] or None,
            f"{prefix}_station_verification_reference": station["verification_reference"] or None,
        })
    return result


def station_elevation_metrics(station_task):
    """Summarize surveyed station-ground elevations without mixing vertical datums."""
    start = station_task["from_station"]
    end = station_task["to_station"]
    start_elevation = start["ground_elevation_m"]
    end_elevation = end["ground_elevation_m"]
    start_datum = start["vertical_datum"]
    end_datum = end["vertical_datum"]
    if start_elevation is None and end_elevation is None:
        status, delta = "dem_only", None
    elif start_elevation is None or end_elevation is None:
        status, delta = "incomplete_station_measurements", None
    elif not start["ground_elevation_source"] or not end["ground_elevation_source"]:
        status, delta = "elevation_source_missing", None
    elif not start_datum or not end_datum:
        status, delta = "vertical_datum_missing", None
    elif start_datum.casefold() != end_datum.casefold():
        status, delta = "incompatible_vertical_datums", None
    else:
        status = "comparable"
        delta = round(end_elevation - start_elevation, 2)
    return {
        "from_station_ground_elevation_m": start_elevation,
        "from_station_ground_elevation_source": start["ground_elevation_source"] or None,
        "from_station_vertical_datum": start_datum or None,
        "to_station_ground_elevation_m": end_elevation,
        "to_station_ground_elevation_source": end["ground_elevation_source"] or None,
        "to_station_vertical_datum": end_datum or None,
        "station_ground_elevation_status": status,
        "station_ground_elevation_delta_m": delta,
    }


def station_clearance_elevation_overrides(station_task, dem_vertical_datum=None):
    """Return surveyed terminal elevations only when their declared datum matches the DEM."""
    if not station_task:
        return None, "dem_only"
    start = station_task["from_station"]
    end = station_task["to_station"]
    if start["ground_elevation_m"] is None and end["ground_elevation_m"] is None:
        return None, "dem_only"

    metrics = station_elevation_metrics(station_task)
    if metrics["station_ground_elevation_status"] != "comparable":
        return None, metrics["station_ground_elevation_status"]
    if not dem_vertical_datum or not str(dem_vertical_datum).strip():
        return None, "dem_vertical_datum_missing"
    station_datum = start["vertical_datum"].strip()
    if station_datum.casefold() != str(dem_vertical_datum).strip().casefold():
        return None, "incompatible_dem_vertical_datum"
    return {
        "start_ground_elevation_m": start["ground_elevation_m"],
        "goal_ground_elevation_m": end["ground_elevation_m"],
    }, "station_and_dem_vertical_datums_declared_matching"


def _dict_reader(file_obj, label, required_fields):
    reader = csv.DictReader(file_obj)
    fields = reader.fieldnames or []
    if any(not field for field in fields) or len(fields) != len(set(fields)):
        raise ValueError(f"{label}表头存在空列名或重复列名")
    missing = required_fields - set(fields)
    if missing:
        raise ValueError(f"{label}缺少列: {', '.join(sorted(missing))}")
    return reader


def load_station_tasks(stations_csv, route_tasks_csv, allow_unverified=False):
    """读取 stations.csv 与 route_tasks.csv，验证站点引用并返回规划任务。"""
    stations = {}
    with open(stations_csv, encoding="utf-8-sig", newline="") as f:
        reader = _dict_reader(f, "站点", {"station_id", "lat", "lon", "coordinate_crs"})
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"站点表第 {row_number} 行字段数多于表头")
            station_id = (row.get("station_id") or "").strip()
            if not station_id:
                raise ValueError("站点表缺少 station_id")
            if station_id in stations:
                raise ValueError(f"站点 ID 重复: {station_id}")
            coordinate_crs = (row.get("coordinate_crs") or "").strip().upper()
            if coordinate_crs != "EPSG:4326":
                raise ValueError(
                    f"站点 {station_id} 的 coordinate_crs 必须明确为 EPSG:4326；"
                    "其他坐标系请先转换后再填写 lat/lon")
            try:
                lat, lon = float(row["lat"]), float(row["lon"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"站点 {station_id} 的 lat/lon 无效") from exc
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"站点 {station_id} 的经纬度超出范围")
            elevation_text = (row.get("ground_elevation_m") or "").strip()
            try:
                ground_elevation = float(elevation_text) if elevation_text else None
            except ValueError as exc:
                raise ValueError(f"站点 {station_id} 的 ground_elevation_m 无效") from exc
            if ground_elevation is not None and not math.isfinite(ground_elevation):
                raise ValueError(f"站点 {station_id} 的 ground_elevation_m 必须是有限数值")
            point_role = (row.get("point_role") or "unknown").strip().casefold()
            verification_status = (row.get("verification_status") or "unknown").strip().casefold()
            if point_role not in {"line_connection_point", "station_centroid", "unknown"}:
                raise ValueError(f"站点 {station_id} 的 point_role 无效")
            if verification_status not in {"verified", "unverified", "unknown"}:
                raise ValueError(f"站点 {station_id} 的 verification_status 无效")
            reference_independence = (
                row.get("reference_independence") or "unknown"
            ).strip().casefold()
            if reference_independence not in {
                    "independent_of_reference", "reference_derived_or_shared", "unknown"}:
                raise ValueError(f"站点 {station_id} 的 reference_independence 无效")
            stations[station_id] = {
                "station_id": station_id,
                "name": (row.get("name") or station_id).strip(),
                "lat": lat,
                "lon": lon,
                "coordinate_crs": coordinate_crs,
                "voltage_kv": (row.get("voltage_kv") or "").strip(),
                "station_type": (row.get("station_type") or "").strip(),
                "source": (row.get("source") or "").strip(),
                "verified": (row.get("verified") or "").strip(),
                "point_role": point_role,
                "verification_status": verification_status,
                "reference_independence": reference_independence,
                "verification_reference": (row.get("verification_reference") or "").strip(),
                "ground_elevation_m": ground_elevation,
                "ground_elevation_source": (row.get("ground_elevation_source") or "").strip(),
                "vertical_datum": (row.get("vertical_datum") or "").strip(),
            }

    tasks = []
    route_ids = set()
    with open(route_tasks_csv, encoding="utf-8-sig", newline="") as f:
        reader = _dict_reader(
            f, "线路任务", {"route_id", "from_station_id", "to_station_id"})
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"线路任务表第 {row_number} 行字段数多于表头")
            route_id = (row.get("route_id") or "").strip()
            from_id = (row.get("from_station_id") or "").strip()
            to_id = (row.get("to_station_id") or "").strip()
            if not route_id or not from_id or not to_id:
                raise ValueError("线路任务表需要 route_id、from_station_id、to_station_id")
            if route_id in route_ids:
                raise ValueError(f"线路任务 ID 重复: {route_id}")
            if from_id == to_id:
                raise ValueError(f"线路任务起终站相同: {route_id}")
            if from_id not in stations or to_id not in stations:
                raise ValueError(f"线路任务 {route_id} 引用了站点表中不存在的 ID")
            start, end = stations[from_id], stations[to_id]
            route_voltage = (row.get("voltage_kv") or "").strip()
            if not route_voltage:
                if start["voltage_kv"] and end["voltage_kv"] and start["voltage_kv"] != end["voltage_kv"]:
                    raise ValueError(f"线路任务 {route_id} 两端电压等级不同，请在任务表填写线路 voltage_kv")
                route_voltage = start["voltage_kv"] or end["voltage_kv"]
            endpoints_declared_verified = all(
                station["point_role"] == "line_connection_point"
                and station["verification_status"] == "verified"
                and station["source"]
                and station["verification_reference"]
                for station in (start, end)
            )
            endpoint_qualification = (
                "declared_verified_line_connection_points"
                if endpoints_declared_verified else "exploratory_only_unverified_endpoints"
            )
            if not endpoints_declared_verified and not allow_unverified:
                raise ValueError(
                    f"线路任务 {route_id} 的两端必须声明为已核验线路接入点，"
                    "并填写 source 与 verification_reference；探索用途可显式使用 "
                    "--allow-unverified-endpoints"
                )
            tasks.append({
                "route_id": route_id,
                "from_station": start,
                "to_station": end,
                "start": (start["lat"], start["lon"]),
                "end": (end["lat"], end["lon"]),
                "voltage_kv": route_voltage,
                "endpoint_qualification": endpoint_qualification,
            })
            route_ids.add(route_id)
    if not tasks:
        raise ValueError("线路任务表为空")
    return tasks
