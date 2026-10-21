"""站点台账与站间线路任务输入。"""

import csv
import hashlib
import re


def route_output_name(route_id):
    """Return a filesystem-safe, collision-resistant directory name for a route ID."""
    route_id = str(route_id)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", route_id).strip("_-")[:64] or "route"
    suffix = hashlib.sha256(route_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem}-{suffix}"


def load_station_tasks(stations_csv, route_tasks_csv):
    """读取 stations.csv 与 route_tasks.csv，验证站点引用并返回规划任务。"""
    stations = {}
    with open(stations_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            station_id = (row.get("station_id") or "").strip()
            if not station_id:
                raise ValueError("站点表缺少 station_id")
            if station_id in stations:
                raise ValueError(f"站点 ID 重复: {station_id}")
            try:
                lat, lon = float(row["lat"]), float(row["lon"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"站点 {station_id} 的 lat/lon 无效") from exc
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"站点 {station_id} 的经纬度超出范围")
            stations[station_id] = {
                "station_id": station_id,
                "name": (row.get("name") or station_id).strip(),
                "lat": lat,
                "lon": lon,
                "voltage_kv": (row.get("voltage_kv") or "").strip(),
                "station_type": (row.get("station_type") or "").strip(),
                "source": (row.get("source") or "").strip(),
                "verified": (row.get("verified") or "").strip(),
            }

    tasks = []
    route_ids = set()
    with open(route_tasks_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
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
            tasks.append({
                "route_id": route_id,
                "from_station": start,
                "to_station": end,
                "start": (start["lat"], start["lon"]),
                "end": (end["lat"], end["lon"]),
                "voltage_kv": route_voltage,
            })
            route_ids.add(route_id)
    if not tasks:
        raise ValueError("线路任务表为空")
    return tasks
