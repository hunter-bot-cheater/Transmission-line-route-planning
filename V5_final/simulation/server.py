"""Self-contained terrain-scan and route-planning demo server (Python stdlib only)."""
from __future__ import annotations

import hashlib
import json
import math
import random
import secrets
import socket
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from v5_adapter import plan_path as plan_v5_path, planner_status as get_planner_status
from geographic_route_api import GEOGRAPHIC_ROUTE_JOBS


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
LAYERS = (
    "elevation", "slope", "water", "road", "landuse", "building",
    "vegetation", "protected", "seismic", "landslide", "typhoon", "ice", "lightning",
    "aspect_cos", "aspect_sin", "tri", "tpi_100", "tpi_300", "tpi_900",
    "profile_curvature", "plan_curvature", "roughness_3", "roughness_9", "roughness_27",
    "dist_road", "dist_water", "dist_existing_line", "dist_railway", "dist_fault",
    "railway", "fault", "existing_line", "hard_blocked",
)
EXACT_LAYERS = {"road", "landuse", "protected", "railway", "fault", "existing_line", "hard_blocked"}
SENSOR_SIGMA = {
    "elevation": 0.012, "slope": 0.025, "water": 0.025, "building": 0.035,
    "vegetation": 0.03, "seismic": 0.035, "landslide": 0.035,
    "typhoon": 0.035, "ice": 0.035, "lightning": 0.035,
}
_SENSOR_NOISE_MASK = (1 << 16) - 1
_sensor_noise_rng = random.Random(0x5A17C0DE)
_sensor_noise_samples = tuple(_sensor_noise_rng.gauss(0.0, 1.0)
                              for _ in range(_SENSOR_NOISE_MASK + 1))
_sensor_noise_mean = math.fsum(_sensor_noise_samples) / len(_sensor_noise_samples)
_sensor_noise_std = math.sqrt(math.fsum((value - _sensor_noise_mean) ** 2
                                        for value in _sensor_noise_samples) / len(_sensor_noise_samples))
_SENSOR_NOISE_TABLE = tuple((value - _sensor_noise_mean) / _sensor_noise_std
                            for value in _sensor_noise_samples)
del _sensor_noise_samples, _sensor_noise_mean, _sensor_noise_std
MAX_SESSIONS = 4
MAX_SESSION_OWNERS = 32
MAX_SCAN_REQUEST_RECEIPTS = 8192
SESSION_CAPACITY_ERROR = "地图会话容量已满；请关闭其他地图标签页，或等待空闲会话过期后重试"
SESSION_LEASE_SECONDS = 90
SCAN_CONTROL_LEASE_SECONDS = 30
CELL_SIZE_METERS = 50
MAX_WORLD_WIDTH = 200
MAX_WORLD_HEIGHT = 144
MAX_SCAN_CANDIDATES = MAX_WORLD_WIDTH * MAX_WORLD_HEIGHT
GOAL_PROGRESS_GAIN = 6.0
GOAL_CORRIDOR_RADIUS = 500.0 / CELL_SIZE_METERS
LOCAL_SCAN_RADIUS = 225.0 / CELL_SIZE_METERS
SCAN_REFERENCE_CELL_METERS = 100.0
SERVICE_INSTANCE_ID = secrets.token_hex(8)
API_PROTOCOL = "server-authorized-scan-v4"
_SESSIONS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SESSIONS_LOCK = threading.Lock()
_WORLD_CREATE_LOCK = threading.Lock()
_WORLD_SEED_KEY = secrets.token_bytes(32)


class SessionControlConflict(Exception):
    """Raised when a page tries to write a session controlled by another lease."""

    def __init__(self, message: str, owner: str | None = None, epoch: int | None = None,
                 reason: str = "occupied"):
        super().__init__(message)
        self.owner = owner
        self.epoch = epoch
        self.reason = reason


class ScanRequestCapacityError(ValueError):
    """Raised when a session reaches its bounded request-id receipt capacity."""


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _scan_footprint_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    return tuple((dx, dy)
                 for dy in range(-radius, radius + 1)
                 for dx in range(-radius, radius + 1)
                 if dx * dx + dy * dy <= radius * radius)


def _sensor_noise(seed: int) -> float:
    """Return deterministic near-normal noise without constructing an RNG per layer read."""
    value = seed & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return _SENSOR_NOISE_TABLE[value & _SENSOR_NOISE_MASK]


def smooth_noise(rng: random.Random, width: int, height: int, block: int) -> list[list[float]]:
    coarse_w, coarse_h = math.ceil(width / block) + 1, math.ceil(height / block) + 1
    coarse = [[rng.random() for _ in range(coarse_w)] for _ in range(coarse_h)]
    result = []
    for y in range(height):
        row = []
        gy, fy = divmod(y, block)
        fy /= block
        for x in range(width):
            gx, fx = divmod(x, block)
            fx /= block
            top = coarse[gy][gx] * (1 - fx) + coarse[gy][gx + 1] * fx
            bottom = coarse[gy + 1][gx] * (1 - fx) + coarse[gy + 1][gx + 1] * fx
            row.append(top * (1 - fy) + bottom * fy)
        result.append(row)
    return result


def _distance_proxy(feature_layer: list[list[float]], width: int, height: int,
                    scale: float = 800.0 / CELL_SIZE_METERS) -> list[list[float]]:
    """Return a normalized 8-neighbor distance proxy from cells above 0.5."""
    diagonal = math.sqrt(2.0)
    distances = [[0.0 if feature_layer[y][x] > 0.5 else math.inf
                  for x in range(width)] for y in range(height)]
    for y in range(height):
        for x in range(width):
            distance = distances[y][x]
            if y:
                distance = min(distance, distances[y - 1][x] + 1.0)
                if x:
                    distance = min(distance, distances[y - 1][x - 1] + diagonal)
                if x + 1 < width:
                    distance = min(distance, distances[y - 1][x + 1] + diagonal)
            if x:
                distance = min(distance, distances[y][x - 1] + 1.0)
            distances[y][x] = distance
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            distance = distances[y][x]
            if y + 1 < height:
                distance = min(distance, distances[y + 1][x] + 1.0)
                if x:
                    distance = min(distance, distances[y + 1][x - 1] + diagonal)
                if x + 1 < width:
                    distance = min(distance, distances[y + 1][x + 1] + diagonal)
            if x + 1 < width:
                distance = min(distance, distances[y][x + 1] + 1.0)
            distances[y][x] = distance
    return [[round(clamp(distance / scale), 4) for distance in row] for row in distances]


def make_world(seed: int, width: int, height: int) -> dict[str, Any]:
    width = max(24, min(MAX_WORLD_WIDTH, int(width)))
    height = max(18, min(MAX_WORLD_HEIGHT, int(height)))
    rng = random.Random(seed)
    resolution_scale = 100.0 / CELL_SIZE_METERS
    broad = smooth_noise(rng, width, height, round(18 * resolution_scale))
    mid = smooth_noise(rng, width, height, round(8 * resolution_scale))
    fine = smooth_noise(rng, width, height, 3)
    base_width, base_height = width / resolution_scale, height / resolution_scale
    layers = {name: [] for name in LAYERS}
    for y in range(height):
        rows = {name: [] for name in LAYERS}
        for x in range(width):
            base_x, base_y = x / resolution_scale, y / resolution_scale
            ridge = abs(math.sin(base_x * 0.13 + base_y * 0.07 + seed * 0.01))
            valley_center = base_height * 0.52 + math.sin(base_x * 0.08 + seed * 0.017) * base_height * 0.13
            valley = math.exp(-((base_y - valley_center) / (base_height * 0.09)) ** 2)
            elev = clamp(0.10 + 0.42 * broad[y][x] + 0.21 * mid[y][x] + 0.13 * fine[y][x] + ridge * 0.12 - valley * 0.18)
            slope = clamp((abs(mid[y][x] - mid[max(0, y - 1)][x])
                           + abs(mid[y][x] - mid[y][max(0, x - 1)])) * 5.2 * resolution_scale
                          + ridge * 0.28)
            water = clamp(0.06 + valley * 0.52
                          + (0.08 if (int(base_x) + 2 * int(base_y) + seed) % 53 == 0 else 0))
            if base_width * 0.20 <= base_x <= base_width * 0.80 and abs(base_y - valley_center) < 0.58:
                water = 1.0
            road = 0.92 if abs(base_y - (base_height * 0.25 + math.sin(base_x * 0.12 + seed) * 3.5)) < 0.8 else 0.10 + fine[y][x] * 0.12
            landuse = (int(base_x) // max(1, int(base_width) // 6) + int(base_y) // max(1, int(base_height) // 5) + seed) % 6 / 5
            building = clamp(0.75 if ((int(base_x) + seed) % max(8, int(base_width) // 5) < 2 and (int(base_y) + seed // 3) % max(7, int(base_height) // 4) < 2) else fine[y][x] * 0.10)
            vegetation = clamp(0.18 + mid[y][x] * 0.42 + (0.18 if landuse < 0.55 else 0))
            protected = 0.92 if (broad[y][x] > 0.72 and elev > 0.42) else 0.04
            seismic = clamp(0.20 + abs(math.sin((base_x + base_y * 0.4 + seed) * 0.045)) * 0.64)
            landslide = clamp(0.10 + slope * 0.62 + elev * 0.12)
            typhoon = clamp(0.25 + broad[y][x] * 0.44 + abs(math.sin(base_x * 0.035 + seed)) * 0.18)
            ice = clamp((elev - 0.48) * 1.5 + slope * 0.12)
            lightning = clamp(0.18 + fine[y][x] * 0.28 + elev * 0.30)
            values = (elev, slope, water, road, landuse, building, vegetation, protected,
                      seismic, landslide, typhoon, ice, lightning)
            for name, value in zip(LAYERS, values):
                rows[name].append(round(value, 4))
            dx = (mid[y][min(width - 1, x + 1)] - mid[y][max(0, x - 1)]) * resolution_scale
            dy = (mid[min(height - 1, y + 1)][x] - mid[max(0, y - 1)][x]) * resolution_scale
            derivative = {
                "aspect_cos": 0.5 + 0.5 * math.cos(math.atan2(dy, dx)),
                "aspect_sin": 0.5 + 0.5 * math.sin(math.atan2(dy, dx)),
                "tri": clamp(slope * 0.72 + abs(fine[y][x] - mid[y][x]) * 2.0),
                "tpi_100": clamp(0.5 + (mid[y][x] - broad[y][x]) * 2.2),
                "tpi_300": clamp(0.5 + (fine[y][x] - broad[y][x]) * 1.7),
                "tpi_900": clamp(0.5 + (elev - 0.48) * 1.2),
                "profile_curvature": clamp(0.5 + (dy - dx) * 1.4),
                "plan_curvature": clamp(0.5 + (dx + dy) * 1.2),
                "roughness_3": clamp(slope * 0.7 + abs(fine[y][x] - mid[y][x]) * 2.0),
                "roughness_9": clamp(slope * 0.8 + abs(mid[y][x] - broad[y][x]) * 1.8),
                "roughness_27": clamp(slope * 0.85 + abs(elev - broad[y][x]) * 1.6),
                "dist_road": clamp(1.0 - road),
                "dist_water": clamp(1.0 - water),
                "railway": 0.9 if abs(base_x - (base_width * 0.72 + math.sin(base_y * 0.15) * 2.5)) < 0.65 else 0.03,
                "fault": 0.88 if abs(base_x - (base_width * 0.43 + math.sin(base_y * 0.09 + seed) * 3.0)) < 0.7 else 0.02,
                "existing_line": 0.92 if abs(base_x - (base_width * 0.57 + math.sin(base_y * 0.11) * 3.4)) < 0.65 else 0.02,
                "hard_blocked": 1.0 if water > 0.90 or slope > 0.96 or building > 0.96 else 0.0,
            }
            for name, value in derivative.items():
                rows[name].append(round(value, 4))
        for name in LAYERS:
            layers[name].append(rows[name])
    for feature_name in ("railway", "fault", "existing_line"):
        layers[f"dist_{feature_name}"] = _distance_proxy(layers[feature_name], width, height)
    endpoint_offset = max(1, round(200 / CELL_SIZE_METERS))
    start = {"x": endpoint_offset, "y": height - round(300 / CELL_SIZE_METERS)}
    goal = {"x": width - round(300 / CELL_SIZE_METERS), "y": endpoint_offset}

    def nearest_passable(point: dict[str, int]) -> dict[str, int]:
        return min(
            ({"x": x, "y": y} for y in range(height) for x in range(width)
             if layers["hard_blocked"][y][x] < 0.5),
            key=lambda cell: ((cell["x"] - point["x"]) ** 2 + (cell["y"] - point["y"]) ** 2,
                              cell["y"], cell["x"]),
        )

    start, goal = nearest_passable(start), nearest_passable(goal)
    return {
        "seed": seed, "width": width, "height": height, "layers": layers,
        "start": start, "goal": goal,
        "cell_size_m": CELL_SIZE_METERS,
        "layer_weights": {"slope": 0.24, "tri": 0.06, "roughness_9": 0.05, "landuse": 0.12,
                          "building": 0.16, "vegetation": 0.08, "protected": 0.16,
                          "seismic": 0.02, "landslide": 0.03, "typhoon": 0.01,
                          "ice": 0.01, "lightning": 0.01, "dist_water": 0.12,
                          "dist_fault": 0.08, "dist_railway": 0.025,
                          "dist_existing_line": 0.0,
                          "dist_road": 0.055},
    }


def _public_world(world: dict[str, Any], session_id: str, release_token: str) -> dict[str, Any]:
    public_world = {key: value for key, value in world.items() if key not in {"layers", "seed"}}
    public_world.update({"session_id": session_id, "instance_id": SERVICE_INSTANCE_ID,
                         "release_token": release_token,
                         "api_protocol": API_PROTOCOL,
                         "map_id": session_id[:8].upper(), "layer_names": list(LAYERS)})
    return public_world


def _private_world_seed(seed: int, width: int, height: int,
                        seed_key: bytes | None = None) -> int:
    """Keep client-requested resets stable without revealing the terrain seed."""
    request = f"{seed}:{width}:{height}".encode("ascii")
    key = _WORLD_SEED_KEY if seed_key is None else seed_key
    digest = hashlib.blake2b(request, key=key, digest_size=8).digest()
    return int.from_bytes(digest, "big")


def _refresh_session_owner(session: dict[str, Any], page_id: str | None,
                           now: float | None = None) -> None:
    if page_id is None:
        return
    now = time.monotonic() if now is None else now
    owners = session["owners"]
    tokens = session.setdefault("owner_release_tokens", OrderedDict())
    if page_id not in owners and len(owners) >= MAX_SESSION_OWNERS:
        raise RuntimeError("地图会话页面数量已达上限，请关闭其他地图标签页后重试")
    if page_id not in tokens and len(tokens) >= MAX_SESSION_OWNERS:
        stale_page = next((candidate for candidate in tokens if candidate not in owners), None)
        if stale_page is None:
            raise RuntimeError("地图会话页面数量已达上限，请关闭其他地图标签页后重试")
        tokens.pop(stale_page, None)
    owners.add(page_id)
    session.setdefault("owner_lease_until", {})[page_id] = now + SESSION_LEASE_SECONDS
    tokens.setdefault(page_id, secrets.token_urlsafe(24))


def _prune_session_owners(session: dict[str, Any], now: float) -> None:
    owner_leases = session.setdefault("owner_lease_until", {})
    for page_id in tuple(session["owners"]):
        lease_until = owner_leases.get(page_id, session["lease_until"])
        if lease_until <= now:
            session["owners"].discard(page_id)
            owner_leases.pop(page_id, None)
        else:
            owner_leases.setdefault(page_id, lease_until)


def _clear_session_control(session: dict[str, Any]) -> None:
    had_lease = session.get("control_page_id") is not None or session.get("control_token") is not None
    session["control_page_id"] = None
    session["control_token"] = None
    session["control_lease_until"] = 0.0
    if had_lease:
        session["control_epoch"] = session.get("control_epoch", 0) + 1
    session["active_scan_leg"] = None


def _existing_world_request(request_id: str | None, fingerprint: tuple[Any, ...],
                            now: float, page_id: str | None) -> dict[str, Any] | None:
    if request_id is None:
        return None
    for existing in _SESSIONS.values():
        if existing.get("world_request_id") == request_id:
            if existing["world_request_fingerprint"] != fingerprint:
                raise ValueError("地图创建请求编号已用于不同参数")
            _prune_session_owners(existing, now)
            existing["lease_until"] = now + SESSION_LEASE_SECONDS
            _refresh_session_owner(existing, page_id, now)
            _SESSIONS.move_to_end(existing["session_id"])
            release_token = (existing["owner_release_tokens"].get(page_id)
                             if page_id is not None else existing["release_token"])
            replay = _public_world(existing["truth"], existing["session_id"], release_token)
            replay["request_replayed"] = True
            return replay
    return None


def _store_world_session(session: dict[str, Any], replace_session_id: str | None,
                         replacement: dict[str, Any] | None, page_id: str | None) -> None:
    with _SESSIONS_LOCK:
        now = time.monotonic()
        for expired_id, existing in list(_SESSIONS.items()):
            if existing["lease_until"] <= now:
                _SESSIONS.pop(expired_id)
        replacing_current = replacement is not None and _SESSIONS.get(replace_session_id) is replacement
        if replacing_current:
            _prune_session_owners(replacement, now)
        replacement_will_be_removed = replacing_current and (
            not replacement["owners"]
            or (page_id is not None and replacement["owners"] <= {page_id})
        )
        if len(_SESSIONS) >= MAX_SESSIONS and not replacement_will_be_removed:
            raise RuntimeError(SESSION_CAPACITY_ERROR)
        if replacing_current:
            if page_id is not None:
                replacement["owners"].discard(page_id)
                replacement.get("owner_lease_until", {}).pop(page_id, None)
                replacement.get("owner_release_tokens", {}).pop(page_id, None)
                if replacement.get("control_page_id") == page_id:
                    _clear_session_control(replacement)
            if not replacement["owners"]:
                replacement["revoked"] = True
                _SESSIONS.pop(replace_session_id)
        _SESSIONS[session["session_id"]] = session


def create_world_session(seed: int, width: int, height: int,
                         replace_session_id: str | None = None,
                         request_id: str | None = None,
                         page_id: str | None = None,
                         *, _world_seed_key: bytes | None = None) -> dict[str, Any]:
    """Keep terrain truth on the server and return only public mission metadata."""
    if request_id is not None and (
        not isinstance(request_id, str) or not request_id or len(request_id) > 80
    ):
        raise ValueError("地图创建请求编号无效")
    if page_id is not None and not _valid_page_id(page_id):
        raise ValueError("页面实例编号无效")
    request_fingerprint = (seed, width, height, replace_session_id)
    with _WORLD_CREATE_LOCK:
        if request_id is not None:
            with _SESSIONS_LOCK:
                now = time.monotonic()
                for expired_id, existing in list(_SESSIONS.items()):
                    if existing["lease_until"] <= now:
                        _SESSIONS.pop(expired_id)
                replay = _existing_world_request(request_id, request_fingerprint, now, page_id)
                if replay is not None:
                    return replay
        with _SESSIONS_LOCK:
            now = time.monotonic()
            for expired_id, existing in list(_SESSIONS.items()):
                if existing["lease_until"] <= now:
                    _SESSIONS.pop(expired_id)
            replacement = _SESSIONS.get(replace_session_id)
            if replacement is not None:
                _prune_session_owners(replacement, now)
            replacement_will_be_removed = replacement is not None and (
                not replacement["owners"]
                or (page_id is not None and replacement["owners"] <= {page_id})
            )
            if len(_SESSIONS) >= MAX_SESSIONS and not replacement_will_be_removed:
                raise RuntimeError(SESSION_CAPACITY_ERROR)
        world = make_world(_private_world_seed(seed, width, height, _world_seed_key), width, height)
        session_id = secrets.token_urlsafe(18)
        observed_layers = {
            name: [[None] * world["width"] for _ in range(world["height"])] for name in LAYERS
        }
        lease_until = time.monotonic() + SESSION_LEASE_SECONDS
        session = {
            "truth": world,
            "layers": observed_layers,
            "cost": [[None] * world["width"] for _ in range(world["height"])],
            "samples": [[0] * world["width"] for _ in range(world["height"])],
            "scan_sequence": 0,
            "snapshot_revision": 0,
            "scan_requests": OrderedDict(),
            "scan_request_receipts": OrderedDict(),
            "drone_position": dict(world["start"]),
            "previous_drone_position": None,
            "scan_candidates": None,
            "scan_pattern_seed": secrets.randbits(32),
            "active_scan_leg": None,
            "mission_start": dict(world["start"]),
            "mission_goal": dict(world["goal"]),
            "route": [],
            "guidance_key": None,
            "guidance_grid": None,
            "lock": threading.RLock(),
            "lease_until": lease_until,
            "session_id": session_id,
            "release_token": secrets.token_urlsafe(24),
            "owners": {page_id} if page_id is not None else set(),
            "owner_lease_until": {page_id: lease_until} if page_id is not None else {},
            "owner_release_tokens": (OrderedDict({page_id: secrets.token_urlsafe(24)})
                                     if page_id is not None else OrderedDict()),
            "control_page_id": None,
            "control_token": None,
            "control_lease_until": 0.0,
            "control_epoch": 0,
            "flight_distance": 0.0,
            "revoked": False,
        }
        if request_id is not None:
            session["world_request_id"] = request_id
            session["world_request_fingerprint"] = request_fingerprint
        if replacement is None:
            _store_world_session(session, replace_session_id, None, page_id)
        else:
            # Wait for active work and preserve sessions owned by another page.
            with replacement["lock"]:
                _store_world_session(session, replace_session_id, replacement, page_id)
        release_token = (session["owner_release_tokens"].get(page_id)
                         if page_id is not None else session["release_token"])
        return _public_world(world, session_id, release_token)


def _get_session(session_id: str, *, renew: bool = True) -> dict[str, Any]:
    with _SESSIONS_LOCK:
        key = str(session_id)
        session = _SESSIONS.get(key)
        now = time.monotonic()
        if session is None or session["lease_until"] <= now:
            _SESSIONS.pop(key, None)
            raise ValueError("地图会话不存在或已过期，请重新生成地图")
        if renew:
            _prune_session_owners(session, now)
            session["lease_until"] = now + SESSION_LEASE_SECONDS
            _SESSIONS.move_to_end(key)
        return session


def _session_scan_snapshot_locked(session: dict[str, Any]) -> dict[str, Any]:
    height, width = len(session["samples"]), len(session["samples"][0])
    cells = []
    for y in range(height):
        for x in range(width):
            samples = session["samples"][y][x]
            if samples == 0:
                continue
            cells.append({
                "x": x, "y": y,
                "layers": {name: session["layers"][name][y][x] for name in LAYERS},
                "samples": samples,
                "confidence": round(samples / (samples + 2.0), 3),
            })
    return {
        "cells": cells,
        "scan_sequence": session["scan_sequence"],
        "snapshot_revision": session["snapshot_revision"],
        "mission_start": dict(session["mission_start"]),
        "mission_goal": dict(session["mission_goal"]),
        "drone_position": dict(session["drone_position"]),
        "previous_drone_position": (dict(session["previous_drone_position"])
                                     if session["previous_drone_position"] else None),
        "route": [dict(point) for point in session["route"]],
        "flight_distance": round(float(session.get("flight_distance", 0.0)), 6),
        "scan_request_capacity_reached": len(session["scan_request_receipts"]) >= MAX_SCAN_REQUEST_RECEIPTS,
    }


def session_scan_snapshot(session_id: str) -> dict[str, Any]:
    """Return only committed sensor observations and resumable mission state."""
    session = _get_session(session_id, renew=False)
    with session["lock"]:
        _ensure_session_not_replaced(session)
        return _session_scan_snapshot_locked(session)


def _ensure_session_not_replaced(session: dict[str, Any]) -> None:
    if session.get("revoked"):
        raise ValueError("地图会话已被替换，请重新生成地图")


def _authorize_session_write_locked(session: dict[str, Any], session_id: str,
                                    page_id: str, control_token: str | None, *,
                                    allow_new_owner: bool = False,
                                    claim_if_unowned: bool = True) -> tuple[str, bool, int]:
    """Check or acquire the sole write lease while the session lock is held."""
    if not _valid_page_id(page_id):
        raise SessionControlConflict("页面实例编号无效")
    key = str(session_id)
    with _SESSIONS_LOCK:
        now = time.monotonic()
        if _SESSIONS.get(key) is not session:
            raise ValueError("地图会话不存在或已过期，请重新生成地图")
        if session["lease_until"] <= now:
            _SESSIONS.pop(key, None)
            raise ValueError("地图会话不存在或已过期，请重新生成地图")
        _ensure_session_not_replaced(session)
        if not allow_new_owner and page_id not in session["owners"]:
            raise SessionControlConflict("当前页面已释放此地图会话，请重新加载地图",
                                         session.get("control_page_id"), session.get("control_epoch", 0),
                                         "page_released")
        current_page = session.get("control_page_id")
        current_token = session.get("control_token")
        expires = session.get("control_lease_until", 0.0)
        expired = current_page is not None and expires <= now
        if expired:
            _clear_session_control(session)
            current_page = None
        if current_page is None:
            if not claim_if_unowned:
                raise SessionControlConflict("地图控制租约已释放，请重新获取控制权",
                                             None, session.get("control_epoch", 0), "lease_expired")
            token = secrets.token_urlsafe(24)
            session["control_page_id"] = page_id
            session["control_token"] = token
            session["control_lease_until"] = now + SCAN_CONTROL_LEASE_SECONDS
            session["control_epoch"] = session.get("control_epoch", 0) + 1
            _prune_session_owners(session, now)
            session["lease_until"] = now + SESSION_LEASE_SECONDS
            _refresh_session_owner(session, page_id, now)
            _SESSIONS.move_to_end(key)
            return token, True, session["control_epoch"]
        if current_page != page_id or not isinstance(control_token, str) or not secrets.compare_digest(
            current_token or "", control_token
        ):
            raise SessionControlConflict("地图正由另一个页面控制；请等待控制权释放或租约过期",
                                         current_page, session.get("control_epoch", 0))
        session["control_lease_until"] = now + SCAN_CONTROL_LEASE_SECONDS
        _prune_session_owners(session, now)
        session["lease_until"] = now + SESSION_LEASE_SECONDS
        _refresh_session_owner(session, page_id, now)
        _SESSIONS.move_to_end(key)
        return current_token, False, session.get("control_epoch", 0)


def acquire_session_control(session_id: str, page_id: str,
                            control_token: str | None = None,
                            owner_token: str | None = None) -> dict[str, Any]:
    session = _get_session(session_id, renew=False)
    key = str(session_id)
    with session["lock"]:
        had_scan_state = session["scan_sequence"] > 0 or session["active_scan_leg"] is not None
        expected_owner_token = session.get("owner_release_tokens", {}).get(page_id, "")
        owner_authenticated = bool(
            isinstance(owner_token, str) and expected_owner_token
            and secrets.compare_digest(expected_owner_token, owner_token)
        )
        current_control_authenticated = bool(
            session.get("control_page_id") == page_id and control_token
            and secrets.compare_digest(session.get("control_token") or "", control_token)
        )
        if not owner_authenticated and not current_control_authenticated:
            raise SessionControlConflict("当前页面缺少有效的会话凭证",
                                         session.get("control_page_id"), session.get("control_epoch", 0),
                                         "page_released")
        if session.get("control_page_id") == page_id and not control_token:
            control_token = session.get("control_token")
        token, claimed, epoch = _authorize_session_write_locked(
            session, key, page_id, control_token, allow_new_owner=True,
        )
        return {**_session_scan_snapshot_locked(session), "control_token": token,
                "control_page_id": page_id, "control_epoch": epoch,
                "control_claimed": claimed,
                "control_taken_over": claimed and had_scan_state}


def release_session_control(session_id: str, page_id: str, control_token: str) -> bool:
    if not _valid_page_id(page_id) or not isinstance(control_token, str) or not control_token:
        return False
    key = str(session_id)
    try:
        session = _get_session(session_id, renew=False)
    except ValueError:
        return False
    with session["lock"]:
        with _SESSIONS_LOCK:
            if _SESSIONS.get(key) is not session:
                return False
            if session.get("control_page_id") != page_id or not secrets.compare_digest(
                session.get("control_token") or "", control_token
            ):
                return False
            _clear_session_control(session)
            return True


def touch_session(session_id: str, page_id: str | None = None,
                  control_token: str | None = None, *, register_owner: bool = True) -> bool:
    """Renew the map owner and only the matching page's existing control lease."""
    key = str(session_id)
    if page_id is not None and not _valid_page_id(page_id):
        return False
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(key)
    if session is None:
        return False
    with session["lock"]:
        with _SESSIONS_LOCK:
            now = time.monotonic()
            if _SESSIONS.get(key) is not session:
                return False
            if session["lease_until"] <= now:
                _SESSIONS.pop(key, None)
                return False
            _prune_session_owners(session, now)
            may_renew = page_id is not None and (register_owner or page_id in session["owners"])
            if may_renew:
                session["lease_until"] = now + SESSION_LEASE_SECONDS
                _refresh_session_owner(session, page_id, now)
            if (page_id is not None and control_token and session.get("control_page_id") == page_id
                    and secrets.compare_digest(session.get("control_token") or "", control_token)
                    and session.get("control_lease_until", 0.0) > now):
                session["control_lease_until"] = now + SCAN_CONTROL_LEASE_SECONDS
            _SESSIONS.move_to_end(key)
            return True


def session_control_status(session_id: str, page_id: str | None,
                           control_token: str | None) -> dict[str, Any]:
    try:
        session = _get_session(session_id, renew=False)
    except ValueError:
        return {"scan_control_page_id": None, "scan_controlled_by_you": False}
    with session["lock"]:
        now = time.monotonic()
        if session.get("control_page_id") is not None and session.get("control_lease_until", 0.0) <= now:
            _clear_session_control(session)
        owner = session.get("control_page_id")
        controlled = bool(
            page_id is not None and page_id == owner and control_token
            and secrets.compare_digest(session.get("control_token") or "", control_token)
        )
        return {"scan_control_page_id": owner, "scan_controlled_by_you": controlled,
                "control_epoch": session.get("control_epoch", 0),
                "snapshot_revision": session["snapshot_revision"]}


def release_world_session(session_id: str, release_token: str, page_id: str,
                          control_token: str | None = None) -> bool:
    """Release a page's session reference after its active work completes."""
    if not isinstance(release_token, str) or not release_token.isascii() or not _valid_page_id(page_id):
        return False
    key = str(session_id)
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(key)
    if session is None:
        return False
    with session["lock"]:
        with _SESSIONS_LOCK:
            if _SESSIONS.get(key) is not session:
                return False
            if page_id not in session["owners"]:
                return False
            owner_token = session.get("owner_release_tokens", {}).get(page_id)
            if not isinstance(owner_token, str) or not secrets.compare_digest(owner_token, release_token):
                return False
            if session.get("control_page_id") == page_id:
                _clear_session_control(session)
            session["owners"].remove(page_id)
            session.get("owner_lease_until", {}).pop(page_id, None)
            session.get("owner_release_tokens", {}).pop(page_id, None)
            if not session["owners"]:
                session["revoked"] = True
                _SESSIONS.pop(key)
            return True


def _session_matrices(session: dict[str, Any]) -> tuple[list[list[bool]], list[list[float | None]]]:
    samples = session["samples"]
    height, width = len(samples), len(samples[0])
    observed = [[samples[y][x] > 0 for x in range(width)] for y in range(height)]
    return observed, session["cost"]


def _cell_cost(layers: dict[str, list[list[float | None]]], x: int, y: int,
               weights: dict[str, float]) -> float | None:
    if any(layers[name][y][x] is None for name in LAYERS):
        return None
    n = lambda name: clamp(float(layers[name][y][x]))
    slope, water, building = n("slope"), n("water"), n("building")
    hard = bool(n("hard_blocked")) or water > .90 or slope > .96 or building > .96
    value = (.08 + slope * weights.get("slope", .20) + n("tri") * weights.get("tri", .06)
             + n("roughness_9") * weights.get("roughness_9", .05)
             + n("landuse") * weights.get("landuse", .12)
             + building * weights.get("building", .12)
             + n("vegetation") * weights.get("vegetation", .06)
             + n("protected") * weights.get("protected", .20)
             + n("seismic") * weights.get("seismic", .04)
             + n("landslide") * weights.get("landslide", .08)
             + n("typhoon") * weights.get("typhoon", .03)
             + n("ice") * weights.get("ice", .025)
             + n("lightning") * weights.get("lightning", .02)
             + (1 - n("dist_water")) * weights.get("dist_water", .12)
             + (1 - n("dist_fault")) * weights.get("dist_fault", .08)
             + (1 - n("dist_railway")) * weights.get("dist_railway", .025)
             - (1 - n("dist_road")) * weights.get("dist_road", .055))
    return 100.0 if hard else max(.02, value)


def build_cost_grid(layers: dict[str, list[list[float | None]]],
                    weights: dict[str, float] | None = None) -> list[list[float | None]]:
    """Convert sensed layer estimates to the simulator's multi-factor route cost."""
    weights = weights or {}
    height = len(layers["elevation"])
    width = len(layers["elevation"][0]) if height else 0
    result: list[list[float | None]] = [[None] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            result[y][x] = _cell_cost(layers, x, y, weights)
    return result


def _grid_coordinates(point: dict[str, int], label: str) -> tuple[int, int]:
    if (not isinstance(point, dict) or type(point.get("x")) is not int
            or type(point.get("y")) is not int):
        raise ValueError(f"{label}必须为整数坐标")
    return point["x"], point["y"]


def _request_radius(payload: dict[str, Any]) -> int:
    radius = payload.get("radius", 2)
    if type(radius) is not int:
        raise ValueError("扫描半径必须为整数")
    return radius


def _request_guidance_weight(payload: dict[str, Any]) -> float:
    value = payload.get("guidance_weight", 0.4)
    if type(value) not in (int, float):
        raise ValueError("引导权重必须为有限数值")
    try:
        weight = float(value)
    except OverflowError as error:
        raise ValueError("引导权重必须为有限数值") from error
    if not math.isfinite(weight):
        raise ValueError("引导权重必须为有限数值")
    return weight


def _request_scan_id(payload: dict[str, Any]) -> str | None:
    request_id = payload.get("request_id")
    if request_id is None:
        return None
    if not isinstance(request_id, str) or not request_id or len(request_id) > 80:
        raise ValueError("扫描请求编号无效")
    return request_id


def _request_leg_id(payload: dict[str, Any]) -> str:
    leg_id = payload.get("leg_id")
    if not isinstance(leg_id, str) or not leg_id or len(leg_id) > 80:
        raise ValueError("扫描航段授权编号无效")
    return leg_id


def _request_replace_session_id(payload: dict[str, Any]) -> str | None:
    session_id = payload.get("replace_session_id")
    if session_id is None:
        return None
    if not isinstance(session_id, str) or not session_id or len(session_id) > 80:
        raise ValueError("待替换地图会话编号无效")
    return session_id


def _request_world_id(payload: dict[str, Any]) -> str | None:
    request_id = payload.get("request_id")
    if request_id is None:
        return None
    if not isinstance(request_id, str) or not request_id or len(request_id) > 80:
        raise ValueError("地图创建请求编号无效")
    return request_id


def _valid_page_id(page_id: Any) -> bool:
    return isinstance(page_id, str) and bool(page_id) and len(page_id) <= 80 and page_id.isascii()


def _request_page_id(payload: dict[str, Any]) -> str | None:
    page_id = payload.get("page_id")
    if page_id is None:
        return None
    if not _valid_page_id(page_id):
        raise ValueError("页面实例编号无效")
    return page_id


def _request_control_token(payload: dict[str, Any], header: str | None = None) -> str | None:
    token = payload.get("control_token") or header
    if token is None:
        return None
    if not isinstance(token, str) or not token or len(token) > 100 or not token.isascii():
        raise ValueError("扫描控制凭证无效")
    return token


def _remember_scan_cell(session: dict[str, Any], undo: dict[tuple[int, int], tuple], x: int, y: int) -> None:
    key = (x, y)
    if key not in undo:
        undo[key] = (
            session["samples"][y][x], session["cost"][y][x],
            tuple(session["layers"][name][y][x] for name in LAYERS),
        )


def _rollback_scan_transaction(session: dict[str, Any], undo: dict[tuple[int, int], tuple],
                               scan_sequence: int, snapshot_revision: int) -> None:
    for (x, y), (samples, cost, layers) in undo.items():
        session["samples"][y][x] = samples
        session["cost"][y][x] = cost
        for name, value in zip(LAYERS, layers):
            session["layers"][name][y][x] = value
    session["scan_sequence"] = scan_sequence
    session["snapshot_revision"] = snapshot_revision


def sense_session(session_id: str, center: dict[str, int], radius: int = 2, *,
                  _session: dict[str, Any] | None = None,
                  _undo: dict[tuple[int, int], tuple] | None = None) -> dict[str, Any]:
    """Read one local footprint and fuse repeat measurements with a running mean."""
    if type(radius) is not int or radius < 0 or radius > 8:
        raise ValueError("扫描半径需在 0 到 8 之间")
    session = _session if _session is not None else _get_session(session_id)
    with session["lock"]:
        _ensure_session_not_replaced(session)
        world = session["truth"]
        x0, y0 = _grid_coordinates(center, "扫描中心")
        if not (0 <= x0 < world["width"] and 0 <= y0 < world["height"]):
            raise ValueError("扫描中心超出地图范围")
        sequence = session["scan_sequence"]
        sequence_before = session["scan_sequence"]
        revision_before = session["snapshot_revision"]
        owns_transaction = _undo is None
        undo = {} if owns_transaction else _undo
        cells = []
        try:
            session["scan_sequence"] += 1
            session["snapshot_revision"] += 1
            for dx, dy in _scan_footprint_offsets(radius):
                x, y = x0 + dx, y0 + dy
                if not (0 <= x < world["width"] and 0 <= y < world["height"]):
                    continue
                _remember_scan_cell(session, undo, x, y)
                previous_count = session["samples"][y][x]
                count = previous_count + 1
                fused = {}
                for layer_index, name in enumerate(LAYERS):
                    truth_value = float(world["layers"][name][y][x])
                    sigma = 0.0 if name in EXACT_LAYERS else SENSOR_SIGMA.get(name, 0.02)
                    seed = (world["seed"] * 73856093 + sequence * 19349663 + y * 83492791
                            + x * 2654435761 + layer_index * 97531) & 0xFFFFFFFF
                    reading = clamp(truth_value + _sensor_noise(seed) * sigma) if sigma else truth_value
                    old_value = session["layers"][name][y][x]
                    fused_value = reading if previous_count == 0 else float(old_value) + (reading - float(old_value)) / count
                    fused[name] = round(fused_value, 5)
                    session["layers"][name][y][x] = fused[name]
                session["samples"][y][x] = count
                session["cost"][y][x] = _cell_cost(session["layers"], x, y, world["layer_weights"])
                cells.append({"x": x, "y": y, "layers": fused, "samples": count,
                              "confidence": round(count / (count + 2.0), 3)})
            return {"cells": cells, "scan_sequence": sequence + 1}
        except Exception:
            if owns_transaction:
                _rollback_scan_transaction(session, undo, sequence_before, revision_before)
            raise


def sense_flight_segment(session_id: str, start: dict[str, int], end: dict[str, int],
                         radius: int = 2, spacing: float = 1.5,
                         request_id: str | None = None, *, leg_id: str | None = None,
                         require_authorization: bool = False, page_id: str | None = None,
                         control_token: str | None = None) -> dict[str, Any]:
    """Fuse sensor footprints sampled along the flown segment, including its endpoint."""
    if type(radius) is not int or radius < 0 or radius > 8:
        raise ValueError("扫描半径需在 0 到 8 之间")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("航段扫描间距必须为正数")
    x0, y0 = _grid_coordinates(start, "航段起点")
    x1, y1 = _grid_coordinates(end, "航段终点")
    if request_id is not None and (not isinstance(request_id, str) or not request_id or len(request_id) > 80):
        raise ValueError("扫描请求编号无效")
    if require_authorization and leg_id is None:
        raise ValueError("扫描航段授权编号无效")
    if leg_id is not None and (not isinstance(leg_id, str) or not leg_id or len(leg_id) > 80):
        raise ValueError("扫描航段授权编号无效")
    session = _get_session(session_id, renew=not require_authorization)
    with session["lock"]:
        _ensure_session_not_replaced(session)
        if require_authorization and page_id is None and session.get("control_page_id") is not None:
            raise ValueError("页面实例编号必填")
        world = session["truth"]
        if not (0 <= x0 < world["width"] and 0 <= x1 < world["width"]
                and 0 <= y0 < world["height"] and 0 <= y1 < world["height"]):
            raise ValueError("航段起点或终点超出地图范围")
        signature = (x0, y0, x1, y1, radius, spacing, leg_id if require_authorization else None)
        cached = session["scan_requests"].get(request_id) if request_id is not None else None
        if cached is not None:
            if cached[0] != signature:
                raise ValueError("扫描请求编号已用于其他航段")
            if require_authorization and page_id is not None:
                _authorize_session_write_locked(
                    session, session_id, page_id, control_token, claim_if_unowned=False,
                )
            session["scan_requests"].move_to_end(request_id)
            replay = dict(cached[1])
            replay["cells"] = []
            replay["scan_sequence"] = session["scan_sequence"]
            replay.pop("control_token", None)
            replay.pop("control_epoch", None)
            if require_authorization:
                replay["drone_position"] = dict(session["drone_position"])
                replay["flight_distance"] = round(float(session.get("flight_distance", 0.0)), 6)
            for cell in cached[1]["cells"]:
                x, y = cell["x"], cell["y"]
                samples = session["samples"][y][x]
                replay["cells"].append({**cell,
                    "layers": {name: session["layers"][name][y][x] for name in LAYERS},
                    "samples": samples,
                    "confidence": round(samples / (samples + 2.0), 3),
                })
            return replay
        receipt = session["scan_request_receipts"].get(request_id) if request_id is not None else None
        if receipt is not None:
            if receipt != signature:
                raise ValueError("扫描请求编号已用于其他航段")
            if require_authorization and page_id is not None:
                _authorize_session_write_locked(
                    session, session_id, page_id, control_token, claim_if_unowned=False,
                )
            return {"cells": [], "scan_sequence": session["scan_sequence"],
                    "sample_centers": 0, "replayed": True, "reconcile_snapshot": True,
                    "drone_position": dict(session["drone_position"])}
        token = None
        epoch = None
        authorized_distance = None
        leg = session.get("active_scan_leg")
        if require_authorization:
            if leg is None or leg_id != leg["leg_id"]:
                raise ValueError("当前没有匹配的扫描航段授权")
            if radius != leg["radius"]:
                raise ValueError("扫描半径与航段授权不一致")
            position = session["drone_position"]
            if (x0, y0) != (position["x"], position["y"]):
                raise ValueError("航段起点与服务端确认的无人机位置不一致")
            origin, target = leg["start"], leg["target"]
            dx, dy = target["x"] - origin["x"], target["y"] - origin["y"]
            length_squared = dx * dx + dy * dy
            if not length_squared:
                if (x1, y1) != (x0, y0):
                    raise ValueError("零距离扫描授权不允许移动")
            else:
                end_progress = ((x1 - origin["x"]) * dx + (y1 - origin["y"]) * dy) / length_squared
                cross_track = abs(dx * (y1 - origin["y"]) - dy * (x1 - origin["x"])) / math.sqrt(length_squared)
                position_progress = ((x0 - origin["x"]) * dx + (y0 - origin["y"]) * dy) / length_squared
                if cross_track > 0.75 or end_progress < position_progress:
                    raise ValueError("扫描航段必须沿授权目标方向连续前进")
                if end_progress > 1.001:
                    raise ValueError("扫描航段超出授权目标")
                authorized_distance = max(0.0, end_progress - position_progress) * math.sqrt(length_squared)
        if require_authorization and page_id is not None:
            token, _, epoch = _authorize_session_write_locked(
                session, session_id, page_id, control_token,
            )
        if request_id is not None and len(session["scan_request_receipts"]) >= MAX_SCAN_REQUEST_RECEIPTS:
            raise ScanRequestCapacityError("本地图的扫描请求编号已达上限，请重新生成地图")
        distance = math.hypot(x1 - x0, y1 - y0)
        steps = max(1, math.ceil(distance / spacing))
        cells = {}
        sequence = 0
        undo = {}
        sequence_before = session["scan_sequence"]
        revision_before = session["snapshot_revision"]
        try:
            for index in range(1, steps + 1):
                fraction = index / steps
                center = {
                    "x": round(x0 + (x1 - x0) * fraction),
                    "y": round(y0 + (y1 - y0) * fraction),
                }
                measurement = sense_session(session_id, center, radius, _session=session, _undo=undo)
                sequence = measurement["scan_sequence"]
                for cell in measurement["cells"]:
                    cells[(cell["x"], cell["y"])] = cell
            result = {"cells": list(cells.values()), "scan_sequence": sequence,
                      "sample_centers": steps}
        except Exception:
            _rollback_scan_transaction(session, undo, sequence_before, revision_before)
            raise
        if token is not None:
            result["control_token"] = token
            result["control_epoch"] = epoch
        if require_authorization:
            session["flight_distance"] = session.get("flight_distance", 0.0) + (
                authorized_distance if authorized_distance is not None else distance
            )
            result["flight_distance"] = round(session["flight_distance"], 6)
            session["drone_position"] = {"x": x1, "y": y1}
            if (x1, y1) == (leg["target"]["x"], leg["target"]["y"]):
                session["previous_drone_position"] = dict(leg["start"])
                session["active_scan_leg"] = None
            result["drone_position"] = dict(session["drone_position"])
        if request_id is not None:
            session["scan_requests"][request_id] = (signature, result)
            session["scan_request_receipts"][request_id] = signature
            while len(session["scan_requests"]) > 32:
                session["scan_requests"].popitem(last=False)
        return result


def plan_session(session_id: str, start: dict[str, int], goal: dict[str, int], *,
                 page_id: str | None = None, control_token: str | None = None) -> dict[str, Any]:
    start_coordinates = _grid_coordinates(start, "规划起点")
    goal_coordinates = _grid_coordinates(goal, "规划终点")
    session = _get_session(session_id, renew=page_id is None)
    with session["lock"]:
        _ensure_session_not_replaced(session)
        if (start_coordinates != _grid_coordinates(session["mission_start"], "当前任务起点")
                or goal_coordinates != _grid_coordinates(session["mission_goal"], "当前任务终点")):
            raise ValueError("规划请求端点与当前任务不一致")
        authorization = (_authorize_session_write_locked(session, session_id, page_id, control_token)
                         if page_id is not None else None)
        token = authorization[0] if authorization else None
        epoch = authorization[2] if authorization else None
        observed, cost = _session_matrices(session)
        result = plan_v5_path(observed, cost, start, goal)
        session["route"] = result["path"]
        session["snapshot_revision"] += 1
        if token is not None:
            result["control_token"] = token
            result["control_epoch"] = epoch
        return result


def _make_scan_candidates(width: int, height: int, start: dict[str, int],
                          pattern_seed: int = 0) -> list[dict[str, int]]:
    """Build one of four equally spaced sweep rasters without consulting hidden terrain."""
    vertical = bool(pattern_seed & 1)
    phase = (pattern_seed >> 1) & 1
    if vertical:
        lanes = sorted(set([0, width - 1, *range(phase, width, 6)]))
        if start["x"] > width / 2:
            lanes.reverse()
        sweep = sorted(set([1, height - 2, *range(1 + phase, height - 1, 4)]))
        reverse = start["y"] > height / 2
        points = []
        for x in lanes:
            ys = reversed(sweep) if reverse else sweep
            points.extend({"x": x, "y": y} for y in ys)
            reverse = not reverse
        return points

    lanes = sorted(set([0, height - 1, *range(phase, height, 6)]))
    if start["y"] > height / 2:
        lanes.reverse()
    sweep = sorted(set([1, width - 2, *range(1 + phase, width - 1, 4)]))
    reverse = start["x"] > width / 2
    points = []
    for y in lanes:
        xs = reversed(sweep) if reverse else sweep
        points.extend({"x": x, "y": y} for x in xs)
        reverse = not reverse
    return points


def configure_session_mission(session_id: str, start: dict[str, int],
                              goal: dict[str, int], *, page_id: str | None = None,
                              control_token: str | None = None,
                              keep_leg_id: str | None = None) -> dict[str, Any]:
    _grid_coordinates(start, "任务起点")
    _grid_coordinates(goal, "任务终点")
    if keep_leg_id is not None and (not isinstance(keep_leg_id, str)
                                    or not keep_leg_id or len(keep_leg_id) > 80):
        raise ValueError("保留扫描航段编号无效")
    session = _get_session(session_id, renew=page_id is None)
    with session["lock"]:
        _ensure_session_not_replaced(session)
        world = session["truth"]
        if not (0 <= start["x"] < world["width"] and 0 <= start["y"] < world["height"]
                and 0 <= goal["x"] < world["width"] and 0 <= goal["y"] < world["height"]):
            raise ValueError("任务起点或终点超出地图范围")
        authorization = (_authorize_session_write_locked(session, session_id, page_id, control_token)
                         if page_id is not None else None)
        token = authorization[0] if authorization else None
        epoch = authorization[2] if authorization else None
        session["mission_start"] = dict(start)
        session["mission_goal"] = dict(goal)
        active_leg = session.get("active_scan_leg")
        if active_leg is not None and keep_leg_id != active_leg["leg_id"]:
            session["active_scan_leg"] = None
        if session["scan_sequence"] == 0 and session["active_scan_leg"] is None:
            session["drone_position"] = dict(start)
            session["scan_candidates"] = _make_scan_candidates(
                world["width"], world["height"], start, session["scan_pattern_seed"]
            )
        route = session["route"]
        if route and (
            (route[0]["x"], route[0]["y"]) != (start["x"], start["y"])
            or (route[-1]["x"], route[-1]["y"]) != (goal["x"], goal["y"])
        ):
            session["route"] = []
            session["guidance_key"] = None
            session["guidance_grid"] = None
        session["snapshot_revision"] += 1
        result = {"start": dict(start), "goal": dict(goal),
                  "drone_position": dict(session["drone_position"])}
        if token is not None:
            result["control_token"] = token
            result["control_epoch"] = epoch
        return result


def select_session_scan(session_id: str, radius: int = 2, guidance_weight: float = 0.4,
                        goal_guided: bool = True, *, page_id: str | None = None,
                        control_token: str | None = None) -> dict[str, Any]:
    session = _get_session(session_id, renew=page_id is None)
    with session["lock"]:
        _ensure_session_not_replaced(session)
        world = session["truth"]
        start = dict(session["mission_start"])
        goal = dict(session["mission_goal"])
        if not 0 <= radius <= 8:
            raise ValueError("扫描半径需在 0 到 8 之间")
        if type(guidance_weight) not in (int, float) or not math.isfinite(guidance_weight):
            raise ValueError("引导权重必须为有限数值")
        authorization = (_authorize_session_write_locked(session, session_id, page_id, control_token)
                         if page_id is not None else None)
        token = authorization[0] if authorization else None
        epoch = authorization[2] if authorization else None
        if session["scan_sequence"] == 0 and session["active_scan_leg"] is None:
            session["scan_candidates"] = _make_scan_candidates(
                world["width"], world["height"], session["drone_position"],
                session["scan_pattern_seed"]
            )
        route = session["route"]
        if route and (
            (route[0]["x"], route[0]["y"]) != (start["x"], start["y"])
            or (route[-1]["x"], route[-1]["y"]) != (goal["x"], goal["y"])
        ):
            session["route"] = []
            session["guidance_key"] = None
            session["guidance_grid"] = None
            session["snapshot_revision"] += 1
        active = session["active_scan_leg"]
        if active is not None:
            result = dict(active["choice"])
            result["drone_position"] = dict(session["drone_position"])
            if token is not None:
                result["control_token"] = token
                result["control_epoch"] = epoch
            return result
        observed, cost = _session_matrices(session)
        drone = dict(session["drone_position"])
        previous = session["previous_drone_position"]
        mission_path = (session["route"] or [start, goal]) if goal_guided else None
        local_first = bool(session["route"])
        guidance_grid = None
        if mission_path:
            key = tuple((int(point["x"]), int(point["y"])) for point in mission_path)
            if session["guidance_key"] != key:
                session["guidance_grid"] = _mission_progress_grid(len(observed[0]), len(observed), mission_path)
                session["guidance_key"] = key
            guidance_grid = session["guidance_grid"]
        result = select_scan_target(observed, cost, session["scan_candidates"], drone, radius,
                                    guidance_weight, [start, goal], mission_path, guidance_grid,
                                    local_first=local_first, previous=previous)
        if result["target"] is not None:
            leg = {"leg_id": secrets.token_urlsafe(18), "start": drone,
                   "target": dict(result["target"]), "radius": radius}
            result["leg_id"] = leg["leg_id"]
            session["active_scan_leg"] = {**leg, "choice": dict(result)}
        result["drone_position"] = drone
        if token is not None:
            result["control_token"] = token
            result["control_epoch"] = epoch
        return result


def _mission_progress_grid(width: int, height: int,
                           mission_path: list[dict[str, int]]) -> list[list[float]]:
    points = [(int(point["x"]), int(point["y"])) for point in mission_path]
    if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
        raise ValueError("任务走廊超出地图范围")
    segments = [(x0, y0, x1 - x0, y1 - y0, (x1 - x0) ** 2 + (y1 - y0) ** 2)
                for (x0, y0), (x1, y1) in zip(points, points[1:])]
    result = [[0.0] * width for _ in range(height)]
    nearest_distances = [[GOAL_CORRIDOR_RADIUS] * width for _ in range(height)]
    if segments:
        for index, (x0, y0, dx, dy, length_squared) in enumerate(segments):
            min_x = max(0, math.floor(min(x0, x0 + dx) - GOAL_CORRIDOR_RADIUS))
            max_x = min(width - 1, math.ceil(max(x0, x0 + dx) + GOAL_CORRIDOR_RADIUS))
            min_y = max(0, math.floor(min(y0, y0 + dy) - GOAL_CORRIDOR_RADIUS))
            max_y = min(height - 1, math.ceil(max(y0, y0 + dy) + GOAL_CORRIDOR_RADIUS))
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    along = ((x - x0) * dx + (y - y0) * dy) / length_squared if length_squared else 0.0
                    along = clamp(along)
                    lateral = math.hypot(x - x0 - along * dx, y - y0 - along * dy)
                    if lateral < nearest_distances[y][x]:
                        nearest_distances[y][x] = lateral
                        result[y][x] = (index + along) / len(segments)
    else:
        x0, y0 = points[0]
        min_x = max(0, math.floor(x0 - GOAL_CORRIDOR_RADIUS))
        max_x = min(width - 1, math.ceil(x0 + GOAL_CORRIDOR_RADIUS))
        min_y = max(0, math.floor(y0 - GOAL_CORRIDOR_RADIUS))
        max_y = min(height - 1, math.ceil(y0 + GOAL_CORRIDOR_RADIUS))
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                distance = math.hypot(x - x0, y - y0)
                if distance < nearest_distances[y][x]:
                    nearest_distances[y][x] = distance
                    result[y][x] = 1.0

    for y in range(height):
        for x in range(width):
            nearest_distance = nearest_distances[y][x]
            if nearest_distance < GOAL_CORRIDOR_RADIUS:
                progress = result[y][x]
                result[y][x] = (1.0 - nearest_distance / GOAL_CORRIDOR_RADIUS) * (0.25 + 0.75 * progress)
    return result


def plan_path(observed: list[list[bool]], cost: list[list[float]], start: dict[str, int],
              goal: dict[str, int]) -> dict[str, Any]:
    """Use the V5 pipeline's A* search over cells the UAV has observed."""
    return plan_v5_path(observed, cost, start, goal)


def select_scan_target(observed: list[list[bool]], cost: list[list[float | None]],
                       candidates: list[dict[str, int]], drone: dict[str, int],
                       radius: int = 2, guidance_weight: float = 0.4,
                       priority_points: list[dict[str, int]] | None = None,
                       mission_path: list[dict[str, int]] | None = None,
                       mission_progress_grid: list[list[float]] | None = None,
                       local_first: bool = False,
                       previous: dict[str, int] | None = None) -> dict[str, Any]:
    """Choose a scan center using information gain and mission-path progress."""
    height = len(observed)
    width = len(observed[0]) if height else 0
    if not width or any(len(row) != width for row in observed):
        raise ValueError("已观测矩阵必须为非空矩形")
    if len(cost) != height or any(len(row) != width for row in cost):
        raise ValueError("代价矩阵尺寸必须与已观测矩阵一致")
    if radius < 0 or radius > 8:
        raise ValueError("扫描半径需在 0 到 8 之间")

    dx0, dy0 = int(drone["x"]), int(drone["y"])
    if not (0 <= dx0 < width and 0 <= dy0 < height):
        raise ValueError("无人机位置超出地图范围")
    heading = None
    if previous is not None:
        hx, hy = dx0 - int(previous["x"]), dy0 - int(previous["y"])
        heading_length = math.hypot(hx, hy)
        if heading_length:
            heading = (hx / heading_length, hy / heading_length)
    guidance_weight = clamp(float(guidance_weight), 0.0, 2.0)
    priorities = set()
    for point in priority_points or []:
        x, y = int(point["x"]), int(point["y"])
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("优先扫描点超出地图范围")
        priorities.add((x, y))
    unknown_cells = sum(not value for row in observed for value in row)
    goal_progress = mission_progress_grid or (
        _mission_progress_grid(width, height, mission_path) if mission_path else [[0.0] * width for _ in range(height)]
    )
    if len(goal_progress) != height or any(len(row) != width for row in goal_progress):
        raise ValueError("任务走廊引导矩阵尺寸必须与地图一致")

    gain_area = (CELL_SIZE_METERS / SCAN_REFERENCE_CELL_METERS) ** 2
    footprint = _scan_footprint_offsets(radius)
    scan_candidates = list(candidates)
    nearby = []
    endpoint_candidates = []
    for point in scan_candidates:
        x, y = int(point["x"]), int(point["y"])
        covers_unknown = any(0 <= x + ox < width and 0 <= y + oy < height
                             and not observed[y + oy][x + ox]
                             for ox, oy in footprint)
        if covers_unknown:
            distance = math.hypot(x - dx0, y - dy0)
            if distance <= LOCAL_SCAN_RADIUS:
                nearby.append(point)
        if covers_unknown and any((x + ox, y + oy) in priorities
                                  and not observed[y + oy][x + ox]
                                  for ox, oy in footprint
                                  if 0 <= x + ox < width and 0 <= y + oy < height):
            endpoint_candidates.append(point)
    if local_first and (nearby or endpoint_candidates):
        scan_candidates = list({(int(p["x"]), int(p["y"])): p
                                for p in nearby + endpoint_candidates}.values())
    best = None
    best_key = None
    best_data = None
    for index, point in enumerate(scan_candidates):
        x, y = int(point["x"]), int(point["y"])
        if not (0 <= x < width and 0 <= y < height):
            continue
        new_cells = 0
        weighted_gain = 0.0
        endpoint_gain = 0
        progress_gain = 0.0
        for ox, oy in footprint:
            nx, ny = x + ox, y + oy
            if 0 <= nx < width and 0 <= ny < height and not observed[ny][nx]:
                new_cells += 1
                weighted_gain += gain_area
                progress_gain += GOAL_PROGRESS_GAIN * guidance_weight * goal_progress[ny][nx] * gain_area
                if (nx, ny) in priorities:
                    endpoint_gain += 1
        # Endpoints must be sensed before a valid route can be returned.
        weighted_gain += endpoint_gain * 8.0 + progress_gain
        if not new_cells:
            continue
        travel = math.hypot(x - dx0, y - dy0)
        # Keep the local sweep coherent; score longer moves by gain per distance.
        turn_penalty = 0.0
        if heading and travel:
            direction = ((x - dx0) / travel, (y - dy0) / travel)
            turn_penalty = 1.5 * (1.0 - heading[0] * direction[0] - heading[1] * direction[1])
        travel_reference_cells = travel * CELL_SIZE_METERS / SCAN_REFERENCE_CELL_METERS
        score = weighted_gain / (1.0 + travel_reference_cells * (1.5 if not local_first else 1.0)
                                 + turn_penalty)
        key = ((1 if endpoint_gain else 0, -index, -turn_penalty, score, weighted_gain)
               if local_first and nearby else
               (1 if endpoint_gain else 0, -travel, score, weighted_gain, -index)
               if local_first else (score, weighted_gain, -travel, -turn_penalty, -index))
        if best_key is None or key > best_key:
            best_key = key
            best = {"x": x, "y": y}
            best_data = (new_cells, weighted_gain, endpoint_gain, progress_gain, travel, score)

    if best is None and unknown_cells:
        # The coarse S raster can leave isolated cells outside every remaining footprint.
        # Search scan centers near the drone so the residual pass does not fly to each cell.
        residual = sorted(({"x": x, "y": y} for y in range(height) for x in range(width)),
                          key=lambda point: math.hypot(point["x"] - dx0, point["y"] - dy0))
        result = select_scan_target(observed, cost, residual, drone, radius, guidance_weight,
                                    priority_points, mission_path, mission_progress_grid,
                                    local_first=local_first, previous=previous)
        result["residual_fallback"] = True
        return result
    if best is None:
        return {"target": None, "reason": "coverage_complete" if not unknown_cells else "candidate_coverage_exhausted",
                "new_cells": 0, "unknown_cells": unknown_cells}
    new_cells, weighted_gain, endpoint_gain, progress_gain, travel, score = best_data
    reason = "route_corridor" if mission_path and len(mission_path) > 2 else "goal_corridor" if mission_path else "frontier"
    return {"target": best, "reason": reason,
            "new_cells": new_cells, "weighted_gain": round(weighted_gain, 3),
            "priority_cells": endpoint_gain,
            "goal_progress_gain": round(progress_gain, 3),
            "travel_cells": round(travel, 2), "score": round(score, 4),
            "unknown_cells": unknown_cells}


class Handler(BaseHTTPRequestHandler):
    server_version = "TerrainFlightDemo/1.0"
    timeout = 10.0
    REQUEST_BODY_TIMEOUT_SECONDS = 10.0

    def log_message(self, fmt: str, *args: Any) -> None:
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def _json(self, status: int, value: Any, close: bool = False) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _reject_cross_origin(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is None:
            return False
        try:
            parsed = urlparse(origin)
            request = urlparse(f"http://{self.headers.get('Host', '')}")
            same_origin = (
                parsed.scheme == "http"
                and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                and parsed.hostname == request.hostname
                and parsed.port == request.port
                and not parsed.username and parsed.password is None
                and not parsed.path and not parsed.params and not parsed.query and not parsed.fragment
                and not request.username and request.password is None
                and not request.path and not request.params and not request.query and not request.fragment
            )
        except ValueError:
            same_origin = False
        if same_origin:
            return False
        self.close_connection = True
        self._json(403, {"error": "跨源请求被拒绝"}, close=True)
        return True

    def do_OPTIONS(self) -> None:
        if self._reject_cross_origin():
            return
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/routes/status":
            self._json(200, GEOGRAPHIC_ROUTE_JOBS.public_status())
            return
        if route.startswith("/api/routes/"):
            parts = route.strip("/").split("/")
            if len(parts) == 3:
                try:
                    self._json(200, GEOGRAPHIC_ROUTE_JOBS.get_status(parts[2]))
                except KeyError:
                    self._json(404, {"error": "任务不存在"})
                except (ValueError, RuntimeError) as error:
                    self._json(400, {"error": str(error)})
                return
            if len(parts) == 4:
                try:
                    artifact, extension = GEOGRAPHIC_ROUTE_JOBS.artifact(parts[2], parts[3])
                    body = artifact.read_bytes()
                except KeyError:
                    self._json(404, {"error": "任务或结果文件不存在"})
                    return
                except FileNotFoundError:
                    self._json(404, {"error": "该任务尚无此结果文件"})
                    return
                except (ValueError, RuntimeError) as error:
                    self._json(400, {"error": str(error)})
                    return
                content_type = {"geojson": "application/geo+json; charset=utf-8",
                                "png": "image/png", "csv": "text/csv; charset=utf-8",
                                "json": "application/json; charset=utf-8",
                                "txt": "text/plain; charset=utf-8"}[extension]
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._json(404, {"error": "任务或结果文件不存在"})
            return
        if route == "/api/health":
            health = {"ok": True, "name": "V5 Terrain Flight Simulator", "planner": "V5 A*",
                      "instance_id": SERVICE_INSTANCE_ID, "api_protocol": API_PROTOCOL}
            planner_status = get_planner_status()
            health["planner_ready"] = planner_status["ready"]
            if not planner_status["ready"]:
                health["planner_error"] = planner_status["error"]
            session_id = self.headers.get("X-Session-ID")
            if session_id is not None:
                page_id = self.headers.get("X-Page-ID")
                try:
                    control_token = _request_control_token({}, self.headers.get("X-Control-Token"))
                except ValueError:
                    control_token = None
                health["session_valid"] = touch_session(
                    session_id, page_id, control_token, register_owner=False,
                )
                control_status = session_control_status(session_id, page_id, control_token)
                health.update(control_status)
                if "snapshot_revision" not in control_status:
                    health["session_valid"] = False
            self._json(200, health)
            return
        target = WEB / ("index.html" if route == "/" else
                        "route_planner.html" if route == "/route-planner" else
                        route.lstrip("/"))
        try:
            target.resolve().relative_to(WEB.resolve())
            body = target.read_bytes()
        except (ValueError, FileNotFoundError, IsADirectoryError):
            self.send_error(404)
            return
        mime = "text/html; charset=utf-8" if target.suffix == ".html" else "text/javascript; charset=utf-8" if target.suffix == ".js" else "text/css; charset=utf-8" if target.suffix == ".css" else "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self._reject_cross_origin():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("请求长度无效")
            if length > 5_000_000:
                raise ValueError("请求过大")
            request_body = bytearray()
            body_deadline = time.monotonic() + self.REQUEST_BODY_TIMEOUT_SECONDS
            timed_out = False
            try:
                while len(request_body) < length:
                    remaining = body_deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        break
                    self.connection.settimeout(remaining)
                    try:
                        chunk = self.rfile.read1(min(65536, length - len(request_body)))
                    except socket.timeout:
                        timed_out = True
                        break
                    if not chunk:
                        break
                    request_body.extend(chunk)
            finally:
                self.connection.settimeout(self.timeout)
            if timed_out:
                self.close_connection = True
                self._json(408, {"error": "请求正文读取超时"}, close=True)
                return
            request_body = bytes(request_body)
            if len(request_body) != length:
                raise ValueError("请求正文长度与 Content-Length 不一致")
            request_body = request_body or b"{}"
            try:
                payload = json.loads(request_body)
            except RecursionError as error:
                raise ValueError("JSON嵌套层级过深") from error
            if not isinstance(payload, dict):
                raise ValueError("请求正文必须是 JSON 对象")
            route = urlparse(self.path).path
            page_id = _request_page_id(payload) or self.headers.get("X-Page-ID")
            if page_id is not None and not _valid_page_id(page_id):
                raise ValueError("页面实例编号无效")
            control_token = _request_control_token(payload, self.headers.get("X-Control-Token"))
            if route in {"/api/world", "/api/session/control", "/api/mission/points", "/api/plan",
                         "/api/scan/next", "/api/scan/path"} and page_id is None:
                raise ValueError("页面实例编号必填")
            if route == "/api/session/control":
                session_id = payload.get("session_id")
                if not isinstance(session_id, str) or not session_id or len(session_id) > 80:
                    raise ValueError("地图会话编号无效")
                action = payload.get("action")
                if action == "acquire":
                    self._json(200, acquire_session_control(
                        session_id, page_id, control_token, payload.get("release_token")
                    ))
                elif action == "release":
                    released = release_session_control(session_id, page_id, control_token or "")
                    self._json(200, {"released": released})
                else:
                    raise ValueError("扫描控制操作无效")
            elif route == "/api/session/release":
                session_id = payload.get("session_id")
                release_token = payload.get("release_token")
                page_id = page_id or payload.get("page_id")
                if not isinstance(session_id, str) or not session_id or len(session_id) > 80:
                    raise ValueError("地图会话编号无效")
                if not isinstance(release_token, str) or not release_token or len(release_token) > 80 or not release_token.isascii():
                    raise ValueError("地图会话释放凭证无效")
                if not _valid_page_id(page_id):
                    raise ValueError("页面实例编号无效")
                self._json(200, {"released": release_world_session(
                    session_id, release_token, page_id, control_token
                )})
            elif route == "/api/session/scan-state":
                session_id = payload.get("session_id")
                if not isinstance(session_id, str) or not session_id or len(session_id) > 80:
                    raise ValueError("地图会话编号无效")
                self._json(200, session_scan_snapshot(session_id))
            elif route == "/api/world":
                seed, width, height = (
                    payload.get("seed", 1), payload.get("width", 144), payload.get("height", 108)
                )
                if any(type(value) is not int for value in (seed, width, height)):
                    raise ValueError("地图种子、宽度和高度必须为整数")
                self._json(200, create_world_session(
                    seed, width, height, _request_replace_session_id(payload),
                    _request_world_id(payload), page_id,
                ))
            elif route == "/api/scan/measure":
                self._json(410, {"error": "任意位置测量接口已关闭，请先申请扫描航段并沿授权路径扫描"})
            elif route == "/api/scan/path":
                radius = _request_radius(payload)
                request_id = _request_scan_id(payload)
                if radius < 0 or radius > 8:
                    raise ValueError("扫描半径需在 0 到 8 之间")
                if request_id is None:
                    raise ValueError("扫描请求编号必填")
                result = sense_flight_segment(payload["session_id"], payload["start"], payload["end"],
                                              radius, request_id=request_id,
                                              leg_id=_request_leg_id(payload),
                                              require_authorization=True, page_id=page_id,
                                              control_token=control_token)
                self._json(200, result)
            elif route == "/api/mission/points":
                result = configure_session_mission(payload["session_id"], payload["start"], payload["goal"],
                                                   page_id=page_id, control_token=control_token,
                                                   keep_leg_id=payload.get("keep_leg_id"))
                self._json(200, result)
            elif route == "/api/plan":
                result = plan_session(payload["session_id"], payload["start"], payload["goal"],
                                      page_id=page_id, control_token=control_token)
                self._json(200, result)
            elif route == "/api/routes/validate-point":
                if not GEOGRAPHIC_ROUTE_JOBS.authorize(self.headers.get("X-Planner-Token")):
                    self._json(403, {"error": "真实规划访问口令无效"})
                    return
                self._json(200, GEOGRAPHIC_ROUTE_JOBS.validate_point(payload))
            elif route == "/api/routes":
                if not GEOGRAPHIC_ROUTE_JOBS.authorize(self.headers.get("X-Planner-Token")):
                    self._json(403, {"error": "真实规划访问口令无效"})
                    return
                self._json(202, GEOGRAPHIC_ROUTE_JOBS.submit(payload))
            elif route == "/api/scan/next":
                radius = _request_radius(payload)
                if radius < 0 or radius > 8:
                    raise ValueError("扫描半径需在 0 到 8 之间")
                result = select_session_scan(
                    payload["session_id"], radius, _request_guidance_weight(payload),
                    page_id=page_id, control_token=control_token,
                )
                self._json(200, result)
            else:
                self._json(404, {"error": "未知接口"})
        except ScanRequestCapacityError as error:
            self._json(409, {"error": str(error), "code": "scan_request_capacity"})
        except SessionControlConflict as error:
            self._json(409, {"error": str(error), "code": "control_conflict",
                             "control_page_id": error.owner, "control_epoch": error.epoch,
                             "control_reason": error.reason})
        except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})
        except RuntimeError as error:
            self._json(503, {"error": str(error)})


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="V5 无人机地形扫描与路径规划演示")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"V5 模拟器已启动: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
