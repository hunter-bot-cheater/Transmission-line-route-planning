"""Thin adapter to the route search used by the V5 offline pipeline."""
from __future__ import annotations

import importlib.util
import math
import threading
from pathlib import Path
from typing import Any


_V5 = None
_PLANNER_IMPORT_ERROR = None
_V5_LOCK = threading.Lock()


def planner_status() -> dict[str, Any]:
    """Check required imports without loading the full V5 module during health checks."""
    if _V5 is not None:
        return {"ready": True}
    if _PLANNER_IMPORT_ERROR is not None:
        return {"ready": False, "error": _PLANNER_IMPORT_ERROR}
    missing = [name for name in ("numpy", "matplotlib")
               if importlib.util.find_spec(name) is None]
    if not missing:
        return {"ready": True}
    packages = "、".join(missing)
    return {
        "ready": False,
        "error": f"当前 Python 环境缺少路径规划依赖：{packages}。请运行 python -m pip install numpy matplotlib 后重启模拟器。",
    }


def _load_v5():
    global _V5, _PLANNER_IMPORT_ERROR
    if _V5 is not None:
        return _V5
    if _PLANNER_IMPORT_ERROR is not None:
        raise RuntimeError(_PLANNER_IMPORT_ERROR)
    with _V5_LOCK:
        if _V5 is not None:
            return _V5
        if _PLANNER_IMPORT_ERROR is not None:
            raise RuntimeError(_PLANNER_IMPORT_ERROR)
        path = Path(__file__).resolve().parents[1] / "scripts" / "ai_path_planning_v5.py"
        try:
            spec = importlib.util.spec_from_file_location("v5_route_planner", path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"无法加载 V5 路径规划模块：{path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except SystemExit as error:
            _PLANNER_IMPORT_ERROR = "V5 路径规划依赖不完整，请使用项目配置的 Python 环境启动模拟器"
            raise RuntimeError(_PLANNER_IMPORT_ERROR) from error
        except ImportError as error:
            _PLANNER_IMPORT_ERROR = f"V5 路径规划依赖不完整：{error}"
            raise RuntimeError(_PLANNER_IMPORT_ERROR) from error
        except Exception as error:
            _PLANNER_IMPORT_ERROR = f"V5 路径规划模块加载失败：{error}"
            raise RuntimeError(_PLANNER_IMPORT_ERROR) from error
        _V5 = module
        return _V5


def plan_path(observed: list[list[bool]], cost: list[list[float | None]],
              start: dict[str, int], goal: dict[str, int]) -> dict[str, Any]:
    """Run V5 A* on known cells only, preserving the simulator's full cost field."""
    height = len(observed)
    width = len(observed[0]) if height else 0
    if not width or any(len(row) != width for row in observed):
        raise ValueError("已观测矩阵必须为非空矩形")
    if len(cost) != height or any(len(row) != width for row in cost):
        raise ValueError("代价矩阵尺寸必须与已观测矩阵一致")

    sx, sy, gx, gy = (int(start["x"]), int(start["y"]), int(goal["x"]), int(goal["y"]))
    if not (0 <= sx < width and 0 <= gx < width and 0 <= sy < height and 0 <= gy < height):
        raise ValueError("起点或终点超出地图范围")
    if not observed[sy][sx] or not observed[gy][gx]:
        return {"path": [], "cost": None, "planner": "V5 A*", "waiting_for_scan": True}

    v5 = _load_v5()
    grid = v5.np.full((height, width), v5.np.inf, dtype=v5.np.float32)
    for y in range(height):
        for x in range(width):
            if not observed[y][x]:
                continue
            value = cost[y][x]
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value):
                continue
            if value < 0:
                raise ValueError("代价不能为负数")
            if value < 50:
                grid[y, x] = max(0.02, value)

    if not math.isfinite(float(grid[sy, sx])) or not math.isfinite(float(grid[gy, gx])):
        return {"path": [], "cost": None, "planner": "V5 A*", "waiting_for_scan": False}

    path_rc = v5.astar_search(grid, (sy, sx), (gy, gx))
    if not path_rc or path_rc[0] != (sy, sx) or path_rc[-1] != (gy, gx):
        return {"path": [], "cost": None, "planner": "V5 A*", "waiting_for_scan": False}

    total_cost = 0.0
    path = []
    for index, (row, column) in enumerate(path_rc):
        path.append({"x": int(column), "y": int(row)})
        if index:
            previous_row, previous_column = path_rc[index - 1]
            step = math.sqrt(2) if row != previous_row and column != previous_column else 1.0
            total_cost += float(grid[row, column]) * step
    return {"path": path, "cost": round(total_cost, 3), "planner": "V5 A*", "waiting_for_scan": False}
