# transmission_path_planner/__init__.py
"""输电线路智能路径规划系统 V5.0

面向固定站点间输电线路走廊初筛，支持 VIN-Grad、PPO 候选与 A* 基线。
真实线路仅在显式引导模式或路径生成后的事后评估中使用。
"""

from .config import *
from . import geometry, io_export, data, osm, cost, methods, smoothing, metrics, stations, visualization, cli
__version__ = "5.0"
__all__ = ["geometry", "io_export", "data", "osm", "cost", "methods",
           "smoothing", "metrics", "stations", "visualization", "cli"]
