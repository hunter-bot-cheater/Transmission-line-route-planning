# transmission_path_planner/__init__.py
"""输电线路启发式规划系统 V5.0

面向固定站点间输电线路走廊初筛，默认使用确定性 A*。
VIN-Grad 与 PPO 仅可选作诊断，真实线路只用于规划后的独立评估。
"""

from .config import *
from . import geometry, io_export, data, osm, cost, methods, smoothing, metrics, stations, visualization, cli
__version__ = "5.0"
__all__ = ["geometry", "io_export", "data", "osm", "cost", "methods",
           "smoothing", "metrics", "stations", "visualization", "cli"]
