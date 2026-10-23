"""Reproduce a small open-grid timing benchmark for the tower-graph router."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transmission_path_planner.experimental_tower_graph import tower_graph_route


def benchmark(size: int, repeats: int) -> dict:
    cost = np.ones((size, size), dtype=np.float32)
    traversable = np.ones((size, size), dtype=bool)
    tower_allowed = np.ones((size, size), dtype=bool)
    args = (cost, traversable, tower_allowed, (0, 0), (size - 1, size - 1),
            175.0, 350.0)
    kwargs = {"pixel_spacing": (35.0, 35.0), "candidate_stride": 2}

    warm_route = tower_graph_route(*args, **kwargs)
    if not warm_route:
        raise RuntimeError(f"no route found on the {size}x{size} open grid")

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        route = tower_graph_route(*args, **kwargs)
        timings.append(time.perf_counter() - started)
        if not route:
            raise RuntimeError(f"no route found on the {size}x{size} open grid")

    return {
        "size": size,
        "repeats": repeats,
        "elapsed_seconds": timings,
        "median_seconds": statistics.median(timings),
        "route_nodes": len(route),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        help="optional JSON file for raw timings and runtime details")
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 5 for size in args.sizes):
        parser.error("sizes must be >= 5 and repeats must be >= 1")

    import sys

    report = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "grid": "all-open, uniform unit resistance; 35m pixels; stride 2; span 175-350m",
        "results": [benchmark(size, args.repeats) for size in args.sizes],
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
