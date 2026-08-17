"""
v3_dl: 端到端集成测试
版本: v3.20260525
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from rasterio.transform import from_origin

import config as cfg
from src.path_planning import (
    geo_to_grid, grid_to_geo, grid_to_geo_coords,
    compute_path_length_km, haversine_m,
    compute_value_function, extract_path_by_gradient,
    smooth_path, quality_gate,
)


class TestCoordinateConversion:
    """坐标转换测试"""

    def test_geo_to_grid(self):
        transform = from_origin(120.0, 25.0, 0.001, 0.001)
        r, c = geo_to_grid(24.5, 120.5, transform)
        assert r >= 0 and c >= 0

    def test_grid_to_geo_roundtrip(self):
        transform = from_origin(120.0, 25.0, 0.001, 0.001)
        lat, lon = grid_to_geo(100, 200, transform)
        r, c = geo_to_grid(lat, lon, transform)
        assert abs(r - 100) <= 1
        assert abs(c - 200) <= 1

    def test_grid_to_geo_coords(self):
        transform = from_origin(120.0, 25.0, 0.001, 0.001)
        grid_coords = [(10.5, 20.5), (30.0, 40.0)]
        geo = grid_to_geo_coords(grid_coords, transform)
        assert len(geo) == 2
        assert all(isinstance(c[0], float) for c in geo)


class TestPathLength:
    """路径长度计算测试"""

    def test_zero_length(self):
        coords = [(120.0, 23.0)]
        assert compute_path_length_km(coords) == 0.0

    def test_short_distance(self):
        coords = [(120.0, 23.0), (120.001, 23.0)]
        length = compute_path_length_km(coords)
        assert 0.1 < length < 0.15

    def test_haversine(self):
        d = haversine_m(120.0, 23.0, 120.001, 23.0)
        assert 80 < d < 150


class TestSmoothPath:
    """路径平滑测试"""

    def test_straight_line(self):
        coords = [(120.0, 23.0), (120.1, 23.0), (120.2, 23.0), (120.3, 23.0)]
        result = smooth_path(coords, rdp_epsilon=1000, resample_spacing=5000)
        assert len(result) >= 2

    def test_short_path_unchanged(self):
        coords = [(120.0, 23.0), (120.001, 23.0)]
        result = smooth_path(coords)
        assert len(result) == 2


class TestValueFunction:
    """值函数计算测试"""

    def test_value_shape(self):
        cost = np.random.rand(20, 20).astype(np.float32) * 0.5
        goal_mask = np.zeros((20, 20), dtype=np.uint8)
        goal_mask[18:21, 18:21] = 1
        value = compute_value_function(cost, goal_mask=goal_mask)
        assert value.shape == (20, 20)
        assert np.all(np.isfinite(value))

    def test_value_low_near_goal(self):
        """Goal region should have relatively low value"""
        cost = np.ones((30, 30), dtype=np.float32) * 0.5
        goal_mask = np.zeros((30, 30), dtype=np.uint8)
        goal_mask[25:30, 25:30] = 1
        value = compute_value_function(cost, goal_mask=goal_mask)
        # Values near goal should be lower than far from goal
        assert value[27, 27] < value[5, 5]


class TestGradientTracker:
    """神经梯度追踪测试"""

    def test_simple_diagonal(self):
        """Simple diagonal low-cost corridor"""
        H, W = 30, 30
        cost = np.ones((H, W), dtype=np.float32) * 0.5
        for i in range(H):
            for j in range(W):
                dist = abs(i - j) / np.sqrt(2)
                cost[i, j] = 0.1 + 0.8 * min(dist / 10, 1.0)

        hard_mask = np.ones((H, W), dtype=np.uint8)
        goal_mask = np.zeros((H, W), dtype=np.uint8)
        goal_mask[25:30, 25:30] = 1

        value = compute_value_function(cost, goal_mask=goal_mask)
        path = extract_path_by_gradient(
            value, cost, hard_mask, (2, 27), (27, 2),
            max_steps=200,
        )
        assert path is not None
        assert len(path) >= 2

    def test_with_obstacle(self):
        """Path should navigate around hard constraint zones"""
        H, W = 40, 40
        cost = np.ones((H, W), dtype=np.float32) * 0.3
        cost[15:25, 15:25] = 0.1  # Low cost center

        hard_mask = np.ones((H, W), dtype=np.uint8)
        hard_mask[15:25, :] = 0  # Block horizontal passage
        hard_mask[18:22, 18:22] = 1  # Small gap

        goal_mask = np.zeros((H, W), dtype=np.uint8)
        goal_mask[35:39, 35:39] = 1

        value = compute_value_function(cost, goal_mask=goal_mask)
        path = extract_path_by_gradient(
            value, cost, hard_mask, (5, 20), (35, 20),
            max_steps=300,
        )
        assert path is not None
        assert len(path) >= 2


class TestQualityGate:
    """质量门控测试"""

    def test_basic_checks(self):
        H, W = 50, 50
        cost = np.random.rand(H, W).astype(np.float32) * 0.3
        hard_mask = np.ones((H, W), dtype=np.uint8)
        transform = from_origin(120.0, 25.0, 0.001, 0.001)

        coords = [(120.01 + i * 0.001, 24.5 + i * 0.001) for i in range(20)]
        aligned = {"shape": (H, W)}
        result = quality_gate(coords, aligned, hard_mask, cost, transform, 10.0)
        assert "passed" in result
        assert "checks" in result
        assert "n_passed" in result


class TestFullPipelineMini:
    """微型端到端测试"""

    def test_mini_pipeline(self):
        """30x30微型场景完整流程: 值函数 → 梯度追踪 → 平滑 → 质量门控"""
        np.random.seed(42)
        H, W = 30, 30

        # Low-cost diagonal corridor
        cost = np.ones((H, W), dtype=np.float32) * 0.5
        for i in range(H):
            for j in range(W):
                dist = abs(i - j) / np.sqrt(2)
                cost[i, j] = 0.1 + 0.7 * min(dist / 12, 1.0)

        hard_mask = np.ones((H, W), dtype=np.uint8)
        transform = from_origin(120.0, 22.0, 0.01, 0.01)

        # Goal mask
        goal_mask = np.zeros((H, W), dtype=np.uint8)
        goal_mask[25:30, 25:30] = 1

        # Value function
        value = compute_value_function(cost, goal_mask=goal_mask)
        assert value.shape == (H, W)
        assert np.all(np.isfinite(value))

        # Gradient tracking
        path = extract_path_by_gradient(
            value, cost, hard_mask, (2, 27), (27, 2),
            max_steps=200,
        )
        assert path is not None
        assert len(path) >= 2

        # Smoothing
        smoothed = smooth_path(path)
        assert len(smoothed) >= 2

        # Path length
        length = compute_path_length_km(smoothed)
        assert length > 0

        # Quality gate
        aligned = {"shape": (H, W)}
        result = quality_gate(smoothed, aligned, hard_mask, cost, transform, length * 0.7)
        assert result["n_total"] == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
