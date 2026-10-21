import unittest

import numpy as np

import ai_path_planning_v5 as planner


@unittest.skipUnless(planner.HAS_TORCH, "PyTorch is required for VIN tests")
class VINBellmanTests(unittest.TestCase):
    def setUp(self):
        self.torch = planner.torch
        self.operator = planner.DifferentiableMinPool()

    def test_blocked_wall_and_corner_cut_are_not_bellman_actions(self):
        torch = self.torch
        cost = torch.ones((1, 1, 3, 5))
        traversable = torch.ones_like(cost)
        value = torch.full_like(cost, 100.0)
        traversable[0, 0, 1, 2] = 0
        traversable[0, 0, 2, 2] = 0
        value[0, 0, 1, 2] = 0
        value[0, 0, 1, 3] = 0

        target, active = self.operator(value, cost, traversable, (1.0, 1.0))

        self.assertTrue(bool(active[0, 0, 1, 1]))
        self.assertGreaterEqual(float(target[0, 0, 1, 1]), 100.0)

    def test_bellman_step_uses_anisotropic_physical_distance(self):
        torch = self.torch
        cost = torch.ones((1, 1, 3, 3))
        traversable = torch.ones_like(cost)
        value = torch.full_like(cost, 100.0)
        value[0, 0, 1, 2] = 0

        target, _ = self.operator(value, cost, traversable, (2.0, 1.0))

        self.assertAlmostEqual(float(target[0, 0, 1, 1]), 1.0, places=5)

    def test_value_iteration_matches_anisotropic_grid_distance(self):
        torch = self.torch
        cost = torch.ones((1, 1, 5, 5))
        traversable = torch.ones_like(cost)
        goal = torch.zeros_like(cost)
        goal[0, 0, 2, 2] = 1
        value = torch.full_like(cost, float("inf"))
        value = torch.where(goal > 0, torch.zeros_like(value), value)

        for _ in range(2):
            value, _ = self.operator(value, cost, traversable, (2.0, 1.0))
            value = torch.where(goal > 0, torch.zeros_like(value), value)

        self.assertAlmostEqual(float(value[0, 0, 0, 0]), 2 * np.sqrt(5), places=5)

    def test_vin_keeps_goal_terminal_and_blocked_values_zero(self):
        torch = self.torch
        model = planner.VIN(n_extra=1, K=1, hidden=4).eval()
        cost = torch.ones((1, 1, 3, 3))
        goal = torch.zeros_like(cost)
        goal[0, 0, 2, 2] = 1
        extra = torch.zeros((1, 1, 3, 3))
        traversable = torch.ones_like(cost)
        traversable[0, 0, 1, 2] = 0

        value = model(cost, goal, extra, traversable, (2.0, 1.0))

        self.assertEqual(float(value[0, 0, 2, 2]), 0.0)
        self.assertEqual(float(value[0, 0, 1, 2]), 0.0)

    def test_training_returns_finite_masked_value_field(self):
        cost = np.ones((5, 5), dtype=np.float32)
        goal = np.zeros((5, 5), dtype=np.float32)
        goal[2, 2] = 1
        hard_mask = np.ones((5, 5), dtype=np.uint8)
        hard_mask[1, 2] = 0
        extra = np.zeros((5, 5, 1), dtype=np.float32)

        value = planner.train_vin(cost, goal, extra, hard_mask, (2.0, 1.0), n_steps=1)

        self.assertTrue(np.isfinite(value).all())
        self.assertEqual(float(value[2, 2]), 0.0)
        self.assertEqual(float(value[1, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
