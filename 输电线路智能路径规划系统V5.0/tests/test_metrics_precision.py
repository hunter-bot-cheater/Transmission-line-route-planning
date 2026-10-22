import unittest

from transmission_path_planner.metrics import compute_deviation


class MetricPrecisionTests(unittest.TestCase):
    def test_parallel_routes_use_local_projected_meters(self):
        planned = [(120.0, 24.0), (120.02, 24.0)]
        reference = [(120.0, 24.001), (120.02, 24.001)]

        metrics = compute_deviation(planned, reference)

        self.assertAlmostEqual(metrics["dev_hausdorff_km"], 0.111, delta=0.002)
        self.assertAlmostEqual(metrics["dev_mean_km"], 0.111, delta=0.002)
        self.assertEqual(metrics["deviation_sample_step_m"], 5)
        self.assertIn("+proj=aeqd", metrics["deviation_crs"])

    def test_symmetric_hausdorff_counts_reference_segments_between_vertices(self):
        planned = [(120.0, 24.0), (120.01, 24.0)]
        reference = [(120.0, 24.0), (120.005, 24.001), (120.01, 24.0)]

        metrics = compute_deviation(planned, reference)

        self.assertGreater(metrics["dev_haus_sym_km"], 0.05)
        self.assertEqual(metrics["dev_hausdorff_km"], metrics["dev_haus_sym_km"])


if __name__ == "__main__":
    unittest.main()
