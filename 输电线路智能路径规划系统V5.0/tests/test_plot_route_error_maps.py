import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from affine import Affine
from shapely.geometry import LineString, MultiLineString

from transmission_path_planner import visualization


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "tools" / "plot_route_error_maps.py"
SPEC = importlib.util.spec_from_file_location("plot_route_error_maps", SCRIPT_PATH)
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


class RouteErrorMapTests(unittest.TestCase):
    def test_chain_error_profile_plot_shows_both_directions(self):
        profile = {
            "version": 1,
            "bin_count": 2,
            "directions": {
                "planned_to_reference": [
                    {"start_fraction": 0.0, "end_fraction": 0.5,
                     "p50_m": 40.0, "p95_m": 90.0,
                     "coverage_within_m": {"100": 0.9}},
                    {"start_fraction": 0.5, "end_fraction": 1.0,
                     "p50_m": 600.0, "p95_m": 900.0,
                     "coverage_within_m": {"100": 0.1}},
                ],
                "reference_to_planned": [
                    {"start_fraction": 0.0, "end_fraction": 0.5,
                     "p50_m": 45.0, "p95_m": 95.0,
                     "coverage_within_m": {"100": 0.85}},
                    {"start_fraction": 0.5, "end_fraction": 1.0,
                     "p50_m": 550.0, "p95_m": 850.0,
                     "coverage_within_m": {"100": 0.15}},
                ],
            },
        }
        before = set(visualization.plt.get_fignums())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "profile.png"
            self.assertTrue(plotter.plot_chain_error_profile("case", profile, output))
            self.assertGreater(output.stat().st_size, 0)
        self.assertEqual(set(visualization.plt.get_fignums()), before)

    def test_chain_error_profile_plot_skips_unknown_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "profile.png"
            self.assertFalse(plotter.plot_chain_error_profile(
                "case", {"version": 2, "directions": {}}, output))
            self.assertFalse(output.exists())

    def test_multifeature_reference_is_loaded_and_colored_in_full(self):
        document = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {}, "geometry": {
                    "type": "LineString", "coordinates": [[0, 0], [10, 0]]}},
                {"type": "Feature", "properties": {}, "geometry": {
                    "type": "MultiLineString", "coordinates": [
                        [[20, 0], [30, 0]], [[40, 0], [50, 0]],
                    ]}},
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.geojson"
            path.write_text(json.dumps(document), encoding="utf-8")
            reference = plotter.load_line(path)

        self.assertIsInstance(reference, MultiLineString)
        self.assertEqual(len(plotter.line_xy(reference)), 3)
        segments, values = plotter.color_segments(
            reference, LineString([(0, 100), (50, 100)]), interval_m=20.0)
        self.assertEqual(len(segments), 3)
        np.testing.assert_allclose(values, 0.1)

    def test_comparison_figure_closes_when_save_fails(self):
        before = set(visualization.plt.get_fignums())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "comparison.png"
            with patch.object(visualization.plt.Figure, "savefig",
                              side_effect=OSError("simulated disk or memory failure")):
                with self.assertRaisesRegex(OSError, "simulated"):
                    visualization.plot_comparison(
                        np.ones((3, 3), dtype=np.float32), Affine.identity(),
                        {"A*(baseline)": [(0.0, 0.0), (2.0, 2.0)]},
                        None, None, output)
        self.assertEqual(set(visualization.plt.get_fignums()), before)

    def test_comparison_figure_closes_when_raster_rendering_fails(self):
        before = set(visualization.plt.get_fignums())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "comparison.png"
            with patch("transmission_path_planner.visualization._fill",
                       side_effect=MemoryError("simulated raster memory failure")):
                with self.assertRaisesRegex(MemoryError, "simulated"):
                    visualization.plot_comparison(
                        np.ones((3, 3), dtype=np.float32), Affine.identity(),
                        {"A*(baseline)": [(0.0, 0.0), (2.0, 2.0)]},
                        None, None, output)
        self.assertEqual(set(visualization.plt.get_fignums()), before)

    def test_large_dem_and_hazard_are_downsampled_for_display_only(self):
        dem = np.ones((1201, 1201), dtype=np.float32)
        hazard = np.zeros_like(dem, dtype=np.uint8)
        hazard[1, 1] = 1
        hazard[2, 2] = 1
        hazard[1200, 1200] = 1
        hillshade_shapes = []
        image_shapes = []

        def record_hillshade(array):
            hillshade_shapes.append(array.shape)
            return np.zeros(array.shape, dtype=np.float32)

        def record_savefig(fig, *_args, **_kwargs):
            image_shapes.extend(image.get_array().shape for image in fig.axes[0].images)

        with patch("transmission_path_planner.visualization._hillshade",
                   side_effect=record_hillshade), patch.object(
                       visualization.plt.Figure, "savefig", autospec=True,
                       side_effect=record_savefig):
            visualization.plot_comparison(
                dem, Affine.identity(), {"A*(baseline)": [(0, 0), (1, 1)]},
                None, None, Path("unused.png"), hazard_masks={"debris_flow_impact": hazard})

        self.assertEqual(hillshade_shapes, [(601, 601)])
        self.assertEqual(image_shapes, [(601, 601)] * 3)
        display_hazard = visualization._block_max_for_display(hazard, 2)
        self.assertEqual(display_hazard[0, 0], 1)
        self.assertEqual(display_hazard[1, 1], 1)
        self.assertEqual(display_hazard[600, 600], 1)


if __name__ == "__main__":
    unittest.main()
