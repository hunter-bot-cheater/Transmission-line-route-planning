import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.recompute_frozen_run_metrics import score, sha256


class RecomputeFrozenRunMetricsTests(unittest.TestCase):
    def test_score_manifest_tracks_computed_metric_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            candidate = results / "TW59_s0_g0"
            candidate.mkdir()
            route = {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": {
                    "type": "LineString", "coordinates": [[121.0, 24.0], [121.1, 24.1]]
                }}],
            }
            final_route = candidate / "final_route.geojson"
            astar_route = candidate / "planned_route_aastar.geojson"
            reference_route = candidate / "real_route.geojson"
            for path in (final_route, astar_route, reference_route):
                path.write_text(json.dumps(route), encoding="utf-8")

            freeze = {
                "reference_geometry_read": False,
                "candidates": [{
                    "case": "TW59_s0_g0",
                    "final_route": str(final_route),
                    "final_route_sha256": sha256(final_route),
                    "astar_route": str(astar_route),
                    "astar_route_sha256": sha256(astar_route),
                }],
            }
            (results / "metric_v5_candidate_freeze.json").write_text(
                json.dumps(freeze), encoding="utf-8")

            computed = {"route_similarity_version": 6, "route_similarity": {}}
            with patch("tools.recompute_frozen_run_metrics.compute_deviation", return_value=computed):
                metrics_path, manifest_path = score(results)

            self.assertEqual(metrics_path.name, "real_comparison_metrics_v6.json")
            self.assertEqual(manifest_path.name, "metric_v6_posthoc_manifest.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["metric_version"], 6)


if __name__ == "__main__":
    unittest.main()
