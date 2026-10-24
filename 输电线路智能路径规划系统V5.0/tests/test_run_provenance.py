import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from transmission_path_planner.provenance import (
    capture_run_provenance,
    write_run_provenance,
)


class RunProvenanceTests(unittest.TestCase):
    def test_captures_planning_inputs_but_never_hashes_reference_geometry(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dem = root / "dem.tif"
            cases_csv = root / "cases.csv"
            dem.write_bytes(b"synthetic dem")
            cases_csv.write_text(
                "name,start_lat,start_lon,end_lat,end_lon\nR1,24,121,24.1,121.1\n",
                encoding="utf-8")

            cache_dir = root / "osm"
            cache_dir.mkdir()
            (cache_dir / "taiwan_buildings.pkl").write_bytes(b"buildings")
            (cache_dir / "taiwan_landuse.pkl").write_bytes(b"landuse")

            hazard_dir = root / "hazards"
            tile = hazard_dir / "impact" / "10" / "1" / "2.pbf"
            tile.parent.mkdir(parents=True)
            tile.write_bytes(b"synthetic tile")
            (hazard_dir / "manifest.json").write_text("{}", encoding="utf-8")
            (hazard_dir / "tiles.sha256").write_text("", encoding="utf-8")
            (hazard_dir / "must_not_hash.shp").write_bytes(b"reference geometry sentinel")

            reference_dir = root / "references"
            reference_dir.mkdir()
            reference_line = reference_dir / "R1_real_path.shp"
            reference_line.write_bytes(b"reference geometry sentinel")
            args = SimpleNamespace(
                stations_csv=None,
                route_tasks_csv=None,
                real_cases_csv=str(cases_csv),
                constraint_geojson=None,
                osm_cache_dir=str(cache_dir),
                official_hazards_dir=str(hazard_dir),
                include_ai_diagnostics=False,
                no_vin=False,
                use_costunet=False,
                real_dir=str(reference_dir),
                real_shp=None,
            )

            snapshot = capture_run_provenance(
                root, args, [("R1", (24.0, 121.0), (24.1, 121.1))],
                dem, cache_dir, has_torch=False)

            input_paths = {record["path"] for record in snapshot["planning_inputs"]}
            self.assertEqual(snapshot["configuration"]["real_dir"], str(reference_dir))
            self.assertNotIn(str(reference_line.resolve()), input_paths)
            self.assertNotIn(str((hazard_dir / "must_not_hash.shp").resolve()), input_paths)
            dem_record = next(record for record in snapshot["planning_inputs"]
                              if record["role"] == "dem")
            self.assertEqual(dem_record["sha256"], hashlib.sha256(b"synthetic dem").hexdigest())
            self.assertEqual(snapshot["reference_geometry_policy"]["hashed_by_provenance"], False)
            self.assertEqual(snapshot["cases"][0]["case"], "R1")

    def test_final_manifest_hashes_outputs_and_excludes_its_own_file_and_references(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "output"
            output_dir.mkdir()
            candidate = output_dir / "R1" / "final_route.geojson"
            candidate.parent.mkdir()
            candidate.write_text('{"type":"FeatureCollection"}', encoding="utf-8")
            (output_dir / "run_provenance.json").write_text("old", encoding="utf-8")
            reference_dir = output_dir / "references"
            reference_dir.mkdir()
            (reference_dir / "R1_real_path.shp").write_bytes(b"reference geometry sentinel")

            manifest_path = write_run_provenance(
                {"schema_version": 1}, output_dir, (reference_dir,))
            document = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest_path.name, "run_provenance.json")
            self.assertEqual(document["output_directory"], str(output_dir.resolve()))
            self.assertEqual([item["path"] for item in document["outputs"]],
                             ["R1/final_route.geojson"])
            self.assertEqual(
                document["outputs"][0]["sha256"],
                hashlib.sha256(candidate.read_bytes()).hexdigest())

    @unittest.skipUnless(shutil.which("git"), "Git is required for source-state test")
    def test_source_state_records_commit_and_dirty_status(self):
        with TemporaryDirectory() as temp_dir:
            repo = Path(temp_dir)
            subprocess.run(["git", "init", str(repo)], check=True,
                           capture_output=True, text=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"],
                           check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                           check=True, capture_output=True, text=True)
            tracked = repo / "source.py"
            tracked.write_text("value = 1\n", encoding="utf-8")
            project = repo / "project"
            package_source = project / "transmission_path_planner" / "planner.py"
            package_source.parent.mkdir(parents=True)
            package_source.write_text("route = 'astar'\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repo), "add", "source.py",
                            "project/transmission_path_planner/planner.py"],
                           check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-m", "initial"],
                           check=True, capture_output=True, text=True)

            args = SimpleNamespace(
                stations_csv=None, route_tasks_csv=None, real_cases_csv=None,
                constraint_geojson=None, osm_cache_dir=str(repo / "osm"),
                official_hazards_dir=str(repo / "hazards"),
                include_ai_diagnostics=False, no_vin=False, use_costunet=False,
                real_dir=None, real_shp=None,
            )
            clean = capture_run_provenance(
                project, args, [("R1", (24.0, 121.0), (24.1, 121.1))],
                repo / "dem.tif", repo / "osm", has_torch=False)["source"]
            self.assertTrue(clean["available"])
            self.assertTrue(clean["commit"])
            self.assertFalse(clean["dirty"])
            clean_source_hash = clean["runtime_source_files"][0]["sha256"]
            self.assertEqual(clean["runtime_source_files"][0]["path"],
                             "project/transmission_path_planner/planner.py")

            tracked.write_text("value = 2\n", encoding="utf-8")
            package_source.write_text("route = 'theta_star'\n", encoding="utf-8")
            dirty = capture_run_provenance(
                project, args, [("R1", (24.0, 121.0), (24.1, 121.1))],
                repo / "dem.tif", repo / "osm", has_torch=False)["source"]
            self.assertTrue(dirty["dirty"])
            self.assertTrue(any("source.py" in entry for entry in dirty["status_porcelain"]))
            self.assertNotEqual(dirty["runtime_source_files"][0]["sha256"],
                                clean_source_hash)


if __name__ == "__main__":
    unittest.main()
