import unittest
from pathlib import Path

from transmission_path_planner.stations import route_output_name


class SafeRouteOutputNameTests(unittest.TestCase):
    def test_route_id_cannot_escape_output_directory(self):
        name = route_output_name("../../outside/route")
        self.assertEqual(Path(name).name, name)
        self.assertNotIn("..", name)
        self.assertNotIn("/", name)
        self.assertNotIn("\\", name)

    def test_sanitized_collisions_keep_distinct_output_names(self):
        self.assertNotEqual(route_output_name("route/one"), route_output_name("route?one"))


if __name__ == "__main__":
    unittest.main()
