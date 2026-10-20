import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from resource_gate import violation


class ResourceGateTests(unittest.TestCase):
    def test_start_requires_five_gib(self):
        self.assertIsNotNone(violation(5119, 1200, 45, 0, startup=True))
        self.assertIsNone(violation(5120, 1200, 45, 0, startup=True))

    def test_running_ram_is_not_startup_ram(self):
        self.assertIsNone(violation(2048, 1200, 45, 1000))
        self.assertIsNotNone(violation(2047, 1200, 45, 1000))

    def test_gpu_temperature_and_process_limits(self):
        self.assertIsNone(violation(6000, 6700, 85, 10000))
        for values in [(6000, 6701, 45, 1000), (6000, 1200, 86, 1000),
                       (6000, 1200, 45, 10001)]:
            self.assertIsNotNone(violation(*values))

    def test_missing_or_nonfinite_measurements_fail_closed(self):
        for value in (None, float("nan"), float("inf"), -1):
            self.assertIsNotNone(violation(value, 1200, 45, 1000))


if __name__ == "__main__":
    unittest.main()
