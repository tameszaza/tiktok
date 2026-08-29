from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from tests.matrix_report.technique_shape import TECHNIQUES, TechniqueResult, build_shape_cases
from tests.technique_shape_report import write_outputs


class TechniqueShapeReportOutputTests(unittest.TestCase):
    def test_technique_result_keeps_legacy_positional_shape(self) -> None:
        shape = build_shape_cases()[0]
        result = TechniqueResult(shape, "00000", "PASS", 0, 1, 2.0, 1.0, 2.0)
        self.assertEqual(result.dtype, "float32")

    def test_lower_precision_outputs_are_self_describing_and_suffixed(self) -> None:
        shape = build_shape_cases()[0]
        results = [
            TechniqueResult(
                shape=shape,
                dtype="float16",
                technique=name,
                accuracy="PASS",
                failed=0,
                checked=1,
                baseline_ms=2.0,
                optimized_ms=1.0,
                speedup=2.0,
            )
            for name, _ in TECHNIQUES
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            write_outputs(output_dir, results, accuracy_trials=1, warmup=0, repeats=1, rounds=1)

            csv_path = output_dir / "technique_shape_results_float16.csv"
            image_path = output_dir / "technique_shape_float16.png"
            report_path = output_dir / "TECHNIQUE_SHAPE_REPORT_FLOAT16.md"
            self.assertTrue(csv_path.is_file())
            self.assertTrue(image_path.is_file())
            self.assertTrue(report_path.is_file())

            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(TECHNIQUES))
            self.assertEqual({row["dtype"] for row in rows}, {"float16"})

            report = report_path.read_text()
            self.assertIn("float16", report)
            self.assertIn(image_path.name, report)
            self.assertIn(csv_path.name, report)


if __name__ == "__main__":
    unittest.main()
