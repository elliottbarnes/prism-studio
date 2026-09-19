import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prism_studio.cli import main
from prism_studio.engine import GenerationError


class CliTests(unittest.TestCase):
    def test_demo_writes_pair_of_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            result = main(
                ["test", "--demo", "--width", "512", "--height", "512", "--output", directory]
            )
            self.assertEqual(result, 0)
            self.assertEqual(len(list(Path(directory).glob("*/image.png"))), 1)
            self.assertEqual(len(list(Path(directory).glob("*/manifest.json"))), 1)

    def test_invalid_prompt_is_readable_error(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            self.assertEqual(main([" ", "--demo"]), 1)
        self.assertIn("non-empty prompt", output.getvalue())

    def test_invalid_output_fails_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output_file = Path(directory) / "occupied"
            output_file.write_text("existing content")
            error = io.StringIO()
            with (
                contextlib.redirect_stderr(error),
                patch("prism_studio.cli.GenerationService.generate") as generate,
            ):
                self.assertEqual(main(["test", "--demo", "--output", str(output_file)]), 1)
            generate.assert_not_called()
            self.assertIn("Prism:", error.getvalue())
            self.assertEqual(output_file.read_text(), "existing content")

    def test_generation_failure_does_not_report_saved_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            output, error = io.StringIO(), io.StringIO()
            with (
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(error),
                patch(
                    "prism_studio.cli.GenerationService.generate",
                    side_effect=GenerationError("Model could not load"),
                ),
            ):
                self.assertEqual(main(["test", "--demo", "--output", directory]), 1)
            self.assertNotIn("Saved", output.getvalue())
            self.assertIn("Model could not load", error.getvalue())
