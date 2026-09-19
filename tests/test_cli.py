import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from prism_studio.cli import main


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
