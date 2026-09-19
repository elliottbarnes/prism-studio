"""Runs with the optional UI extra, and never imports model dependencies."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch


class UiTests(unittest.TestCase):
    def test_demo_generation_and_validation(self):
        from streamlit.testing.v1 import AppTest

        launcher = Path(__file__).resolve().parents[1] / "generate_image_streamlit.py"
        with patch.dict(os.environ, {"PRISM_DEMO": "1"}):
            app = AppTest.from_file(str(launcher)).run()
            self.assertEqual(len(app.exception), 0)
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["artifact"].manifest["backend"]["kind"], "demo")
            self.assertEqual(
                [button.proto.label for button in app.get("download_button")],
                ["Download PNG", "Download manifest"],
            )
            self.assertTrue(app.session_state["artifact"].png.startswith(b"\x89PNG\r\n\x1a\n"))
            app.text_area[0].set_value(" ")
            app.button[0].click().run()
            self.assertEqual(len(app.error), 1)
            self.assertEqual(len(app.exception), 0)
