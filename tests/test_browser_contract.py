"""Shared acceptance fixtures prevent browser and Python validation from drifting."""

import json
import unittest
from pathlib import Path

from prism_studio.config import GenerationSettings


class BrowserContractTests(unittest.TestCase):
    def test_shared_settings_contract(self):
        cases = json.loads((Path(__file__).parent / "web" / "cases.json").read_text())
        for case in cases:
            with self.subTest(case=case["name"]):
                settings = {"prompt": "A glass observatory above the clouds", **case["changes"]}
                if case["valid"]:
                    GenerationSettings(**settings)
                else:
                    with self.assertRaises(ValueError):
                        GenerationSettings(**settings)
