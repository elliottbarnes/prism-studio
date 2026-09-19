import unittest

from prism_studio.config import GenerationSettings, RuntimeSettings, resolve_policy


class SettingsTests(unittest.TestCase):
    def test_valid_settings_and_boundaries(self):
        settings = GenerationSettings("observatory", width=1536, height=1024, seed=2**32 - 1)
        self.assertEqual(settings.width, 1536)

    def test_invalid_settings_fail_before_loading(self):
        invalid = [
            {"prompt": " "},
            {"prompt": "x" * 2001},
            {"negative_prompt": None},
            {"seed": -1},
            {"seed": 2**32},
            {"seed": True},
            {"steps": 0},
            {"steps": 101},
            {"width": 513},
            {"height": 256},
            {"width": 1536, "height": 1536},
            {"guidance": float("nan")},
            {"guidance": float("inf")},
            {"guidance": 10**1000},
            {"guidance": -1},
            {"guidance": True},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                GenerationSettings(**{"prompt": "test", **changes})

    def test_auto_device_priority_and_precision(self):
        for cuda, mps, device, precision in [
            (True, True, "cuda", "float16"),
            (False, True, "mps", "float32"),
            (False, False, "cpu", "float32"),
        ]:
            actual = resolve_policy(RuntimeSettings(), cuda=cuda, mps=mps)
            self.assertEqual((actual["device"], actual["precision"]), (device, precision))

    def test_explicit_policy_does_not_silently_fall_back(self):
        for settings in [
            RuntimeSettings(device="cuda"),
            RuntimeSettings(device="mps"),
            RuntimeSettings(device="cpu", precision="float16"),
            RuntimeSettings(device="cpu", cpu_offload=True),
        ]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                resolve_policy(settings, cuda=False, mps=False)

    def test_revision_must_be_immutable(self):
        for revision in ("main", "latest", "abc123", "../model", "0" * 39):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                RuntimeSettings(revision=revision)
