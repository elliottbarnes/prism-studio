import hashlib
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from prism_studio.config import MODEL_ID, MODEL_REVISION, GenerationSettings, RuntimeSettings
from prism_studio.demo import DemoBackend
from prism_studio.engine import DiffusersBackend, GenerationError, GenerationService


class FakeImage:
    def save(self, file, format):
        file.write(b"fake-image")


class FakeBackend:
    metadata = {"kind": "fake", "model_id": "test-only"}

    def render(self, settings):
        return FakeImage()


class ServiceTests(unittest.TestCase):
    def test_model_is_loaded_once_and_calls_are_serial(self):
        class ConcurrentBackend(FakeBackend):
            active = 0
            peak = 0

            def render(self, settings):
                self.active += 1
                self.peak = max(self.peak, self.active)
                time.sleep(0.01)
                self.active -= 1
                return FakeImage()

        backend = ConcurrentBackend()
        factory = MagicMock(return_value=backend)
        service = GenerationService(factory)
        barrier = threading.Barrier(4)

        def generate(_):
            barrier.wait()
            return service.generate(GenerationSettings("test"))

        with ThreadPoolExecutor(max_workers=4) as executor:
            artifacts = list(executor.map(generate, range(4)))
        factory.assert_called_once()
        self.assertEqual(backend.peak, 1)
        self.assertEqual(len({a.manifest["run_id"] for a in artifacts}), 4)

    def test_manifest_and_no_overwrite(self):
        artifact = GenerationService(FakeBackend).generate(GenerationSettings("an observatory"))
        manifest = json.loads(artifact.manifest_json)
        self.assertEqual(manifest["settings"]["seed"], 42)
        self.assertEqual(manifest["image"]["sha256"], hashlib.sha256(artifact.png).hexdigest())
        self.assertIn("python", manifest["versions"])
        self.assertGreaterEqual(manifest["timing_seconds"]["total"], 0)
        with tempfile.TemporaryDirectory() as directory:
            path = artifact.save(directory)
            self.assertEqual((path / "image.png").read_bytes(), artifact.png)
            self.assertEqual(json.loads((path / "manifest.json").read_text()), manifest)
            with self.assertRaises(FileExistsError):
                artifact.save(directory)

    def test_loading_failure_can_be_retried(self):
        factory = MagicMock(side_effect=[RuntimeError("download failed"), FakeBackend()])
        service = GenerationService(factory)
        with self.assertRaises(GenerationError):
            service.generate(GenerationSettings("test"))
        self.assertEqual(service.generate(GenerationSettings("test")).png, b"fake-image")

    def test_actionable_memory_error(self):
        factory = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(GenerationError, "Reduce resolution"):
            GenerationService(factory).generate(GenerationSettings("test"))

    def test_demo_is_deterministic_and_explicitly_labeled(self):
        service = GenerationService(DemoBackend)
        a = service.generate(GenerationSettings("test", width=512, height=512))
        b = service.generate(GenerationSettings("another prompt", width=512, height=512))
        c = service.generate(GenerationSettings("test", seed=43, width=512, height=512))
        self.assertEqual(a.png, b.png)
        self.assertNotEqual(a.png, c.png)
        self.assertEqual(a.manifest["backend"]["kind"], "demo")
        self.assertIsNone(a.manifest["backend"]["model_id"])


class BackendTests(unittest.TestCase):
    def test_loader_policy_and_generation_contract_without_torch(self):
        pipeline = MagicMock()
        pipeline.scheduler.to_json_string.return_value = '{"name":"fake"}'
        pipeline.return_value = SimpleNamespace(images=[FakeImage()])
        loader = MagicMock(return_value=pipeline)
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            float16="float16",
            float32="float32",
            Generator=MagicMock(side_effect=lambda **_: MagicMock()),
            inference_mode=nullcontext,
        )
        diffusers = SimpleNamespace(
            StableDiffusionXLPipeline=SimpleNamespace(from_pretrained=loader)
        )
        with patch.dict("sys.modules", {"torch": torch, "diffusers": diffusers}):
            backend = DiffusersBackend(RuntimeSettings(cpu_offload=True, local_files_only=True))
            settings = GenerationSettings("test", seed=7)
            backend.render(settings)
            backend.render(settings)
        loader.assert_called_once_with(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype="float16",
            use_safetensors=True,
            local_files_only=True,
            add_watermarker=True,
            variant="fp16",
        )
        pipeline.enable_model_cpu_offload.assert_called_once()
        pipeline.to.assert_not_called()
        self.assertEqual(torch.Generator.call_count, 2)
        self.assertEqual(pipeline.call_args.kwargs["num_images_per_prompt"], 1)
        self.assertEqual(pipeline.call_args.kwargs["num_inference_steps"], 30)
        self.assertEqual(backend.metadata["revision"], MODEL_REVISION)
