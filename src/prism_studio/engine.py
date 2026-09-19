"""One cached model, serialized inference, and an auditable record for every run."""

import hashlib
import importlib.metadata
import io
import json
import platform
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from . import __version__
from .config import MODEL_ID, GenerationSettings, RuntimeSettings, resolve_policy


class GenerationError(RuntimeError):
    """A generation failure suitable for CLI or UI display."""


class Backend(Protocol):
    metadata: dict[str, Any]

    def render(self, settings: GenerationSettings) -> Any: ...


def package_versions() -> dict[str, str]:
    versions = {"prism-studio": __version__, "python": platform.python_version()}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "pillow",
        "huggingface-hub",
        "invisible-watermark",
        "streamlit",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


@dataclass(frozen=True)
class Artifact:
    png: bytes
    manifest: dict[str, Any]

    @property
    def manifest_json(self) -> str:
        return json.dumps(self.manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"

    def save(self, directory: str | Path) -> Path:
        """Create a unique run directory; never overwrite an existing result."""
        run_dir = Path(directory) / self.manifest["run_id"]
        run_dir.mkdir(parents=True, exist_ok=False)
        try:
            (run_dir / "image.png").write_bytes(self.png)
            (run_dir / "manifest.json").write_text(self.manifest_json, encoding="utf-8")
        except OSError:
            # Only remove this call's incomplete output; existing runs are never touched.
            (run_dir / "image.png").unlink(missing_ok=True)
            (run_dir / "manifest.json").unlink(missing_ok=True)
            run_dir.rmdir()
            raise
        return run_dir


class GenerationService:
    def __init__(self, backend_factory: Callable[[], Backend]):
        self._factory = backend_factory
        self._backend: Backend | None = None
        self._lock = threading.Lock()

    def generate(self, settings: GenerationSettings) -> Artifact:
        # Include loading and PNG encoding: Diffusers schedulers and model state are mutable.
        with self._lock:
            started = time.perf_counter()
            try:
                if self._backend is None:
                    self._backend = self._factory()
                load_finished = time.perf_counter()
                image = self._backend.render(settings)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                png = buffer.getvalue()
                finished = time.perf_counter()
                manifest = {
                    "schema_version": 1,
                    "run_id": uuid.uuid4().hex,
                    "created_at": datetime.now(UTC).isoformat(),
                    "settings": asdict(settings),
                    "backend": self._backend.metadata,
                    "versions": package_versions(),
                    "platform": platform.platform(),
                    "timing_seconds": {
                        "load": round(load_finished - started, 4),
                        "render_and_encode": round(finished - load_finished, 4),
                        "total": round(finished - started, 4),
                    },
                    "image": {"file": "image.png", "sha256": hashlib.sha256(png).hexdigest()},
                    "reproducibility": (
                        "Seeds aid repeatability; hardware and library changes may alter pixels."
                    ),
                }
                return Artifact(png=png, manifest=manifest)
            except GenerationError:
                raise
            except ValueError as exc:
                raise GenerationError(str(exc)) from exc
            except (ImportError, ModuleNotFoundError) as exc:
                raise GenerationError(
                    "Inference dependencies are missing. Run uv sync --extra inference."
                ) from exc
            except Exception as exc:
                message = str(exc).lower()
                if "out of memory" in message:
                    raise GenerationError(
                        "Not enough memory. Reduce resolution, enable CUDA CPU offload, "
                        "or restart after closing other GPU applications."
                    ) from exc
                raise GenerationError(
                    "Generation failed. Check available memory, model access and cached files. "
                    "The CLI --debug flag prints the underlying error."
                ) from exc


class DiffusersBackend:
    def __init__(self, settings: RuntimeSettings):
        import torch
        from diffusers import StableDiffusionXLPipeline

        policy = resolve_policy(
            settings,
            cuda=torch.cuda.is_available(),
            mps=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        )
        self._torch = torch
        load_options = {
            "revision": settings.revision,
            "dtype": getattr(torch, policy["precision"]),
            "use_safetensors": True,
            "local_files_only": settings.local_files_only,
            "add_watermarker": True,
        }
        if policy["precision"] == "float16":
            load_options["variant"] = "fp16"
        # Use the built-in class and a fixed model repository: no custom_pipeline or remote code.
        self._pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, **load_options)
        self._pipe.enable_vae_tiling()
        if policy["cpu_offload"]:
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to(policy["device"])
        # Native PyTorch attention is retained; do not stack attention slicing on top of SDPA.
        self.metadata = {
            "kind": "diffusers",
            "model_id": MODEL_ID,
            "revision": settings.revision,
            "pipeline": "StableDiffusionXLPipeline",
            "scheduler": type(self._pipe.scheduler).__name__,
            "scheduler_config": json.loads(self._pipe.scheduler.to_json_string()),
            "generator_device": "cpu",
            "vae_tiling": True,
            "watermark": True,
            **policy,
        }

    def render(self, settings: GenerationSettings):
        # A fresh CPU generator for each call prevents accumulated RNG state across reruns.
        generator = self._torch.Generator(device="cpu").manual_seed(settings.seed)
        with self._torch.inference_mode():
            result = self._pipe(
                prompt=settings.prompt,
                negative_prompt=settings.negative_prompt or None,
                generator=generator,
                width=settings.width,
                height=settings.height,
                num_inference_steps=settings.steps,
                guidance_scale=settings.guidance,
                num_images_per_prompt=1,
                output_type="pil",
            )
        if not result.images:
            raise GenerationError("The model did not return an image. Try a different prompt.")
        return result.images[0]
