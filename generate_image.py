"""Compatibility wrapper. Install the project first; CLI: uv run prism --help."""

import io
from functools import lru_cache

from prism_studio.config import GenerationSettings, RuntimeSettings
from prism_studio.engine import DiffusersBackend, GenerationService


@lru_cache(maxsize=1)
def _service():
    return GenerationService(lambda: DiffusersBackend(RuntimeSettings()))


def generate_image(prompt):
    """Keep the original import API while using the shared, seeded SDXL backend."""
    from PIL import Image

    artifact = _service().generate(GenerationSettings(prompt))
    return Image.open(io.BytesIO(artifact.png)).copy()


if __name__ == "__main__":
    from prism_studio.cli import main

    raise SystemExit(main())
