"""Regenerate the clearly labeled offline workflow example; no model is involved."""

from pathlib import Path

from prism_studio.config import GenerationSettings
from prism_studio.demo import DemoBackend
from prism_studio.engine import GenerationService

artifact = GenerationService(DemoBackend).generate(
    GenerationSettings("Procedural workflow preview (not AI inference)", width=512, height=512)
)
destination = Path(__file__).resolve().parents[1] / "docs" / "examples"
destination.mkdir(parents=True, exist_ok=True)
(destination / "demo.png").write_bytes(artifact.png)
(destination / "manifest.json").write_text(artifact.manifest_json, encoding="utf-8")
print(f"Wrote labeled procedural demo to {destination}")
