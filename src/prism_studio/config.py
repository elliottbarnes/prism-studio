"""Validated settings; deliberately independent of inference dependencies."""

import math
import re
from dataclasses import dataclass

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# Verified against the public Hugging Face model API on 2026-09-18.
MODEL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"


@dataclass(frozen=True)
class GenerationSettings:
    prompt: str
    negative_prompt: str = ""
    seed: int = 42
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 7.0

    def __post_init__(self):
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("Describe an image with a non-empty prompt.")
        if not isinstance(self.negative_prompt, str):
            raise ValueError("Negative prompt must be text.")
        if len(self.prompt) > 2000 or len(self.negative_prompt) > 2000:
            raise ValueError("Prompts must be at most 2,000 characters.")
        for name, low, high in (("seed", 0, 2**32 - 1), ("steps", 1, 100)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name.title()} must be an integer from {low} to {high}.")
        for name in ("width", "height"):
            value = getattr(self, name)
            if type(value) is not int or not 512 <= value <= 1536 or value % 64:
                raise ValueError(f"{name.title()} must be 512–1536 in multiples of 64.")
        if self.width * self.height > 1_572_864:
            raise ValueError("Image area must not exceed 1.5 megapixels.")
        if (
            isinstance(self.guidance, bool)
            or not isinstance(self.guidance, (int, float))
            or not 0 <= self.guidance <= 20
            or not math.isfinite(self.guidance)
        ):
            raise ValueError("Guidance must be a finite number from 0 to 20.")


@dataclass(frozen=True)
class RuntimeSettings:
    device: str = "auto"
    precision: str = "auto"
    cpu_offload: bool = False
    local_files_only: bool = False
    revision: str = MODEL_REVISION

    def __post_init__(self):
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("Device must be auto, cuda, mps, or cpu.")
        if self.precision not in {"auto", "float16", "float32"}:
            raise ValueError("Precision must be auto, float16, or float32.")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[a-f0-9]{40}", self.revision):
            raise ValueError("Model revision must be a full 40-character commit SHA.")


def resolve_policy(settings: RuntimeSettings, *, cuda: bool, mps: bool) -> dict:
    device = settings.device
    if device == "auto":
        device = "cuda" if cuda else "mps" if mps else "cpu"
    if device == "cuda" and not cuda:
        raise ValueError("CUDA was requested but is not available. Check your PyTorch install.")
    if device == "mps" and not mps:
        raise ValueError("MPS was requested but is not available on this Python/PyTorch host.")
    precision = settings.precision
    if precision == "auto":
        precision = "float16" if device == "cuda" else "float32"
    if precision == "float16" and device != "cuda":
        raise ValueError("This baseline supports float16 on CUDA only; use float32 on MPS/CPU.")
    if settings.cpu_offload and device != "cuda":
        raise ValueError("CPU offload requires CUDA; disable it for MPS/CPU.")
    return {"device": device, "precision": precision, "cpu_offload": settings.cpu_offload}
