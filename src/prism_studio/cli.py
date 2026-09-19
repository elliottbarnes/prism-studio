"""Command line entry point. Help and demo mode never import torch."""

import argparse
import sys

from .config import MODEL_REVISION, GenerationSettings, RuntimeSettings
from .demo import DemoBackend
from .engine import DiffusersBackend, GenerationError, GenerationService


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prism Studio — save an image and its run manifest."
    )
    parser.add_argument("prompt", help="Description of the image")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--precision", choices=["auto", "float16", "float32"], default="auto")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--local-files-only", action="store_true", help="Never download model files"
    )
    parser.add_argument("--revision", default=MODEL_REVISION, help="Immutable model commit SHA")
    parser.add_argument(
        "--output", default="outputs", help="Parent folder for unique run directories"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Procedural preview: no model or network"
    )
    parser.add_argument("--debug", action="store_true", help="Show original error traceback")
    args = parser.parse_args(argv)
    try:
        settings = GenerationSettings(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            width=args.width,
            height=args.height,
            steps=args.steps,
            guidance=args.guidance,
        )
        runtime = RuntimeSettings(
            device=args.device,
            precision=args.precision,
            cpu_offload=args.cpu_offload,
            local_files_only=args.local_files_only,
            revision=args.revision,
        )
        service = GenerationService(DemoBackend if args.demo else lambda: DiffusersBackend(runtime))
        if args.demo:
            print(
                "Procedural demo: no AI inference; "
                "the prompt is recorded but does not affect pixels."
            )
        else:
            print(
                "Loading local SDXL runtime. "
                "The first run may download several GB of model weights."
            )
        run_dir = service.generate(settings).save(args.output)
        print(f"Saved {run_dir / 'image.png'}")
        print(f"Manifest {run_dir / 'manifest.json'}")
        return 0
    except (ValueError, GenerationError, OSError) as exc:
        if args.debug:
            raise
        print(f"Prism: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
