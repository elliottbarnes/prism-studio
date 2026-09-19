"""A labeled, deterministic illustration for trying the workflow without model weights."""

import random

from .config import GenerationSettings


class DemoBackend:
    metadata = {
        "kind": "demo",
        "model_id": None,
        "revision": None,
        "device": "cpu",
        "precision": "not-applicable",
        "description": "Procedural preview. No AI model was loaded; prompt does not affect pixels.",
    }

    def render(self, settings: GenerationSettings):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (settings.width, settings.height), "#11172a")
        draw = ImageDraw.Draw(image)
        rng = random.Random(settings.seed)
        colors = ["#a4ffe3", "#9bd9ff", "#d5b5ff", "#ffc3d8"]
        for i in range(14):
            size = rng.randint(settings.width // 10, settings.width // 2)
            x, y = rng.randint(-size, settings.width), rng.randint(-size, settings.height)
            draw.ellipse((x, y, x + size, y + size), outline=colors[i % len(colors)], width=3)
        draw.rounded_rectangle((24, 24, 284, 72), radius=12, fill="#11172a", outline="#a4ffe3")
        draw.text((40, 42), "PRISM / PROCEDURAL DEMO / NO AI", fill="#a4ffe3")
        return image
