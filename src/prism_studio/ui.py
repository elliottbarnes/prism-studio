"""Streamlit presentation; run state stays in the current browser session."""

import os

import streamlit as st

from prism_studio.config import GenerationSettings, RuntimeSettings
from prism_studio.demo import DemoBackend
from prism_studio.engine import DiffusersBackend, GenerationError, GenerationService


@st.cache_resource(show_spinner=False)
def get_service(demo: bool) -> GenerationService:
    # One runtime configuration per server, not per visitor, bounds model memory.
    if demo:
        return GenerationService(DemoBackend)
    runtime = RuntimeSettings(
        device=os.environ.get("PRISM_DEVICE", "auto"),
        precision=os.environ.get("PRISM_PRECISION", "auto"),
        cpu_offload=os.environ.get("PRISM_CPU_OFFLOAD", "0") == "1",
        local_files_only=os.environ.get("PRISM_LOCAL_FILES_ONLY", "0") == "1",
    )
    return GenerationService(lambda: DiffusersBackend(runtime))


def main():
    st.set_page_config(page_title="Prism Studio", page_icon="◈", layout="wide")
    st.caption("PRISM STUDIO / LOCAL IMAGE LAB")
    st.title("An idea. An image. Every detail saved.")
    st.write("Explore a prompt, hold the seed steady, and take the whole experiment with you.")
    demo = os.environ.get("PRISM_DEMO", "0") == "1"
    if demo:
        st.info(
            "Demo mode · procedural shapes, no AI model, no downloads. "
            "Prompts do not affect the preview."
        )
    else:
        st.caption("SDXL 1.0 · pinned model revision · one image at a time · runs on this machine")
    controls, canvas = st.columns([1, 1.35], gap="large")
    with controls:
        with st.form("generation"):
            prompt = st.text_area(
                "What do you imagine?",
                "A tiny observatory above a sea of clouds, "
                "soft morning light, editorial illustration",
                height=140,
                max_chars=2000,
            )
            negative = st.text_input("Leave out (optional)", max_chars=2000)
            shape = st.selectbox(
                "Canvas",
                [
                    "Square · 1024 × 1024",
                    "Landscape · 1216 × 832",
                    "Portrait · 832 × 1216",
                    "Small · 512 × 512",
                ],
            )
            seed = st.number_input("Seed", min_value=0, max_value=2**32 - 1, value=42, step=1)
            with st.expander("Fine-tune the run"):
                steps = st.slider("Steps", 1, 100, 30)
                guidance = st.slider("Prompt guidance", 0.0, 20.0, 7.0, 0.5)
                st.caption(
                    "Higher values do not always produce better images. "
                    "Long prompts may be truncated by the model tokenizer."
                )
            submitted = st.form_submit_button(
                "Make a preview" if demo else "Generate image", type="primary", width="stretch"
            )
        st.caption(
            "First model run downloads several GB. CPU generation is slow; "
            "sufficient RAM is required. No hosted API is used."
        )
        st.caption(
            "Manifests include your prompt. Review them before sharing. "
            "SDXL has no built-in content moderation; this is a local creative tool."
        )
    if submitted:
        width, height = {
            "Square": (1024, 1024),
            "Landscape": (1216, 832),
            "Portrait": (832, 1216),
            "Small": (512, 512),
        }[shape.split(" ·")[0]]
        try:
            settings = GenerationSettings(
                prompt, negative, int(seed), width, height, steps, guidance
            )
            with st.spinner(
                "Making your preview…"
                if demo
                else "Loading / generating… other runs may be ahead of you."
            ):
                st.session_state["artifact"] = get_service(demo).generate(settings)
        except (ValueError, GenerationError) as exc:
            st.error(str(exc))
            if "artifact" in st.session_state:
                st.caption("The previous successful result remains below.")
    with canvas:
        artifact = st.session_state.get("artifact")
        if artifact:
            st.image(
                artifact.png,
                caption="Procedural demo · not AI generated" if demo else "Your latest run",
                width="stretch",
            )
            left, right = st.columns(2)
            left.download_button(
                "Download PNG", artifact.png, "prism-image.png", "image/png", width="stretch"
            )
            right.download_button(
                "Download manifest",
                artifact.manifest_json,
                "prism-manifest.json",
                "application/json",
                width="stretch",
            )
            with st.expander("Run details"):
                st.json(artifact.manifest)
        else:
            with st.container(border=True):
                st.subheader("A little room for imagination.")
                st.write(
                    "Your image will appear here. Every run includes its seed, settings, "
                    "model revision, timing and image fingerprint."
                )
                st.caption("Tip: keep the seed fixed to compare prompt changes.")


if __name__ == "__main__":
    main()
