"""genHouse3D online demo: a Gradio wrapper around ``housegen.demo`` (CPU only).

The Space folder is assembled by ``python -m housegen.space`` (``make space``): this file,
the ``housegen`` package, the EMA weights, the test graphs and two config files.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from housegen.demo import Demo

HERE = Path(__file__).parent
ENGINE = Demo.from_files(
    HERE / "model" / "ema.pt",
    HERE / "graphs_test.json",
    HERE / "configs" / "building.yaml",
    HERE / "configs" / "eval.yaml",
)
OUTPUT = Path(tempfile.mkdtemp(prefix="genhouse3d_"))

INTRO = """# genHouse3D
Choose room counts: a real room graph with those counts is drawn from the test set, a
diffusion model generates a **new** floor plan for it, and the plan is extruded into a 3D
house. [Code and evaluation](https://github.com/Wazoou/genHouse3D) ·
floor plans derived from Swiss Dwellings (Archilyse AG, CC BY 4.0)."""


def generate(bedrooms, bathrooms, kitchen, roof, pitch, seed, cutaway):
    result = ENGINE.run(
        int(bedrooms), int(bathrooms), kitchen, roof, float(pitch), int(seed), OUTPUT
    )
    if not result.valid:
        return str(result.plan_image), None, None, result.text
    shown = result.cutaway_glb if cutaway else result.house_glb
    return (
        str(result.plan_image),
        str(shown),
        [str(result.house_glb), str(result.cutaway_glb)],
        result.text,
    )


with gr.Blocks(title="genHouse3D") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=1):
            bedrooms = gr.Slider(minimum=0, maximum=4, value=2, step=1, label="Bedrooms")
            bathrooms = gr.Slider(minimum=1, maximum=3, value=1, step=1, label="Bathrooms")
            kitchen = gr.Radio(choices=["any", "open", "closed"], value="any", label="Kitchen")
            roof = gr.Radio(choices=["hip", "gable", "flat"], value="hip", label="Roof")
            pitch = gr.Slider(
                minimum=25, maximum=45, value=30, step=1, label="Roof pitch (degrees)"
            )
            seed = gr.Number(value=0, precision=0, label="Seed")
            cutaway = gr.Checkbox(value=False, label="Cutaway view (without the roof)")
            button = gr.Button(value="Generate", variant="primary")
        with gr.Column(scale=2):
            model = gr.Model3D(label="3D house", clear_color=(1.0, 1.0, 1.0, 1.0), height=480)
            with gr.Row():
                plan = gr.Image(label="Floor plan", type="filepath")
                files = gr.File(label="Download (GLB)", file_count="multiple")
            info = gr.Markdown()
    inputs = [bedrooms, bathrooms, kitchen, roof, pitch, seed, cutaway]
    outputs = [plan, model, files, info]
    button.click(generate, inputs=inputs, outputs=outputs)
    cutaway.change(generate, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch()
