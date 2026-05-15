import gradio as gr
from diffusers import FluxPipeline
import torch
from PIL import Image
import os

# Load model (use 'schnell' for faster inference)
model_id = "black-forest-labs/FLUX.1-schnell"

print("Loading model... This may take a while on first run.")
pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipeline.enable_model_cpu_offload()  # Good for lower VRAM

def generate_image(prompt, num_inference_steps=4, guidance_scale=0.0, height=1024, width=1024, seed=None):
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = None
    
    image = pipeline(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        max_sequence_length=512,
    ).images[0]
    
    return image

# Gradio Interface
with gr.Blocks(title="Fairy ✨ AI Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧚 Fairy - AI Image Generator")
    gr.Markdown("### Create magical images with FLUX.1 (Schnell)")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="A beautiful glowing fairy with sparkling wings flying over an enchanted forest at twilight, magical particles, highly detailed, cinematic lighting",
                lines=3
            )
            
            with gr.Row():
                steps = gr.Slider(1, 20, value=4, step=1, label="Inference Steps (4 is fastest)")
                guidance = gr.Slider(0.0, 3.5, value=0.0, step=0.1, label="Guidance Scale")
            
            with gr.Row():
                height = gr.Slider(512, 1440, value=1024, step=64, label="Height")
                width = gr.Slider(512, 1440, value=1024, step=64, label="Width")
            
            seed = gr.Number(label="Seed (optional for reproducibility)", value=None)
            
            generate_btn = gr.Button("✨ Generate Magic", variant="primary", size="large")
        
        with gr.Column(scale=2):
            output = gr.Image(label="Generated Image", height=600)
    
    gr.Examples(
        examples=[
            ["A majestic fairy queen sitting on a giant mushroom in a glowing forest"],
            ["Cyberpunk fairy with neon wings flying through rainy Tokyo streets"],
            ["Cute baby fairy playing with fireflies at night"]
        ],
        inputs=prompt
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, steps, guidance, height, width, seed],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(share=True)