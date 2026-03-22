import gradio as gr
import numpy as np
import random
import torch
# spaces import REMOVED — ZeroGPU is HuggingFace-only infrastructure, not needed locally


import gc

from safetensors.torch import load_file
# hf_hub_download import REMOVED — replaced with local path constants below


from PIL import Image
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPlusPipeline, EulerAncestralDiscreteScheduler, FlowMatchEulerDiscreteScheduler
# from optimization import optimize_pipeline_
# from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
# from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
# from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

# InferenceClient import REMOVED — was calling Nebius/HuggingFace cloud API
# Replaced below with a local Qwen2.5-VL model running on the same GPU
import math

import os
import base64
from io import BytesIO
import json

# ---------------------------------------------------------------------------
# LOCAL MODEL PATHS — edit these or set the env vars on your VPS.
#
# Pre-download with huggingface-cli ONCE, then everything runs offline:
#
#   huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
#       --local-dir /models/Qwen-Image-Edit-2511
#
#   huggingface-cli download Phr00t/Qwen-Image-Edit-Rapid-AIO \
#       --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" \
#       --local-dir /models/rapid-aio
#
#   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
#       --local-dir /models/Qwen2.5-VL-7B-Instruct
#   (switch to 72B on the PRO 6000 Blackwell for higher-quality rewrites)
# ---------------------------------------------------------------------------
BASE_MODEL_LOCAL_PATH = os.environ.get(
    "BASE_MODEL_PATH", "/models/Qwen-Image-Edit-2511"
)
NSFW_WEIGHTS_LOCAL_PATH = os.environ.get(
    "NSFW_WEIGHTS_PATH", "/models/rapid-aio/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"
)
REWRITER_MODEL_LOCAL_PATH = os.environ.get(
    "REWRITER_MODEL_PATH", "/models/Qwen2.5-VL-7B-Instruct"
)

SYSTEM_PROMPT = '''
# Edit Instruction Rewriter
You are a professional edit instruction rewriter. Your task is to generate a precise, concise, and visually achievable professional-level edit instruction based on the user-provided instruction and the image to be edited.  

Please strictly follow the rewriting rules below:

## 1. General Principles
- Keep the rewritten prompt **concise and comprehensive**. Avoid overly long sentences and unnecessary descriptive language.  
- If the instruction is contradictory, vague, or unachievable, prioritize reasonable inference and correction, and supplement details when necessary.  
- Keep the main part of the original instruction unchanged, only enhancing its clarity, rationality, and visual feasibility.  
- All added objects or modifications must align with the logic and style of the scene in the input images.  
- If multiple sub-images are to be generated, describe the content of each sub-image individually.  

## 2. Task-Type Handling Rules

### 1. Add, Delete, Replace Tasks
- If the instruction is clear (already includes task type, target entity, position, quantity, attributes), preserve the original intent and only refine the grammar.  
- If the description is vague, supplement with minimal but sufficient details (category, color, size, orientation, position, etc.). For example:  
    > Original: "Add an animal"  
    > Rewritten: "Add a light-gray cat in the bottom-right corner, sitting and facing the camera"  
- Remove meaningless instructions: e.g., "Add 0 objects" should be ignored or flagged as invalid.  
- For replacement tasks, specify "Replace Y with X" and briefly describe the key visual features of X.  

### 2. Text Editing Tasks
- All text content must be enclosed in English double quotes `" "`. Keep the original language of the text, and keep the capitalization.  
- Both adding new text and replacing existing text are text replacement tasks, For example:  
    - Replace "xx" to "yy"  
    - Replace the mask / bounding box to "yy"  
    - Replace the visual object to "yy"  
- Specify text position, color, and layout only if user has required.  
- If font is specified, keep the original language of the font.  

### 3. Human Editing Tasks
- Make the smallest changes to the given user's prompt.  
- If changes to background, action, expression, camera shot, or ambient lighting are required, please list each modification individually.
- **Edits to makeup or facial features / expression must be subtle, not exaggerated, and must preserve the subject's identity consistency.**
    > Original: "Add eyebrows to the face"  
    > Rewritten: "Slightly thicken the person's eyebrows with little change, look natural."

### 4. Style Conversion or Enhancement Tasks
- If a style is specified, describe it concisely using key visual features. For example:  
    > Original: "Disco style"  
    > Rewritten: "1970s disco style: flashing lights, disco ball, mirrored walls, vibrant colors"  
- For style reference, analyze the original image and extract key characteristics (color, composition, texture, lighting, artistic style, etc.), integrating them into the instruction.  
- **Colorization tasks (including old photo restoration) must use the fixed template:**  
  "Restore and colorize the old photo."  
- Clearly specify the object to be modified. For example:  
    > Original: Modify the subject in Picture 1 to match the style of Picture 2.  
    > Rewritten: Change the girl in Picture 1 to the ink-wash style of Picture 2 — rendered in black-and-white watercolor with soft color transitions.

### 5. Material Replacement
- Clearly specify the object and the material. For example: "Change the material of the apple to papercut style."
- For text material replacement, use the fixed template:
    "Change the material of text "xxxx" to laser style"

### 6. Logo/Pattern Editing
- Material replacement should preserve the original shape and structure as much as possible. For example:
   > Original: "Convert to sapphire material"  
   > Rewritten: "Convert the main subject in the image to sapphire material, preserving similar shape and structure"
- When migrating logos/patterns to new scenes, ensure shape and structure consistency. For example:
   > Original: "Migrate the logo in the image to a new scene"  
   > Rewritten: "Migrate the logo in the image to a new scene, preserving similar shape and structure"

### 7. Multi-Image Tasks
- Rewritten prompts must clearly point out which image's element is being modified. For example:  
    > Original: "Replace the subject of picture 1 with the subject of picture 2"  
    > Rewritten: "Replace the girl of picture 1 with the boy of picture 2, keeping picture 2's background unchanged"  
- For stylization tasks, describe the reference image's style in the rewritten prompt, while preserving the visual content of the source image.  

## 3. Rationale and Logic Check
- Resolve contradictory instructions: e.g., "Remove all trees but keep all trees" requires logical correction.
- Supplement missing critical information: e.g., if position is unspecified, choose a reasonable area based on composition (near subject, blank space, center/edge, etc.).

# Output Format Example
```json
{
   "Rewritten": "..."
}
'''

# ---------------------------------------------------------------------------
# CHANGE 1 of 4: LOCAL PROMPT REWRITER
# Original function used InferenceClient(provider="nebius") -> HuggingFace cloud
# -> Qwen2.5-VL-72B-Instruct running on Nebius servers.
# Replacement: identical function name/signature, runs the same model family
# locally on the VPS GPU. Zero bytes leave this machine.
# ---------------------------------------------------------------------------

# Lazy-loaded on first rewrite call so the main diffusion pipeline
# can claim GPU memory at startup without competition.
_rewriter_model = None
_rewriter_processor = None


def _load_rewriter():
    """Load the local Qwen2.5-VL rewriter model once, on first use."""
    global _rewriter_model, _rewriter_processor
    if _rewriter_model is not None:
        return
    print(f"Loading local rewriter model from: {REWRITER_MODEL_LOCAL_PATH}")
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        _rewriter_processor = AutoProcessor.from_pretrained(
            REWRITER_MODEL_LOCAL_PATH,
            local_files_only=True,
        )
        _rewriter_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            REWRITER_MODEL_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",          # spreads across available GPUs automatically
            local_files_only=True,
        )
        print("Local rewriter model loaded successfully.")
    except Exception as e:
        print(f"WARNING: Could not load local rewriter model: {e}")
        print("Prompt rewriting will fall back to the original prompt.")
        _rewriter_model = None
        _rewriter_processor = None


def polish_prompt_hf(original_prompt, img_list):
    """
    Rewrites the prompt using a LOCAL Qwen2.5-VL model.
    Drop-in replacement for the original InferenceClient-based version.
    Identical call signature, identical return value. No data leaves this machine.
    """
    _load_rewriter()
    if _rewriter_model is None:
        print("Warning: Local rewriter unavailable. Falling back to original prompt.")
        return original_prompt

    prompt = f"{SYSTEM_PROMPT}\n\nUser Input: {original_prompt}\n\nRewritten Prompt:"
    system_prompt = "you are a helpful assistant, you should provide useful answers to users."

    try:
        # Convert list of images to PIL for the processor
        pil_imgs = []
        if img_list is not None:
            if not isinstance(img_list, list):
                img_list = [img_list]
            for img in img_list:
                if hasattr(img, 'save'):           # already a PIL Image
                    pil_imgs.append(img.convert("RGB"))
                elif isinstance(img, str):
                    pil_imgs.append(Image.open(img).convert("RGB"))
                else:
                    print(f"Warning: Unexpected image type: {type(img)}, skipping...")

        # Build the messages in Qwen2.5-VL chat format
        content = [{"type": "text", "text": prompt}]
        for pil_img in pil_imgs:
            content.append({"type": "image", "image": pil_img})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]

        # Apply chat template and tokenise
        text = _rewriter_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _rewriter_processor(
            text=[text],
            images=pil_imgs if pil_imgs else None,
            return_tensors="pt",
        ).to(_rewriter_model.device)

        # Generate — cap at 512 tokens, matching quality of the original cloud call
        with torch.no_grad():
            output_ids = _rewriter_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        # Decode only the newly generated tokens (strip the prompt)
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        result = _rewriter_processor.decode(generated, skip_special_tokens=True)

        # Parse the response — identical JSON extraction logic as original
        if '"Rewritten"' in result:
            try:
                result = result.replace('```json', '').replace('```', '')
                result_json = json.loads(result)
                polished_prompt = result_json.get('Rewritten', result)
            except Exception:
                polished_prompt = result
        else:
            polished_prompt = result

        polished_prompt = polished_prompt.strip().replace("\n", " ")
        return polished_prompt

    except Exception as e:
        print(f"Error during local rewriter inference: {e}")
        return original_prompt


def encode_image(pil_image):
    import io
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Model Loading ---
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"

# Scheduler configuration for Lightning
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

# Initialize scheduler with Lightning config
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

# Load the model pipeline
from safetensors.torch import load_file
import torch.nn.functional as F


#################################

# ---------------------------------------------------------------------------
# CHANGE 2 of 4: BASE PIPELINE — local directory instead of HuggingFace hub
# Original: from_pretrained("Qwen/Qwen-Image-Edit-2511", ...)  <- hits HF servers
# Local:    from_pretrained(BASE_MODEL_LOCAL_PATH, local_files_only=True)
# ---------------------------------------------------------------------------
print("loading base pipeline architecture...")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    BASE_MODEL_LOCAL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda")

# force euler ancestral scheduler
#pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# 2. LOAD RAW WEIGHTS FROM LOCAL FILE
# ------------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# CHANGE 3 of 4: NSFW WEIGHTS — local file instead of hf_hub_download()
# Original: hf_hub_download(repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO", ...) <- HF servers
# Local:    direct path to the pre-downloaded .safetensors file
# ---------------------------------------------------------------------------
print("accessing v23 checkpoint...")
v23_path = NSFW_WEIGHTS_LOCAL_PATH

print(f"loading 28GB state dict into cpu memory...")
state_dict = load_file(v23_path)

# 3. DYNAMIC COMPONENT MAPPING (NO ASSUMPTIONS)
# ------------------------------------------------------------------------------
print("sorting weights into components...")

# containers for the sorted weights
transformer_weights = {}
vae_weights = {}
text_encoder_weights = {}

# analyze the first key to determine the format
first_key = next(iter(state_dict.keys()))
print(f"format detection - first key detected: {first_key}")

# iterate and sort
for k, v in state_dict.items():
    # MAPPING: TRANSFORMER
    # ComfyUI usually prefixes with 'model.diffusion_model.'
    if k.startswith("model.diffusion_model."):
        new_key = k.replace("model.diffusion_model.", "")
        transformer_weights[new_key] = v
    # Or sometimes just 'transformer.' or 'model.'
    elif k.startswith("transformer."):
        new_key = k.replace("transformer.", "")
        transformer_weights[new_key] = v
    
    # MAPPING: VAE
    # ComfyUI prefix: 'first_stage_model.'
    elif k.startswith("first_stage_model."):
        new_key = k.replace("first_stage_model.", "")
        vae_weights[new_key] = v
    # Diffusers prefix: 'vae.'
    elif k.startswith("vae."):
        new_key = k.replace("vae.", "")
        vae_weights[new_key] = v

    # MAPPING: TEXT ENCODER
    # ComfyUI prefix: 'conditioner.embedders.' or 'text_encoder.'
    elif "text_encoder" in k or "conditioner" in k:
        # this is tricky, we try to keep the suffix
        if "conditioner.embedders.0." in k:
            new_key = k.replace("conditioner.embedders.0.", "")
            text_encoder_weights[new_key] = v
        elif "text_encoder." in k:
            new_key = k.replace("text_encoder.", "")
            text_encoder_weights[new_key] = v

# 4. INJECT WEIGHTS (COMPONENT LEVEL)
# ------------------------------------------------------------------------------
print(f"injection statistics:")
print(f" - transformer keys found: {len(transformer_weights)}")
print(f" - vae keys found: {len(vae_weights)}")
print(f" - text encoder keys found: {len(text_encoder_weights)}")

if len(transformer_weights) > 0:
    print("injecting transformer weights...")
    msg = pipe.transformer.load_state_dict(transformer_weights, strict=False)
    print(f"transformer missing keys: {len(msg.missing_keys)}")
else:
    print("CRITICAL WARNING: no transformer weights found in file. check mapping logic.")

if len(vae_weights) > 0:
    print("injecting vae weights...")
    pipe.vae.load_state_dict(vae_weights, strict=False)

if len(text_encoder_weights) > 0:
    print("injecting text encoder weights...")
    # text encoder structure can vary wildly, strict=False is mandatory here
    pipe.text_encoder.load_state_dict(text_encoder_weights, strict=False)

# 5. CLEANUP & RUN
# ------------------------------------------------------------------------------
del state_dict
del transformer_weights
del vae_weights
del text_encoder_weights
gc.collect()
torch.cuda.empty_cache()


#################################


# # --- 1. setup pipeline with lightning (this works fine) ---
# pipe = QwenImageEditPlusPipeline.from_single_file(
#     "path/to/Qwen-Rapid-AIO-NSFW-v21.safetensors",
#     original_config="Qwen/Qwen-Image-Edit-2511", # pulls the config from the base repo
#     scheduler=scheduler,
#     torch_dtype=torch.bfloat16 # use bf16 for speed on zerogpu
# ).to("cuda")

# print("loading lightning lora...")
# pipe.load_lora_weights(
#     "lightx2v/Qwen-Image-Edit-2511-Lightning", 
#     weight_name="Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
# )
# pipe.fuse_lora()
# print("lightning lora fused.")


# # Apply the same optimizations from the first version
# pipe.transformer.__class__ = QwenImageTransformer2DModel
# pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())

# # --- Ahead-of-time compilation ---
# optimize_pipeline_(pipe, image=[Image.new("RGB", (1024, 1024)), Image.new("RGB", (1024, 1024))], prompt="prompt")

# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

def use_output_as_input(output_images):
    """Convert output images to input format for the gallery"""
    if output_images is None or len(output_images) == 0:
        return []
    return output_images

# --- Main Inference Function (with hardcoded negative prompt) ---
# ---------------------------------------------------------------------------
# CHANGE 4 of 4: @spaces.GPU() decorator REMOVED
# That decorator allocates a GPU from HuggingFace's shared ZeroGPU pool.
# On a local VPS the GPU is always available — the decorator is not needed
# and would crash without the `spaces` library installed.
# The entire function body below is 100% identical to the original.
# ---------------------------------------------------------------------------
def infer(
    images,
    prompt,
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    rewrite_prompt=True,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Run image-editing inference using the Qwen-Image-Edit pipeline.

    Parameters:
        images (list): Input images from the Gradio gallery (PIL or path-based).
        prompt (str): Editing instruction (may be rewritten by LLM if enabled).
        seed (int): Random seed for reproducibility.
        randomize_seed (bool): If True, overrides seed with a random value.
        true_guidance_scale (float): CFG scale used by Qwen-Image.
        num_inference_steps (int): Number of diffusion steps.
        height (int | None): Optional output height override.
        width (int | None): Optional output width override.
        rewrite_prompt (bool): Whether to rewrite the prompt using Qwen-2.5-VL.
        num_images_per_prompt (int): Number of images to generate.
        progress: Gradio progress callback.

    Returns:
        tuple: (generated_images, seed_used, UI_visibility_update)
    """
    
    # Hardcode the negative prompt as requested
    negative_prompt = " "
    
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Set up the generator for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Load input images into PIL Images
    pil_images = []
    if images is not None:
        for item in images:
            try:
                if isinstance(item[0], Image.Image):
                    pil_images.append(item[0].convert("RGB"))
                elif isinstance(item[0], str):
                    pil_images.append(Image.open(item[0]).convert("RGB"))
                elif hasattr(item, "name"):
                    pil_images.append(Image.open(item.name).convert("RGB"))
            except Exception:
                continue

    if height==256 and width==256:
        height, width = None, None
    print(f"Calling pipeline with prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}, Size: {width}x{height}")
    if rewrite_prompt and len(pil_images) > 0:
        prompt = polish_prompt_hf(prompt, pil_images)
        print(f"Rewritten Prompt: {prompt}")
    

    # Generate the image
    image = pipe(
        image=pil_images if len(pil_images) > 0 else None,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
    ).images

    # Return images, seed, and make button visible
    return image, seed, gr.update(visible=True)

# --- Examples and UI Layout ---
examples = []

css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#logo-title {
    text-align: center;
}
#logo-title img {
    width: 400px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML("""
        <div id="logo-title">
            <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/qwen_image_edit_logo.png" alt="Qwen-Image Edit Logo" width="400" style="display: block; margin: 0 auto;">
            <h2 style="font-style: italic;color: #5b47d1;margin-top: -27px !important;margin-left: 96px">[Plus] Fast, 4-steps with LightX2V LoRA</h2>
        </div>
        """)
        gr.Markdown("""
        [Learn more](https://github.com/QwenLM/Qwen-Image) about the Qwen-Image series. 
        This demo uses the new [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) with the [Qwen-Image-Lightning-2511](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) LoRA for accelerated inference.
        Try on [Qwen Chat](https://chat.qwen.ai/), or [download model](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) to run locally with ComfyUI or diffusers.
        """)
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input Images", 
                                          show_label=False, 
                                          type="pil", 
                                          interactive=True)

            with gr.Column():
                result = gr.Gallery(label="Result", show_label=False, type="pil", interactive=False)
                # Add this button right after the result gallery - initially hidden
                use_output_btn = gr.Button("↗️ Use as input", variant="secondary", size="sm", visible=False)

        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    show_label=False,
                    placeholder="describe the edit instruction",
                    container=False,
            )
            run_button = gr.Button("Edit!", variant="primary")

        with gr.Accordion("Advanced Settings", open=False):
            # Negative prompt UI element is removed here

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0
                )

                num_inference_steps = gr.Slider(
                    label="Number of inference steps",
                    minimum=1,
                    maximum=40,
                    step=1,
                    value=4,
                )
                
                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=None,
                )
                
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=2048,
                    step=8,
                    value=None,
                )
                
                
                rewrite_prompt = gr.Checkbox(label="Rewrite prompt", value=True)

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn=infer,
        inputs=[
            input_images,
            prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            height,
            width,
            rewrite_prompt,
        ],
        outputs=[result, seed, use_output_btn],  # Added use_output_btn to outputs
    )

    # Add the new event handler for the "Use Output as Input" button
    use_output_btn.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_images]
    )

if __name__ == "__main__":
    demo.launch(share=True)
