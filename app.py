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
# LOCAL MODEL PATHS — relative to this script for portability
# Models will auto-download to ./models/ folder on first run
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

BASE_MODEL_LOCAL_PATH = os.path.join(MODELS_DIR, "Qwen-Image-Edit-2511")
NSFW_WEIGHTS_LOCAL_PATH = os.path.join(MODELS_DIR, "rapid-aio", "v23", "Qwen-Rapid-AIO-NSFW-v23.safetensors")


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


# ---------------------------------------------------------------------------
# CHANGE 2 of 4: BASE PIPELINE — auto-download to local models folder
# ---------------------------------------------------------------------------
print("loading base pipeline architecture...")

# Auto-download if not present
if not os.path.exists(BASE_MODEL_LOCAL_PATH):
    print(f"Base model not found. Downloading to {BASE_MODEL_LOCAL_PATH}...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        cache_dir=MODELS_DIR,
    ).to("cuda")
    # Save locally for future use
    pipe.save_pretrained(BASE_MODEL_LOCAL_PATH)
else:
    print(f"Loading from local path: {BASE_MODEL_LOCAL_PATH}")
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
# CHANGE 3 of 4: NSFW WEIGHTS — auto-download to local models folder
# ---------------------------------------------------------------------------
print("accessing v23 checkpoint...")

# Auto-download if not present
if not os.path.exists(NSFW_WEIGHTS_LOCAL_PATH):
    print(f"NSFW weights not found. Downloading to {NSFW_WEIGHTS_LOCAL_PATH}...")
    os.makedirs(os.path.dirname(NSFW_WEIGHTS_LOCAL_PATH), exist_ok=True)
    from huggingface_hub import hf_hub_download
    v23_path = hf_hub_download(
        repo_id="Phr00t/Qwen-Image-Edit-Rapid-AIO",
        filename="v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
        cache_dir=MODELS_DIR,
        local_dir=os.path.join(MODELS_DIR, "rapid-aio"),
        local_dir_use_symlinks=False,
    )
else:
    print(f"Loading from local path: {NSFW_WEIGHTS_LOCAL_PATH}")
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
    negative_prompt=" ",
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Run image-editing inference using the Qwen-Image-Edit pipeline.

    Parameters:
        images (list): Input images from the Gradio gallery (PIL or path-based).
        prompt (str): Editing instruction.
        seed (int): Random seed for reproducibility.
        randomize_seed (bool): If True, overrides seed with a random value.
        true_guidance_scale (float): CFG scale used by Qwen-Image.
        num_inference_steps (int): Number of diffusion steps.
        height (int | None): Optional output height override.
        width (int | None): Optional output width override.
        num_images_per_prompt (int): Number of images to generate.
        progress: Gradio progress callback.

    Returns:
        tuple: (generated_images, seed_used, UI_visibility_update)
    """
    
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
    
    print(f"Prompt: '{prompt}'")
    print(f"Negative Prompt: '{negative_prompt}'")
    print(f"Seed: {seed}, Steps: {num_inference_steps}, Guidance: {true_guidance_scale}, Size: {width}x{height}")
    

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
body, .gradio-container {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
}
#col-container {
    margin: 0 !important;
    max-width: 100% !important;
    padding: 0 !important;
}
.contain {
    padding: 0 !important;
}
#preset-row {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
#preset-row > * {
    flex: 1 !important;
}
#preset-row button {
    flex: 0 0 auto !important;
    min-width: 80px !important;
}
#preset-row input[type="text"] {
    pointer-events: none !important;
    user-select: none !important;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        with gr.Row(elem_id="preset-row"):
            preset_dropdown = gr.Dropdown(
                label="Quick Prompts",
                choices=[
                    "Lift Up Vag Flash",
                    "Same Place Sitting Tease",
                    "Same Place Shh Tease",
                    "Same Place Shh Tease 2",
                    "1 Leg Up Inviting Tease",
                    "Same Place Squatting Frontal",
                    "Dick Holding Pleasure Tease",
                    "Dick Holding Almost BJ",
                    "Same Place Partial BJ",
                    "Side View Oral",
                    "Remove Her Clothes",
                    "Secret Silent Sex",
                    "Rear View Anal",
                    "Spread On Bed Pose",
                    "Come Get It Pose",
                    "Secret Silent Room Sex",
                    "Front View Couch Sex",
                    "Rear View Couch Sex"
                ],
                value=None,
                interactive=True,
                show_label=False,
                allow_custom_value=False
            )
            run_button_top = gr.Button("Edit!", variant="primary", size="sm")
        
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input Images", 
                                          show_label=False, 
                                          type="pil", 
                                          interactive=True)

            with gr.Column():
                result = gr.Gallery(label="Result", show_label=False, type="pil", interactive=False)
                use_output_btn = gr.Button("↗️ Use as input", variant="secondary", size="sm", visible=False)

        with gr.Row():
            prompt = gr.Textbox(
                    label="Prompt",
                    show_label=False,
                    placeholder="describe the edit instruction",
                    container=False,
                    lines=3,
                    max_lines=10,
            )
            run_button = gr.Button("Edit!", variant="primary")

        num_images_per_prompt = gr.Slider(
            label="Number of images",
            minimum=1,
            maximum=4,
            step=1,
            value=1,
        )
        
        negative_prompt = gr.Textbox(
            label="Negative Prompt (what you DON'T want)",
            placeholder="censored, mosaic, blurred, clothed, soft, partial",
            value="",
            lines=2,
            max_lines=5,
        )

        with gr.Accordion("Advanced Settings", open=False):
            # Negative prompt UI element is removed here

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
                visible=False,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True, visible=False)

            with gr.Row():

                true_guidance_scale = gr.Slider(
                    label="True guidance scale",
                    minimum=1.0,
                    maximum=10.0,
                    step=0.1,
                    value=1.0,
                    visible=False,
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

        # gr.Examples(examples=examples, inputs=[prompt], outputs=[result, seed], fn=infer, cache_examples=False)

    # Preset prompt dictionary (in exact order from Image Editor Prompts.txt)
    prompts_dict = {
        "Lift Up Vag Flash": "She lifts the entire bottom portion of her clothing upward to just above her vagina area. All of the bottom fabric is now fully scrunched and bunched tightly in her hands and not visible anywhere else, completely exposing her shaved, beautiful vagina. Any other objects that were in her hands must be removed. Tiny gap of space between her legs. Everything else in the scene remains exactly the same as before.",
        "Same Place Sitting Tease": "She sits on the floor in the same room with the same clothes on but the top of them are pulled down to be under her exposed breasts which must be firm and perky and standing up straight with nipples hard and pointing right at the camera without any  sagging. and the bottom of her clothes are pulled up in a way to not cover her vagina. but with her knees facing straight towards me but her vagina is facing at me, same exact identical face with all of it's exact features and same exact original expression with no changes except her eyes are looking directly at the camera, she uses her hands to grip both sides of her vulva and fully part/spread both sides so that her vagina insides are visible and as far as they be parted  as those same hands also push against her inner thighs visibly, everything else the same",
        "Same Place Shh Tease": "Her left hand's index finger is hovering away from her vagina and she is pointing it directly at her vagina hole while she also holds her right hand's index finger on her mouth saying shh but still smiling extremely sexually suggestively and invitingly, everything else stays the same!",
        "Same Place Shh Tease 2": "Her left hand is being used to touch her vaginal clitoris without covering the whole vagina while she also holds her right hand's index finger on her mouth saying shh but still smiling extremely sexually suggestively and invitingly, vagina fully visible between her now slightly more spread legs. tonge is pressed between her teeth in a sexually suggestive way. Exact same identical face with all the exact same identical features. everything else stays the same!",
        "1 Leg Up Inviting Tease": "She holds 1 leg far up in the air, to fully show vagina and, vagina is facing at me, everything else the same",
        "Same Place Squatting Frontal": "Same exact identical face with all the same exact identical features. \nhead is turned almost fully around naturally to look directly into my eyes. She is being fully penetrated by a small but wide penis coming from the bottom of the picture attached to a man who is no visible who has his legs stretched under her. her mouth wide open with her lips making an o shape. She looks like she is imdeep in love and dreaming in a fantasy. She is very close to the camera sitting on the penis that is now fully inside her vagina hole, showing how it parts her vulva from its penetration.\nremove the objects in the way. She touches her clitoris with just one of her hand's fingers and her other hand stays is grabbing the penis in her vagina.\neverything else is the exact same!",
        "Dick Holding Pleasure Tease": "She is squatting up in the air close to the camera with her legs only partially spread. With her right hand, she is eagerly, firmly, and tightly gripping a small penis that enters the frame from the bottom of the image; the penis belongs to an otherwise completely unseen man whose legs are stretched out flat beneath her. \nShe hovers directly above the tip of the penis. the head of the penis is touching / pressing against her vaginal area. \nShe uses her left hand to reach underneath her clothing as she has her fingers deeply inside her vagina, they disappear inside her.\nHer head is turned almost completely around in a natural way so she can stare directly into the viewer's eyes.  Both of her breasts are fully exposed outside of her clothing, with visibly hardened nipples. her open mouth—lips shaped into a perfect \"O\"—Remove all flowers from the image. Everything else must remain exactly the same.",
        "Dick Holding Almost BJ": "She leans forward with her mouth wide open and lips in an O shape so that her mouth is closer to the penis but not touching it. her tongue is visible only behind the penis pressed against it. She is still firmly grasping the penis the same exact way. Close up of her face. Same exact identical face with all the same features. Everything else is the same",
        "Same Place Partial BJ": "Same environment.\nShe is now on her stomach as far back enough as needed and bending forward enough so that the penis that is coming up from the bottom of the photo that is attached to an unseen man is now all the way in her mouth. put the penis much deeper in her mouth.  You can see her butt visibly behind her head as well which has smooth skin and arches for the cheeks. same exact identical face with all the same exact identical features. Everything else is the exact same.",
        "Side View Oral": "She is wearing the same clothing the same exact way. Her left hand is sexually touching her vagina between her fully spread legs. Her vagina is fully visible and you can see she put fingers inside of it's hole. \nThe same sized but smaller penis from the original photo is now  all the way deep inside of her mouth and you can see that it is in her throat because there is a bulge from it on her throat. The very short penis is coming from the right edge of the photo. Her lips are tightly closed surrounding the bottom part of the penis. She is squatting at a side angle in a way that you can see what I said she is doing with her vagina. The camera angle makes that possible. Her right hand holds the penis and pulls it more into her mouth. Same exact identical face with exact same facial features. her eyes are open and rolled back in pleasure. Remove the man from the photo and keep only the shaft of the penis coming from the right edge of the photo.",
        "Remove Her Clothes": "no clothing, naked, perfectly accurate, everything else is the exact same",
        "Secret Silent Sex": "close up, same exact face, same clothes but breasts fully out with hard nipples and vagina fully showing as the clothes just barely don't cover it. she is leaning very far forwards very close to the camera, full breasts visible and perfect, sitting on a very small penis fully penetrating all the way to the farthest depths of inside her vagina, coming up from an unseen man with his legs stretched underneath her, everything else the same, vulva fully  parted from the penis, she also firmly presses her right hand's index finger up to the center of her lips as if to say shh  but with her mouth wide open, tongue partially showing inside of it with her lips making a small partially closed O shape.\neverything else stays the same!",
        "Rear View Anal": "She is turned around  facing away from the camera, squatting and bent over a chair she holds that is far away from her but in front of her. He clothing is still the same and each part is the same but her breasts are visible outside and the bottom of her clothing is bunched up above her waist. She uses the mans penis that is in her ass to support her. Her feet are touching the ground. her butt is higher in the air so that her ass hole is penetrated by the penis. Same exact identical face as the original woman in the original photo. Her head is turned almost completely around naturally and her eyes are seen looking directly into the camera with her mouth wide open and her lips making an O shape. She is pushing her body backwards into man's penis from the bottom of the photo to get the penis further than it already is inside her ass hole. the ass is higher than the vagina and the vagina cannot be seen. her butt is angled upwards so her vagina is close up facing right at me. Make the ass hole itself higher than normal where her tailbone is instead",
        "Spread On Bed Pose": "no clothes, laying on her girly bed in a just slightly dark room, same exact identical face with all the same exact identical features. her legs are much thinner skinnier and shorter and spread apart to fully show her tiny tight beautiful vagina, identical face and expression, vagina pointed at me. everything else the same!",
        "Come Get It Pose": "no clothes, laying on her girly bed in a slightly dimly lit room, her legs are much thinner skinnier and shorter and spread apart to fully show her tiny tight beautiful vagina, exact same identical face and all the exact same facial features, vagina pointed at me, her left hand's index finger is hovering away from her vagina and she is pointing it directly at her vagina hole while she also holds her right hand's index finger on her mouth saying shh behind a partial kissy face. but still looking right into my eyes  extremely sexually suggestively and excitingly and invitingly, slightly smiling looking excited to finally have me all to herself. close up of vagina. full breasts are visible with hardened nipples. everything else stays the same!",
        "Secret Silent Room Sex": "close up, same exact identical face with the exact same identical features, on a couch, no clothes, she is leaning very far forwards very close to the camera, full breasts visible and perfect, sitting on a very small penis fully penetrating all the way to the farthest depths of inside her vagina, coming up from an unseen man with his legs stretched underneath her, everything else the same, vulva fully  parted from the penis, she also holds her right hand's index finger on her mouth saying shh but with a sexy secretive sneaky look   extremely sexually but distracted by the pleasure, mouth slightly open with lips making an o shape from the immense pleasure, dark bedroom background, everything else stays the same!",
        "Front View Couch Sex": "front view, close up of her face, she is on a very small  penis that cannot be fully seen because it is already fully all the way inside her vagina, coming up from a man that cannot be seen on the bottom of the photo underneath her, same exact face, fully naked, on a couch, full perky breasts visible and perfect,  same exact identical face and person in general but her mouth is open with lips in a closed small surprised o shape with her eyes looking at me but slightly to the top right, dark bedroom background, top part of her sexy cocktail dress is pulled down to be underneath her visible breasts and the bottom part of her dress is pulled upwards still around her legs but not covering her vagina, Exact same identical face with all the exact same identical features. everything else the same.",
        "Rear View Couch Sex": "she is squatting all the way down at the bottom center of the photo with her vagina fully penetrated by a short little penis also coming from all the way at the bottom center of the photo. \nback view, fully naked on a couch with a slight dark bedroom background. the man's legs are seen completely closed together but out straight between her legs. the entire penis is hidden inside of her vagina. she wears a sexy cocktail dress that doesn't cover her vagina. her breasts are also outside of the dress with hardened nipple showing. Her legs are much thinner and petite. both sides of her fat  vulva are noticeably spread apart from the penis and her head is turned almost completely around naturally as far as she can to look directly into my eyes, her mouth is open with lips in a closed small surprised o shape. her butt is pointing up at the camera so you can see the intimate details of her full penetrated vagina. Exact same identical face with all the exact same identical features. she is also reaching behind her back and using only  her little hand to grab and spreading more open her right butt cheek so the penis is seen even deeper inside of her. only the penis of the man is visible. everything else is the exact same!"
    }
    
    # Dropdown handler
    def update_prompt_from_dropdown(choice):
        if choice:
            return prompts_dict.get(choice, "")
        return ""
    
    preset_dropdown.change(
        fn=update_prompt_from_dropdown,
        inputs=[preset_dropdown],
        outputs=[prompt],
        scroll_to_output=False
    )

    gr.on(
        triggers=[run_button.click, run_button_top.click, prompt.submit],
        fn=infer,
        inputs=[
            input_images,
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            true_guidance_scale,
            num_inference_steps,
            height,
            width,
            num_images_per_prompt,
        ],
        outputs=[result, seed, use_output_btn],
    )

    # Add the new event handler for the "Use Output as Input" button
    use_output_btn.click(
        fn=use_output_as_input,
        inputs=[result],
        outputs=[input_images]
    )

    # Preset handler for negative prompts
    def update_negative_prompt(preset):
        presets = {
            "None": "",
            "Full Penetration (Vaginal)": "partial insertion, visible shaft, shallow, tip only, halfway, partially inserted, pulling out, not fully inside",
            "Full Penetration (Oral)": "shallow, tip only, not deepthroat, partially in mouth, lips at tip, halfway, visible shaft outside mouth",
            "Full Penetration (Anal)": "partial insertion, visible shaft, shallow, tip only, halfway, partially inserted, not fully inside, pulling out",
        }
        return presets.get(preset, "")
    
    # REMOVED - negative_preset dropdown no longer exists

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
