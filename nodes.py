import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import folder_paths
import comfy.model_management as mm

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Add SmolLM3 to model folders
if not "smollm3" in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path("smollm3", os.path.join(folder_paths.models_dir, "smollm3"))


class SmolLM3ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["HuggingFaceTB/SmolLM3-3B-Base", "HuggingFaceTB/SmolLM3-3B"],
                               {"default": "HuggingFaceTB/SmolLM3-3B"}),
                "precision": (["fp16", "fp32", "bf16"], {"default": "fp16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("SMOLLM3_MODEL", "SMOLLM3_TOKENIZER")
    RETURN_NAMES = ("model", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "SmolLM3"

    def load_model(self, model_name, precision, device):
        # Get device
        if device == "cuda":
            device = mm.get_torch_device()

        # Set dtype
        dtype_map = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        dtype = dtype_map[precision]

        log.info(f"Loading SmolLM3 model: {model_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device if device == "cuda" else None
        )

        if device == "cuda":
            model = model.to(device)

        model.eval()

        log.info(f"Model loaded successfully on {device} with {precision} precision")

        return (model, tokenizer)


class SmolLM3Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("SMOLLM3_MODEL",),
                "tokenizer": ("SMOLLM3_TOKENIZER",),
                "prompt": ("STRING",
                           {"default": "Give me a brief explanation of gravity in simple terms.", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 512, "min": 1, "max": 32768, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "use_chat_template": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("generated_text", "full_output")
    FUNCTION = "generate"
    CATEGORY = "SmolLM3"

    def generate(self, model, tokenizer, prompt, max_new_tokens, temperature, top_p, seed, use_chat_template,
                 system_prompt=""):
        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Get device from model
        device = next(model.parameters()).device

        # Prepare input
        if use_chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt

        # Tokenize
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # Generate
        log.info(f"Generating with temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}")

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode output
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Full output including input
        full_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        log.info(f"Generated {len(output_ids)} tokens")

        return (generated_text, full_output)


class SmolLM3SimpleGenerate:
    """Simple generation node without chat template"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("SMOLLM3_MODEL",),
                "tokenizer": ("SMOLLM3_TOKENIZER",),
                "prompt": ("STRING", {"default": "Gravity is", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 128, "min": 1, "max": 2048, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate"
    CATEGORY = "SmolLM3"

    def generate(self, model, tokenizer, prompt, max_new_tokens):
        # Get device from model
        device = next(model.parameters()).device

        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=max_new_tokens)

        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return (generated_text,)


NODE_CLASS_MAPPINGS = {
    "SmolLM3ModelLoader": SmolLM3ModelLoader,
    "SmolLM3Sampler": SmolLM3Sampler,
    "SmolLM3SimpleGenerate": SmolLM3SimpleGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SmolLM3ModelLoader": "SmolLM3 Model Loader",
    "SmolLM3Sampler": "SmolLM3 Sampler (Chat)",
    "SmolLM3SimpleGenerate": "SmolLM3 Simple Generate",
}