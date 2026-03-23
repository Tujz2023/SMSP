import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import EulerAncestralDiscreteScheduler
from diffusers.utils import load_image
from PIL import ImageEnhance, ImageOps
from semantic_configs import configs
import os

origin_images_dir = "output_images/5.png"
output_path = "output_images/5_semantic.png"
semantic_type = "chinese_architecture"  # Options: chinese_architecture, cyberpunk_city, winter_valley

controlnet_path = "path_to_your_controlnet_model"  # Update this to the actual path of your ControlNet model
controlnet_subfolder = "v2"
base_model_path = "path_to_your_diffusion_model"  # Update this to the actual path of your Stable Diffusion model

def main():

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    controlnet = ControlNetModel.from_pretrained(
        controlnet_path,
        torch_dtype=torch.float16,
        subfolder=controlnet_subfolder
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_xformers_memory_efficient_attention()
    
    condition = load_image(origin_images_dir)

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    condition = condition.convert("RGB")

    for type in configs:

        if type['name'] == semantic_type:

            if type['invert'] == True:
                condition = ImageOps.invert(condition)

            if type['enhance'] != None:
                enhancer = ImageEnhance.Contrast(condition)
                condition = enhancer.enhance(type['enhance'])

            output = pipe(
                prompt=type['prompt'],
                image=condition,
                guidance_scale=type['guidance_scale'],
                num_inference_steps=type['num_inference_steps'],
                control_guidance_end=type['control_guidance_end'],
                height=1000,
                width=1000,
            )

            generated_image = output.images[0]
            generated_image.save(output_path)

if __name__ == "__main__":
    main()