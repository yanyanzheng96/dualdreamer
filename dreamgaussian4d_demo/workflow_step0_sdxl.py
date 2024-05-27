from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch

import os

import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")



visual_dir = '../demo_data/test_visuals' 
os.makedirs( visual_dir, exist_ok = True )
for i in range(10):
    prompt = "top view of zoom in cloud"
    torch.manual_seed(i)

    image = pipe(prompt).images[0]
    image.save( os.path.join(visual_dir, f'./sdxl_prompt_{prompt}_{i}.png') )