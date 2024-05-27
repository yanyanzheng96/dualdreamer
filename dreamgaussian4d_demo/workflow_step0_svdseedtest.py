import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif

from PIL import Image
import numpy as np

import cv2
import rembg

import argparse
import os


def resize_image(image_path, new_width=None, new_height=None):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print("Failed to open image:", e)
        return None

    # Get original dimensions
    orig_width, orig_height = image.size

    # Calculate new dimensions
    if new_width is not None:
        # Calculate the new height maintaining the aspect ratio
        aspect_ratio = orig_height / orig_width
        new_height = int(new_width * aspect_ratio)
    elif new_height is not None:
        # Calculate the new width maintaining the aspect ratio
        aspect_ratio = orig_width / orig_height
        new_width = int(new_height * aspect_ratio)
    else:
        print("Either new_width or new_height must be provided.")
        return None


     # Resize the image
    resized_image = image.resize((new_width, new_height))

    return resized_image, new_height, new_width


pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
# pipe.enable_model_cpu_offload()
pipe.to("cuda")




workdir = '../demo_data/objects_set'

#category = 'background' # 'object_0', 'object_1', ...
category = 'object_1' # 'object_0', 'object_1', ...



target_dir = os.path.join( workdir, category, 'image_prompt')

# Open the file in read mode
txt_path = os.path.join( workdir, category, 'random_seed', 'seed.txt')
with open(txt_path, 'r') as file:
    # Read the first line of the file
    data = file.readline()
    # Convert the line to an integer
    seed = int(data.strip())


# Check if the directory exists
if os.path.exists(target_dir):
    # List files in the directory
    files = os.listdir(target_dir)
    # Filter for PNG files
    png_files = [file for file in files if file.endswith('.png')]
    # Check if there is any PNG file
    if png_files:
        # Path to the first PNG file (assuming you want to open the first one found)
        png_path = os.path.join(target_dir, png_files[0])
        # Open the image
        #image = Image.open(png_path)


        generator = torch.manual_seed(seed)
        frames = pipe( Image.open(png_path).convert('RGB') , 512, 1024, generator=generator).frames[0]
        #export_to_gif(frames, './ocean.gif')

        new_height = 512
        resized_image, new_height, new_width = resize_image(png_path, new_height = new_height)


        # Create and save copies of the image
        frames_wide = []
        for i in range( len(frames) ):
            frames_wide.append( frames[i].resize( (new_width, new_height) ) )
            filename = os.path.join( workdir, category, 'images_generate', f'{i:05}.png')               
            frames_wide[i].save(filename)
        export_to_gif(frames_wide, os.path.join( workdir, category, 'visualize.gif' ))







