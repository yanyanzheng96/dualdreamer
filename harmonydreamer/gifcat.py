from PIL import Image
import imageio
import os


from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.lora import LoRALinearLayer
from diffusers.utils import load_image, export_to_video, export_to_gif

#from FastSAM.fastsam import FastSAM, FastSAMPrompt

import torch
import torch.nn.functional as F





# Function to extract the i-th frame from a gif
def extract_frame(gif_path, frame_number):
    with Image.open(gif_path) as img:
        img.seek(frame_number)
        return img.copy()


folderpath = '/mnt/vita-nas/zy3724/4Dprojects/demo_logs_4D/exp_2024-05-29 17:56:26_4D'
# List of gif files
gif_files = [ os.path.join(folderpath, f"inversion_0_hor_{i}.gif") for i in range(-45, 45, 3)]

# Frame number to extract
frame_number_to_extract = 13  # For example, the 2nd frame (index 1)

# Extract frames and store them in a list
frames = []
for gif_file in gif_files:
    if os.path.exists(gif_file):
        frame = extract_frame(gif_file, frame_number_to_extract)
        frames.append(frame)

# Save the frames as a new gif
output_gif = "./test_visual/concatenated_frames.gif"
frames[0].save(output_gif, save_all=True, append_images=frames[1:], loop=0, duration=100)
print(f"New GIF saved as {output_gif}")