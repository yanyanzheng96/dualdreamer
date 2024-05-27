import torch

from inversion_videopipeline_yan import StableVideoDiffusionPipeline
from inversion_videoscheduler_yan import EulerDiscreteScheduler


from diffusers.utils import load_image, export_to_video, export_to_gif

import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

import cv2
import rembg

import argparse
import os

name = 'cat_astronaut_13'
bg_name = 'cat_astronaut_13'
dir_name = 'test_video_inversion'
height = 512
width = 512
resize = True
seed = 20
generator = torch.manual_seed(seed)

print('seed', seed)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
)
# pipe.enable_model_cpu_offload()
pipe.to("cuda")
pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="scheduler", torch_dtype=torch.float16, variant="fp16"
        )

try:
    image_bg = Image.open( f'./{dir_name}/{bg_name}.png' )
except:
    image_bg = Image.open( f'./{dir_name}/{bg_name}.jpeg' )

if resize:
    image_bg_resize = image_bg.resize((width, height))
    image_bg_resize = image_bg_resize.convert("RGB")


else:
    image_bg_resize = image_bg.convert("RGB")


bg_name = bg_name + '-' + str(seed)

frames = pipe(image_bg_resize, image_bg_resize.size[1], image_bg_resize.size[0], generator=generator).frames[0]
export_to_gif(frames, f"./{dir_name}/{bg_name}.gif")


# Construct the folder path
folder_path = f"./{dir_name}/{bg_name}_images/images"
# Ensure the directory exists
os.makedirs(folder_path, exist_ok=True)
# Iterate over each frame and save it
for i, frame in enumerate(frames):
    # Construct the file path
    file_path = os.path.join(folder_path, f"output_{i:04d}.png")
    # Save the image
    frame.save(file_path)
print(f"Saved {len(frames)} images to '{folder_path}'.")



imagetensors = [] # each element should be 3*512*512 tensor
for i, frame in enumerate(frames):
    imagetensor = frame

    # Resize the image to the desired size (512x512)
    imagetensor = imagetensor.resize((512, 512))
    # Convert the PIL image to a PyTorch tensor
    imagetensor = TF.to_tensor(imagetensor)
    # Ensure the tensor is in the shape [3, 512, 512]
    imagetensor = imagetensor[:3, :, :].to("cuda")
    imagetensors.append( imagetensor.unsqueeze(0) )
pred_rgb = torch.cat(imagetensors, dim=0)
#torch.save(pred_rgb, 'pred_rgb.pt')
a = pred_rgb

##################### test inversion ####################################
print(f"[INFO] loading SVD...")
from guidance.svd_utils import StableVideoDiffusion
guidance_svd = StableVideoDiffusion("cuda")
print(f"[INFO] loaded SVD!")


b = torch.load('pred_rgb.pt')

# Extract the left half of each image in `a`
left_halves = a[:, :, :, :256]  # This takes all rows up to the middle (256 out of 512)
# Extract the right half of each image in `b`
right_halves = b[:, :, :, 256:]  

pred_rgb = torch.cat((left_halves, right_halves), dim=3)


guidance_svd.get_img_embeds( Image.open( f'./test_video_inversion/{name}.png' ) )

inverse_steps = 17
frames = guidance_svd.get_inverse_loop(pred_rgb = pred_rgb, inverse_steps=inverse_steps)
frames = (255 * np.clip(frames, 0, 1)).astype(np.uint8)
images = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]



export_to_gif(images, f"./test_video_inversion/inversion_{inverse_steps}.gif")











