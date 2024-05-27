import os
import cv2
import time
import tqdm
import numpy as np

import torch
import torch.nn.functional as F

import re

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize
import torchvision.transforms.functional as TF

from PIL import Image 

# Load the background image
background = Image.open('paper_demo/background.png')
# Load the cutout image with alpha mask
cutout = Image.open('paper_demo/fishcut.png')
# Ensure the cutout image has an alpha channel (transparency)
cutout = cutout.convert("RGBA")

# Define the scale factor
scale_factor = 2  # Adjust this value as needed

# Resize the cutout image
cutout_width, cutout_height = cutout.size
new_size = (int(cutout_width * scale_factor), int(cutout_height * scale_factor))
cutout_resized = cutout.resize(new_size)

# Calculate the position to center the resized cutout image on the background
bg_width, bg_height = background.size
resized_width, resized_height = cutout_resized.size
# Calculate the position
position = ((bg_width - resized_width) // 2, (bg_height - resized_height) // 2)
# Paste the resized cutout image onto the background image at the calculated position
background.paste(cutout_resized, position, cutout_resized)
# Save the resulting image
background.save('paper_demo/combined_image_centered.png')


# new_height = 512
# bg_image_resized = background
# # Convert the PIL image to a PyTorch tensor
# bg_tensor = TF.to_tensor(bg_image_resized)
# # Ensure the tensor is in the shape [3, 512, 512]
# bg_tensor = bg_tensor[:3, :, :].to("cuda")






# from PIL import Image
# def image_pt2pil(img): # input image tensor shoud be of shape 3,512,512
#     input_img_torch_resized = img.permute(1, 2, 0)
#     input_img_np = input_img_torch_resized.detach().cpu().numpy()
#     input_img_np = (input_img_np * 255).astype(np.uint8)
#     return Image.fromarray((input_img_np).astype(np.uint8)) 


# from guidance.sd_utils import StableDiffusion
# guidance_sd = StableDiffusion("cuda")



# # Calculate the total number of samples
# total_samples = render_views_.size(0)

# with torch.no_grad(): 
#     for t in range(total_samples):
#         print("Batch sample:", t)

#         if t <= 3:
#             backratio = 0.6 - radii[_]/10
#             imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(render_views_[t:t+1], prompts = self.opt.prompt, negative_prompts = '', back_ratio_list = [backratio])

#             image_inverses = image_inverses + image_inverses_batch
#             imgs_reverse.append(imgs_reverse_batch)

#         if t > 3:
#             image_inverses.append( image_pt2pil(render_views_[t,:,:,:])  )
#             imgs_reverse.append( render_views_[t:t+1,:,:,:] )

#     imgs_reverse = torch.cat(imgs_reverse, dim=0)


