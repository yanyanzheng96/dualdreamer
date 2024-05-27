import torch

from inversion_videopipeline_yan_ import StableVideoDiffusionPipeline
from inversion_videoscheduler_yan_ import EulerDiscreteScheduler

from diffusers.utils import load_image  # export_to_video, export_to_gif

import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np

import cv2
import rembg

import argparse
import os


def export_to_gif(image_list, gif_path):
    # Save the images as a GIF
    image_list[0].save(
        gif_path,
        save_all=True,
        append_images=image_list[1:],  # Appending all images after the first
        duration=200,  # Duration between frames in milliseconds
        loop=0  # Loop count, 0 for infinite loop
    )



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
pipe.scheduler = EulerDiscreteScheduler.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="scheduler", torch_dtype=torch.float16, variant="fp16"
        )

print(f"[INFO] loading SVD...")
from guidance.svd_utils_ import StableVideoDiffusion
guidance_svd = StableVideoDiffusion("cuda")
print(f"[INFO] loaded SVD!")




##################### test inversion ####################################


image_path = './desert.jpeg'
save_path = './test_bginversion/desert'
os.makedirs( save_path, exist_ok=True )
new_height = 512
#width = 512
seed = 100
generator = torch.manual_seed(seed)
num_frames = 14



# ##################### inversion 0 ####################################
###Ensure the target directory exists
target_dir = os.path.join(save_path, 'iteration_0', 'generate_frames')
os.makedirs(target_dir, exist_ok=True)


resized_image, new_height, new_width = resize_image(image_path, new_height = new_height)
# frames = pipe( Image.open(image_path) , generator=generator).frames[0]


# # Create and save copies of the image
# frames_wide = []
# for i in range(num_frames):
#     frames_wide.append( frames[i].resize( (new_width, new_height) ) )
#     filename = os.path.join(target_dir, f'{i:05}.png')
#     frames_wide[i].save(filename)
# export_to_gif(frames_wide, os.path.join( save_path, 'iteration_0', 'iteration_0.gif' ))


# ##################### inversion 1 ####################################
# # Ensure the target directory exists
# target_dir = os.path.join(save_path, 'iteration_1', 'generate_frames')
# os.makedirs(target_dir, exist_ok=True)
# ### define crop size
# left = 0
# top = 0
# right = new_height * 2  # Double the new height for the width
# bottom = new_height

# frames = pipe(resized_image.crop((left, top, right, bottom)), new_height, new_height*2, generator=generator).frames[0]
# export_to_gif(frames, './test_visual/gif_.gif')

# frames_paste = [] 
# for i, (wide_image, frame) in enumerate(zip(frames_wide, frames)):
#     # Paste the smaller image into the larger image at the desired position
#     # This replaces the left part (0 to 1024 in width) of the wide_image
#     wide_image.paste(frame, (0, 0))  # Pasting at position (0,0)

#     # Save the modified wide image
#     wide_image.save(  os.path.join(target_dir, f'{i:05}.png') )
#     frames_paste.append(wide_image)

# export_to_gif(frames_paste, os.path.join( save_path, 'iteration_1', 'iteration_1.gif' ))




##################### inversion i ... #################################
top = 0
bottom = new_height
wid = 512
gap = 100 #
count_iter = 0
count_width = new_height*2
if count_width < new_width:

    count_width = count_width + gap 
    ### define crop size
    left = count_width - wid

    right = count_width  # Double the new height for the width
    top = top
    bottom = bottom

    imagetensors = [] # each element should be 3*512*1024 tensor
    frames_wide = []
    for i in range(num_frames):
        wide_image = Image.open( os.path.join(save_path, f'iteration_{count_iter}', 'generate_frames', f'{i:05}.png') ) 
        frames_wide.append(wide_image)

        imagetensor = wide_image.crop( (left, top, right, bottom) )
        # # Resize the image to the desired size (512x512)
        # imagetensor = imagetensor.resize((512, 512))
        # Convert the PIL image to a PyTorch tensor
        imagetensor = TF.to_tensor(imagetensor)
        # Ensure the tensor is in the shape [3, 512, 512]
        imagetensor = imagetensor[:3, :, :].to("cuda")
        imagetensors.append( imagetensor.unsqueeze(0) )
    
    pred_rgb = torch.cat( imagetensors, dim = 0 )


    guidance_svd.get_img_embeds(  Image.open( os.path.join(save_path, f'prompt.jpeg') )  )

    inverse_steps = 10
    frames = guidance_svd.get_inverse_loop(pred_rgb = pred_rgb, inverse_steps = inverse_steps)
    frames_uint8 = (frames * 255).astype(np.uint8)
    frames = [Image.fromarray(frame) for frame in frames_uint8]


    count_iter = count_iter + 1
    target_dir = os.path.join(save_path, f'iteration_{count_iter}', 'generate_frames')
    os.makedirs(target_dir, exist_ok=True)
    frames_paste = [] 
    for i, (wide_image, frame) in enumerate(zip(frames_wide, frames)):
        # Paste the smaller image into the larger image at the desired position
        # This replaces the left part (0 to 1024 in width) of the wide_image
        wide_image.paste(frame, (count_width - wid, 0))  # Pasting at position (0,0)

        # Save the modified wide image
        wide_image.save(  os.path.join(target_dir, f'{i:05}.png') )
        frames_paste.append(wide_image)

    export_to_gif(frames_paste, os.path.join( save_path, f'iteration_{count_iter}', f'iteration_{count_iter}.gif' ))

    breakpoint()






    export_to_gif(pngs, f"./test_bginversion/inversion_depth_inverse_{inverse_steps}.gif")

    breakpoint()





# frames = pipe(image_bg_resize, image_bg_resize.size[1], image_bg_resize.size[0], generator=generator).frames[0]
# export_to_gif(frames, f"./{dir_name}/{bg_name}.gif")

# # Construct the folder path
# folder_path = f"./{dir_name}/{bg_name}_images/images"
# # Ensure the directory exists
# os.makedirs(folder_path, exist_ok=True)
# # Iterate over each frame and save it
# for i, frame in enumerate(frames):
#     # Construct the file path
#     file_path = os.path.join(folder_path, f"output_{i:04d}.png")
#     # Save the image
#     frame.save(file_path)
# print(f"Saved {len(frames)} images to '{folder_path}'.")




#######################################################################################################
imagetensors = [] # each element should be 3*512*512 tensor
for i in range(14):
    #imagetensor = Image.open(f'./test_video_inversion/fish_ocean-42_images/images/output_{i:04}.png')
    imagetensor = Image.open(f'./test_video_inversion/generate_frames/{i:05}.png')

    # Resize the image to the desired size (512x512)
    imagetensor = imagetensor.resize((512, 512))
    # Convert the PIL image to a PyTorch tensor
    imagetensor = TF.to_tensor(imagetensor)
    # Ensure the tensor is in the shape [3, 512, 512]
    imagetensor = imagetensor[:3, :, :].to("cuda")
    imagetensors.append( imagetensor.unsqueeze(0) )
pred_rgb = torch.cat( imagetensors, dim = 0 )


# imagetensors = [] # each element should be 3*512*512 tensor
# for i, frame in enumerate(frames):
#     imagetensor = frame
#     # Resize the image to the desired size (512x512)
#     imagetensor = imagetensor.resize((512, 512))
#     # Convert the PIL image to a PyTorch tensor
#     imagetensor = TF.to_tensor(imagetensor)
#     # Ensure the tensor is in the shape [3, 512, 512]
#     imagetensor = imagetensor[:3, :, :].to("cuda")
#     imagetensors.append( imagetensor.unsqueeze(0) )
# pred_rgb = torch.cat( imagetensors, dim = 0 )


guidance_svd.get_img_embeds(  Image.open(f'./test_video_inversion/generate_frames/{0:05}.png') )

inverse_steps = 12
frames = guidance_svd.get_inverse_loop(pred_rgb = pred_rgb, inverse_steps = inverse_steps)
frames_uint8 = (frames * 255).astype(np.uint8)
pngs = [Image.fromarray(frame) for frame in frames_uint8]

export_to_gif(pngs, f"./test_video_inversion/inversion_depth_inverse_{inverse_steps}.gif")







