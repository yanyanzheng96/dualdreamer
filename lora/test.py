from diffusers import StableDiffusionPipeline, DDIMScheduler
from pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from pipeline_stable_video_diffusion import tensor2vid
from scheduling_euler_discrete import EulerDiscreteScheduler

from pipeline_loraSDE_inference import loraSDE

import torch
import argparse
import os, sys
#from lora_diffusion import tune_lora_scale, patch_pipe
from lora_diffusion_2d import tune_lora_scale, patch_pipe, monkeypatch_remove_lora, collapse_lora, _find_modules, LoraInjectedLinear, LoraInjectedConv2d

import random
import json
from PIL import Image
import numpy as np

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import load_image, export_to_video

from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

import subprocess

import PIL
#if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }


class StableVideoDiffusion:
    def __init__(
        self,
        device,
        fp16=True,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.guidance_type = [
            'sds',
            'pixel reconstruction',
            'latent reconstruction'
        ][1]

        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.to(device)

        self.pipe = pipe
        self.pipe.scheduler = EulerDiscreteScheduler()

        self.num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps if self.guidance_type == 'sds' else 25
        self.pipe.scheduler.set_timesteps(self.num_train_timesteps,  device=device)  # set sigma for euler discrete scheduling

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = None
        self.image = None
        self.target_cache = None

    @torch.no_grad()
    def get_img_embeds(self, image):
        self.image = Image.fromarray(np.uint8(image*255))

    def encode_image(self, image):
        #print(image.dtype)     
        image = image.to(torch.float16) # yan: the bux has been fixed here
        image = image * 2 -1
        latents = self.pipe._encode_vae_image(image, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=False)
        latents = self.pipe.vae.config.scaling_factor * latents
        return latents
    
    def refine(self,
        pred_rgb,
        steps=25, strength=0.8,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
    ):
        # strength = 0.8
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!

        latents = self.encode_image(pred_rgb_512)
        latents = latents.unsqueeze(0)

        if strength == 0:
            init_step = 0
            latents = torch.randn_like(latents)
        else:
            init_step = int(steps * strength)
            latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents), self.pipe.scheduler.timesteps[init_step:init_step+1])

        target = self.pipe(
            image=self.image,
            height=512,
            width=512,
            latents=latents,
            denoise_beg=init_step,
            denoise_end=steps,
            output_type='frame', 
            num_frames=batch_size,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            num_inference_steps=steps,
            decode_chunk_size=1
        ).frames[0]
        target = (target + 1) * 0.5
        target  = target.permute(1,0,2,3)
        return target
            
        # frames = self.pipe(
        #     image=self.image,
        #     height=512,
        #     width=512,
        #     latents=latents,
        #     denoise_beg=init_step,
        #     denoise_end=steps,
        #     num_frames=batch_size,
        #     min_guidance_scale=min_guidance_scale,
        #     max_guidance_scale=max_guidance_scale,
        #     num_inference_steps=steps,
        #     decode_chunk_size=1
        # ).frames[0]
        # export_to_gif(frames, f"tmp.gif")
        # raise
 
    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
    ):

        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
    
        print(f'SVD frame size {batch_size}')


        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!
        # latents = self.pipe._encode_image(pred_rgb_512, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=True)
        latents = self.encode_image(pred_rgb_512)
        latents = latents.unsqueeze(0)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=self.device)
        # print(t)


        w = (1 - self.alphas[t]).view(1, 1, 1, 1)
        if self.guidance_type == 'sds_visualization':
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                t = self.num_train_timesteps - t.item()
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy, sigma = self.pipe.scheduler.add_noise_info(latents, noise, self.pipe.scheduler.timesteps[t:t+1]) # t=0 noise;t=999 clean
                print('sigma is', sigma)
                noise_pred = self.pipe(
                    image=self.image,
                    # image_embeddings=self.embeddings, 
                    height=512,
                    width=512,
                    latents=latents_noisy,
                    output_type='noise', 
                    denoise_beg=t,
                    denoise_end=t + 1,
                    min_guidance_scale=min_guidance_scale,
                    max_guidance_scale=max_guidance_scale,
                    num_frames=batch_size,
                    num_inference_steps=self.num_train_timesteps
                ).frames[0]
            
            # target default 
            # grad = w * (noise_pred - noise)
            # grad = torch.nan_to_num(grad)
            # target = (latents - 1*grad).detach()
            # print(target[0,0,0,0:10,0])

            # target by cancel prediction 
            #target = latents_noisy - sigma * noise_pred
            manual_latents_noisy = latents + sigma*noise
            #print('manual_latents_noisy',manual_latents_noisy[0,0,0,0,0:10])
            print(torch.norm( latents_noisy - (latents + sigma*noise)  ))
            target = latents + sigma*noise - sigma*noise_pred

            #self.pipe.vae.to(dtype=torch.float16)
            target = target.to(dtype=torch.float16)
            #frames = self.pipe(image=self.image, output_type="latent2visual", num_frames=batch_size, latent2visual=target)
            frames = self.pipe(image=self.image, output_type="latent2visual", num_frames=batch_size, latent2visual=target)

            return frames, latents


        if self.guidance_type == 'sds':
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                t = self.num_train_timesteps - t.item()
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.pipe.scheduler.add_noise(latents, noise, self.pipe.scheduler.timesteps[t:t+1]) # t=0 noise;t=999 clean
                noise_pred = self.pipe(
                    image=self.image,
                    # image_embeddings=self.embeddings, 
                    height=512,
                    width=512,
                    latents=latents_noisy,
                    output_type='noise', 
                    denoise_beg=t,
                    denoise_end=t + 1,
                    min_guidance_scale=min_guidance_scale,
                    max_guidance_scale=max_guidance_scale,
                    num_frames=batch_size,
                    num_inference_steps=self.num_train_timesteps
                ).frames[0]
            
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[1]
            print(loss.item())
            return loss
        
        elif self.guidance_type == 'pixel reconstruction':
            # pixel space reconstruction
            if self.target_cache is None:
                with torch.no_grad():
                    self.target_cache = self.pipe(
                        image=self.image,
                        height=512,
                        width=512,
                        output_type='frame', 
                        num_frames=batch_size,
                        num_inference_steps=self.num_train_timesteps,
                        decode_chunk_size=1
                    ).frames[0]
                    self.target_cache = (self.target_cache + 1) * 0.5
                    self.target_cache  = self.target_cache.permute(1,0,2,3)

            loss = 0.5 * F.mse_loss(pred_rgb_512.float(), self.target_cache.detach().float(), reduction='sum') / latents.shape[1]
            print(loss.item())

            return loss

        elif self.guidance_type == 'latent reconstruction':
            # latent space reconstruction
            if self.target_cache is None:
                with torch.no_grad():
                    self.target_cache = self.pipe(
                        image=self.image,
                        height=512,
                        width=512,
                        output_type='latent', 
                        num_frames=batch_size,
                        num_inference_steps=self.num_train_timesteps,
                    ).frames[0]

            loss = 0.5 * F.mse_loss(latents.float(), self.target_cache.detach().float(), reduction='sum') / latents.shape[1]
            print(loss.item())

            return loss


    def get_visualizations(
        self,
        pred_rgb,
        step_ratio=None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
    ):

        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)
    
        print(f'SVD frame size {batch_size}')


        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!
        # latents = self.pipe._encode_image(pred_rgb_512, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=True)
        latents = self.encode_image(pred_rgb_512)
        latents = latents.unsqueeze(0)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=self.device)
        
        print(f'the converted t is {t}')

        #w = (1 - self.alphas[t]).view(1, 1, 1, 1)


        if True:
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                t = self.num_train_timesteps - t.item()
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.pipe.scheduler.add_noise(latents, noise, self.pipe.scheduler.timesteps[t:t+1]) # t=0 noise;t=999 clean
                frames = self.pipe(
                    image=self.image,
                    # image_embeddings=self.embeddings, 
                    height=512,
                    width=512,
                    latents=latents_noisy,
                    output_type='visual', 
                    denoise_beg=t,
                    denoise_end=None, #denoise_end=t + 1,
                    min_guidance_scale=min_guidance_scale,
                    max_guidance_scale=max_guidance_scale,
                    num_frames=batch_size,
                    num_inference_steps=self.num_train_timesteps
                )
            
            return frames


def main():

    device = "cuda"

    ####### robot style ###########################################################
    lora_path_1 = './cache_dir/dance_robot/gallery_ckpt/final_lora.safetensors'
    alpha_1 = 1
    save_path = './outputs'
    os.makedirs(save_path, exist_ok=True)
    img_root = './outputs/a <dance1> <robot1>_99_mid_0.8.png'
    edit_prompt = "a <dance1> <robot1>"


    seed = 0
    repeat_num = 2
    insert_t = 981
    save_path = save_path
    root_path = img_root


    # load checkpoint
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to(
        "cuda"
    )
    scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    pipe.scheduler = scheduler

    pipe.vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    pipe.image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)    


    #########################################################
    ckpt_path = lora_path_1
    patch_pipe(
    pipe,
    ckpt_path,
    patch_text=True,
    patch_ti=True,
    patch_unet=True,
    )

    tune_lora_scale(pipe.unet, 1)
    tune_lora_scale(pipe.text_encoder, 1)

    # ###############################################################
    
    pipe.to(torch.float32)
    ### initialization for images ###
    recordseed = 9
    torch.manual_seed(recordseed)
    latent_noise_1 = 1*torch.randn((1,4,64,64), device = device) #.to(torch.float16)
    save_name = f'{edit_prompt}_{recordseed}.png'
    #image = pipe(prompt = edit_prompt, latents = latent_noise_1, num_inference_steps=50, guidance_scale=7.5).images[0]
    #image.save( os.path.join( save_path, save_name ) )


    recordseed = 99
    torch.manual_seed(recordseed)
    latent_noise_2 = 1*torch.randn((1,4,64,64), device = device) #.to(torch.float16)
    save_name = f'{edit_prompt}_{recordseed}.png'
    #image = pipe(prompt = edit_prompt, latents = latent_noise_2, num_inference_steps=50, guidance_scale=7.5).images[0]
    #image.save( os.path.join( save_path, save_name ) )


    # sequence_arange = np.arange(0.0, 1.0, 0.1)
    # images = []
    # for s in sequence_arange:
    #     print( np.round_(s, 2) )
    #     s = np.round_(s, 2)
    #     latent_noise_mid = s*latent_noise_1 + (1-s)*latent_noise_2
    #     save_name = f'{edit_prompt}_{recordseed}_mid_{s}.png'
    #     image = pipe(prompt = edit_prompt, latents = latent_noise_mid, num_inference_steps=50, guidance_scale=7.5).images[0]
    #     #image.save( os.path.join( save_path, save_name ) )
    #     images.append(image)


    # read from './output' to save images as list
    # Define a transform to convert images to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts the image to a tensor and scales to [0, 1]
    ])
    # Initialize an empty list to hold the image tensors
    tensor_list = []
    directory = './outputs'

    png_files = [filename for filename in os.listdir(directory) if filename.endswith('.png')]

    sorted_png_files = sorted(png_files)
    print(sorted_png_files)

    # Loop through each file in the directory
    #with autocast():
    for filename in sorted_png_files:
        # Check for PNG images
        if filename.endswith('.png'):
            # Construct the full path to the image file
            file_path = os.path.join(directory, filename)
            # Open the image and convert it to RGB
            image = Image.open(file_path).convert('RGB')
            # Apply the transform to the image
            image_tensor = transform(image).unsqueeze(0)
            # Append the image tensor to the list
            tensor_list.append(image_tensor)

    images = torch.cat(tensor_list, dim=0)
    print(images.shape)
    guidance_svd = StableVideoDiffusion(device, fp16=False)
    guidance_svd.image = Image.open('./outputs/a <dance1> <robot1>_99_mid_0.0.png')
    # scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    # guidance_svd.pipe.scheduler = scheduler


    # # check SDS grad
    # guidance_svd.guidance_type = 'sds'
    # grad = guidance_svd.train_step(images, step_ratio = 0.1)
    # print(grad)


    # check SDS grad
    guidance_svd.guidance_type = 'sds_visualization'
    frames, latents = guidance_svd.train_step(images, step_ratio = 0.8)
    #print(frames[0].shape)
    frames_int8 = []
    for f in range(frames[0].shape[0]):
        frame = frames[0][f]
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
            frames_int8.append(frame)
    export_to_video(frames_int8, "./output_videos/generated.mp4", fps=7)
    print(latents.shape)

    latent = latents[0,:,:,:,:]
    print(latent.shape)
    #loraSDE(latent_input = latent)



    # # get frame denoising visualizations
    # frames = guidance_svd.get_visualizations(images, step_ratio = 0.9)
    # frames_int8 = []
    # for f in range(frames[0].shape[0]):
    #     frame = frames[0][f]
    #     if frame.dtype != np.uint8:
    #         frame = (frame * 255).astype(np.uint8)
    #         frames_int8.append(frame)
    # export_to_video(frames_int8, "./output_videos/generated.mp4", fps=7)


    # generator = torch.manual_seed(42)
    # frames = guidance_svd.pipe(guidance_svd.image, height = 512, width = 512, num_inference_steps=30, decode_chunk_size=8, generator=generator).frames[0]
    # export_to_video(frames, "generated.mp4", fps=7)





    ####################
    ### initialization for images ###










if __name__ == "__main__":
    main()







# # put a sequence of images to latents sequence through a image encoder and put the sequence of latents into pytorch trainable parameters and correpsponding optimizer
# # Encode a sequence of images into latents, make them trainable parameters, and create an optimizer
# def encode_and_optimize(self, image_sequence):
#     # Convert image sequence to tensor if not already
#     if not isinstance(image_sequence, torch.Tensor):
#         image_sequence = torch.tensor(image_sequence, device=self.device, dtype=torch.float32)
    
#     # Normalize and encode images to latent space
#     encoded_latents = []
#     for image in image_sequence:
#         encoded_latent = self.encode_image(image)
#         encoded_latents.append(encoded_latent)

#     # Stack encoded latents into a single tensor
#     latents_tensor = torch.stack(encoded_latents)

#     # Make latents trainable parameters
#     latents_parameter = torch.nn.Parameter(latents_tensor)

#     # Initialize an optimizer for the latents
#     optimizer = torch.optim.Adam([latents_parameter], lr=0.01)

#     return latents_parameter, optimizer



















    # # import torch
    # from torch import nn, optim
    # from torchvision.models import resnet18  # Using ResNet18 as an example encoder

    # # Step 1: Define or load a pre-trained image encoder
    # class ImageEncoder(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         # Load a pre-trained ResNet18 without the final classification layer
    #         self.resnet = resnet18(pretrained=True)
    #         self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the last layer
    #         for param in self.resnet.parameters():
    #             param.requires_grad = False  # Freeze the encoder
            
    #     def forward(self, x):
    #         return self.resnet(x).flatten(1)  # Flatten the output

    # # Initialize the encoder
    # encoder = ImageEncoder()

    # # Step 2: Process a sequence of images to obtain latents
    # # Assuming `image_sequence` is a list of PyTorch tensors representing your images
    # image_sequence = [...]  # Replace with actual data
    # latent_sequence = [encoder(image.unsqueeze(0)) for image in image_sequence]  # Add batch dimension

    # # Step 3: Convert latents to trainable parameters
    # latent_params = [nn.Parameter(latent) for latent in latent_sequence]

    # # Step 4: Create an optimizer for the latents
    # optimizer = optim.Adam(latent_params, lr=0.001)

    # # Now you can use optimizer.step() to update the latents after computing gradients