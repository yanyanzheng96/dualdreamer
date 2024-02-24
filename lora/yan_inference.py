from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
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

from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms

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
# else:
#     PIL_INTERPOLATION = {
#         "linear": PIL.Image.LINEAR,
#         "bilinear": PIL.Image.BILINEAR,
#         "bicubic": PIL.Image.BICUBIC,
#         "lanczos": PIL.Image.LANCZOS,
#         "nearest": PIL.Image.NEAREST,
#     }





def main():

    ####### robot style ###########################################################
    lora_path_1 = './cache_dir/dance_robot/gallery_ckpt/final_lora.safetensors'
    alpha_1 = 1
    save_path = './output_sde'
    os.makedirs(save_path, exist_ok=True)
    img_root = './output_sde/a blooming flower.png'
    edit_prompt = "white flower"


    seed = 0
    repeat_num = 5
    insert_t = 321
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


    #### patch lora #######################################
    # ckpt_path = lora_path_1
    # patch_pipe(
    # pipe,
    # ckpt_path,
    # patch_text=True,
    # patch_ti=True,
    # patch_unet=True,
    # )

    # tune_lora_scale(pipe.unet, 1)
    # tune_lora_scale(pipe.text_encoder, 1)




    # for _module, name, _child_module in _find_modules(
    #     pipe.unet, search_class=[LoraInjectedLinear, LoraInjectedConv2d]
    # ):
    #     if isinstance(_child_module, LoraInjectedLinear) or isinstance(_child_module, LoraInjectedConv2d):
    #         pass
    #         #print('before, True')

    # collapse_lora(model=pipe.unet, alpha=1.0)
    # collapse_lora(model=pipe.text_encoder, alpha=1.0)

    # monkeypatch_remove_lora(pipe.unet)
    # monkeypatch_remove_lora(pipe.text_encoder)


    # for _module, name, _child_module in _find_modules(
    #     pipe.unet, search_class=[LoraInjectedLinear, LoraInjectedConv2d]
    # ):
    #     if isinstance(_child_module, LoraInjectedLinear) or isinstance(_child_module, LoraInjectedConv2d):
    #         print('after, True')


    #########################################################

    device = "cuda"
    # generator = torch.Generator(device=device )
    # generator.manual_seed(10)

    # recordseed = 9
    # torch.manual_seed(recordseed)
    # latent_noise_1 = 1*torch.randn((1,4,64,64), device = device).to(torch.float16)
    # save_name = f'{edit_prompt}_{recordseed}.png'
    # image = pipe(prompt = edit_prompt, latents = latent_noise_1, num_inference_steps=50, guidance_scale=7.5).images[0]
    # image.save( os.path.join( save_path, save_name ) )


    #### SDEdit #############################################


    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    #print(torch.cuda.memory_summary())



    batch_size = 1
    uncond_tokens = [""] * batch_size
    uncond_inputs = tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input_ids = uncond_inputs.input_ids ##############################################



    def image2latent(imagepath):
        # image process setting and module
        size = 512
        interpolation="bicubic"
        flip_p=1
        center_crop=False

        interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        flip_transform = transforms.RandomHorizontalFlip(flip_p)

        # load image and preprocess 
        image = Image.open(imagepath)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        img = np.array(image).astype(np.uint8)

        if center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((size, size), resample=interpolation)

        #image = flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example = {}
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1).to(device)
        #print(example["pixel_values"].shape) # torch.Size([3, 512, 512])

        example["pixel_values"] = torch.unsqueeze( example["pixel_values"] , 0 ).to(torch.float16)

        latents = pipe.vae.encode(example["pixel_values"]).latent_dist.sample().detach()
        latents = latents * pipe.vae.config.scaling_factor
        return latents




    
    #for ee in prompt_index_list:
    for rr in range(1):
        edit_prompt = edit_prompt
        #prompt = edit_prompt.format(asset_description)
        prompt = edit_prompt
        print('!!!!!!!!!!!', prompt)


        for big_loop in range(repeat_num):

            torch.manual_seed((seed)*10000 + big_loop)

            SDE_insert_timesteps = [insert_t]
            images_path = root_path
            latents_target = image2latent(images_path)

            scheduler = scheduler

            with torch.no_grad(): 

                for insert_t in SDE_insert_timesteps:

                    for seed_s in range(1):
                        alpha_prod_t = scheduler.alphas_cumprod[torch.tensor(insert_t).to(device)]
                    
                        randv_param = torch.randn(1, 4, 64, 64).to(device).to(torch.float16)
                        latents = (alpha_prod_t)**(0.5) * (latents_target) + (1 - alpha_prod_t)**(0.5) * ((randv_param))
                        latents_t = latents.clone().detach()  # latents at timestep t 

                        # start SDEedit generation 
                        prompt = prompt
                        insert_timestep = insert_t
                        guidance_scale = 7.5


                        timestep_path = list(range(insert_t, 1, -20))
                        timestep_path.append(timestep_path[-1])

                        for i,t in enumerate(timestep_path[:-1]):

                            #print(latents[0,0,0,0:5])
                            print(t)

                            t = torch.tensor(t).to(device)

                            # feed text prompt to text encoder
                            negative_prompt_embeds = text_encoder(
                                input_ids = uncond_input_ids.to(device),
                                attention_mask=None,
                            )
                            negative_prompt_embeds = negative_prompt_embeds[0]

                            prompt = prompt
                            text_inputs = tokenizer(
                                prompt ,
                                padding="max_length",
                                max_length=tokenizer.model_max_length,
                                truncation=True,
                                return_tensors="pt",
                            )
                            text_input_ids = text_inputs.input_ids 
                            guide_prompt_embeds = text_encoder(
                                input_ids = text_input_ids.to(device),
                                attention_mask=None,
                            )
                            guide_prompt_embeds = guide_prompt_embeds[0]
                            guide_prompt_embeds = guide_prompt_embeds.to(device)
                            text_prompt_embeds = guide_prompt_embeds

                            prompt_embeds = torch.cat([negative_prompt_embeds, text_prompt_embeds])


                            latent_model_input = torch.cat([latents] * 2)
                            ts = torch.cat([t.unsqueeze(0)] * 2)

                            print(latent_model_input.shape)
                            print(ts.shape)
                            print(prompt_embeds.shape)
                            
                            noise_pred = unet(
                                sample = latent_model_input.to(device),
                                timestep = ts.to(device),
                                encoder_hidden_states=prompt_embeds.to(device),
                                return_dict=False,
                            )[0]
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            guidance_scale = guidance_scale
                            
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                            #noise_pred = 1.5* noise_pred_text - 0.5* noise_pred_uncond 
                            
                            #model_output = noise_pred_text 
                            model_output = noise_pred


                            X_t = latents    
                            scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(device)
                            t_prev = timestep_path[i+1]
                            alpha_prod_t = scheduler.alphas_cumprod[t]
                            alpha_prod_t_prev = scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else scheduler.final_alpha_cumprod
                            beta_prod_t = 1 - alpha_prod_t
                            X_t_scale = X_t / (alpha_prod_t)**(0.5)
                            epsilon_coefficient = (1 - alpha_prod_t)**(0.5) / (alpha_prod_t)**(0.5) - (1 - alpha_prod_t_prev)**(0.5) / (alpha_prod_t_prev)**(0.5)
                            X_t_prev_scale = X_t_scale - epsilon_coefficient * model_output
                            X_t_prev = X_t_prev_scale * (alpha_prod_t_prev)**(0.5)
                            latents = X_t_prev



                            latents_predict = ( latents - (1 - alpha_prod_t)**(0.5) * model_output ) / (alpha_prod_t)**(0.5)
                            image = pipe.vae.decode( latents_predict.clone() / pipe.vae.config.scaling_factor, return_dict=False)[0].detach().cpu()
                            image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])
                            image = image[0] 
                            #display(image)

                        
                        image.save( os.path.join( save_path , f'{edit_prompt}_{big_loop}.png' ) )


































if __name__ == "__main__":
    main()
