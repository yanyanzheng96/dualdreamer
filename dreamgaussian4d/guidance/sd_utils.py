from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        vram_O=False,
        sd_version="2.1",
        hf_key=None,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif self.sd_version == "2.1":
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == "2.0":
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == "1.5":
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.dtype = torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = None

    @torch.no_grad()
    def get_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)
        self.embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]
    
    @torch.no_grad()
    def get_text_embeds_batch(self, prompts, negative_prompts, latents):
        pos_embeds = self.encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self.encode_text(negative_prompts)

        batch_embeds = torch.cat([neg_embeds.repeat(latents.shape[0], 1, 1),  \
                                  pos_embeds.repeat(latents.shape[0], 1, 1)])

        self.embeddings = batch_embeds  # [2, 77, 768]


    def encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def refine(self, pred_rgb,
               guidance_scale=100, steps=50, strength=0.8,
        ):

        batch_size = pred_rgb.shape[0]
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        latents = self.encode_imgs(pred_rgb_512.to(self.dtype))
        # latents = torch.randn((1, 4, 64, 64), device=self.device, dtype=self.dtype)

        self.scheduler.set_timesteps(steps)
        init_step = int(steps * strength)
        latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
    
            latent_model_input = torch.cat([latents] * 2)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.embeddings,
            ).sample

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = self.decode_latents(latents) # [1, 3, 512, 512]
        return imgs

    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            noise_pred = self.unet(
                latent_model_input, tt, encoder_hidden_states=self.embeddings.repeat(batch_size, 1, 1)
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_pos - noise_pred_uncond
            )

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # seems important to avoid NaN...
        # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[0]

        return loss

    @torch.no_grad()
    def produce_latents(
        self,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    self.embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                ),
                device=self.device,
            )

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=self.embeddings
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(
        self,
        prompts,
        negative_prompts="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        self.get_text_embeds(prompts, negative_prompts)
        
        # Text embeds -> img latents
        latents = self.produce_latents(
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents)  # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")

        return imgs


    def get_inverse_loop(self, pred_rgb, prompts, negative_prompts, back_ratio_list=[0.5], guidance_scale=7.5, as_latent=False):

        gap = 20
        start = 1
        limit = 999

        num_elements = (limit - start) // gap + 1
        timestep_path = [start + gap * i for i in range(num_elements)]

        back_timeindex_list = []
        for back_ratio in back_ratio_list:
            t = np.round(back_ratio * self.num_train_timesteps)
            closest_index, closest_timestep = min(enumerate(timestep_path), key=lambda x: abs(x[1] - t))
            back_timeindex_list.append(closest_index)

        pred_rgb = pred_rgb.to(self.dtype)
        batch_size = pred_rgb.shape[0]
        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            latents_grad = self.encode_imgs(pred_rgb_256.to(self.dtype))
        

        latents = latents_grad.detach()
        self.get_text_embeds_batch(prompts, negative_prompts, latents)
        dict_latents4predict = {}
        for timeindex in back_timeindex_list:
            dict_latents4predict[timeindex] = None
        dict_latents4predict[1] = latents

        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        with torch.no_grad():
            for i, t in enumerate(timestep_path[0:back_timeindex_list[-1]]):
                print(i,t)
                ####### get inverse coefficient 
                t = torch.tensor(t, device = self.device)

                t_back = t + gap
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_back = self.scheduler.alphas_cumprod[t_back] 
                beta_prod_t = 1 - alpha_prod_t
                epsilon_coefficient = (1 - alpha_prod_t_back)**(0.5) / (alpha_prod_t_back)**(0.5) - (1 - alpha_prod_t)**(0.5) / (alpha_prod_t)**(0.5)

                ######## get inverse direction 
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                t_in = torch.cat([t] * 2)
                x_in = torch.cat([latents] * 2)

                noise_pred = self.unet(
                    x_in, t_in, encoder_hidden_states=self.embeddings,
                ).sample
    

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                ######## get back node 
                V = noise_pred_cond
                latents_t = latents
                X_t = latents_t   # latents at timestep t 
                X_t_scale = X_t / (alpha_prod_t)**(0.5)
                X_t_back_scale = X_t_scale + epsilon_coefficient * V
                X_t_back = X_t_back_scale * (alpha_prod_t_back)**(0.5)
                latents = X_t_back       

                model_output = noise_pred_uncond + 1 * (noise_pred_cond - noise_pred_uncond)
                latents_predict = ( latents - (1 - alpha_prod_t_back)**(0.5) * model_output ) / (alpha_prod_t_back)**(0.5)

                ######## save needed back prediction 
                if t_back in dict_latents4predict.keys():
                    dict_latents4predict[t_back] = latents_predict

                
                ######## visual sanity check 
                # idx = 0
                # if True:
                #     imgs = self.decode_latents(latents)
                #     input_img_torch_resized = imgs[idx,:,:,:].permute(1, 2, 0)
                #     input_img_np = input_img_torch_resized.detach().cpu().numpy()
                #     input_img_np = (input_img_np * 255).astype(np.uint8)
                #     Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversion/checkinverse_{i}_{t}.png')

                # if True:
                #     imgs = self.decode_latents(latents_predict)
                #     input_img_torch_resized = imgs[idx,:,:,:].permute(1, 2, 0)
                #     input_img_np = input_img_torch_resized.detach().cpu().numpy()
                #     input_img_np = (input_img_np * 255).astype(np.uint8)
                #     Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversion/checkpredict_{i}_{t}.png')
                
    
            ######## reverse sanity check 
            latents=latents
            #latents = (alpha_prod_t_back)**(0.5)*dict_latents4predict[1] + (1 - alpha_prod_t_back)**(0.5) * torch.randn_like(model_output).to(self.device)
            if back_ratio_list[0] == 1:
                print('generate from normal')
                latents = torch.randn_like(latents).to(self.device)

            for i, t in enumerate(reversed(timestep_path[0:back_timeindex_list[-1]])):
                t = torch.tensor(t + gap, device = self.device)
                print(t)

                ####### get reverse coefficient 
                t_prev = t - gap
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] 
                beta_prod_t = 1 - alpha_prod_t
                epsilon_coefficient = (1 - alpha_prod_t)**(0.5) / (alpha_prod_t)**(0.5) - (1 - alpha_prod_t_prev)**(0.5) / (alpha_prod_t_prev)**(0.5)


                ######## get reverse direction 
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                t_in = torch.cat([t] * 2)
                x_in = torch.cat([latents] * 2)


                noise_pred = self.unet(
                    x_in, t_in, encoder_hidden_states=self.embeddings,
                ).sample

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                ######## get prev node 
                V = noise_pred
                X_t = latents    
                X_t_scale = X_t / (alpha_prod_t)**(0.5)
                epsilon_coefficient = (1 - alpha_prod_t)**(0.5) / (alpha_prod_t)**(0.5) - (1 - alpha_prod_t_prev)**(0.5) / (alpha_prod_t_prev)**(0.5)
                X_t_prev_scale = X_t_scale - epsilon_coefficient * V
                X_t_prev = X_t_prev_scale * (alpha_prod_t_prev)**(0.5)
                latents = X_t_prev

                model_output = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_predict = ( latents - (1 - alpha_prod_t_prev)**(0.5) * model_output ) / (alpha_prod_t_prev)**(0.5)


    
            imgs = self.decode_latents(latents_predict[:,:,:,:])
            #imgs = self.decode_latents(latents[:,:,:,:])
            imgs_reverse = imgs


            image_inverses = []
            for idx in range(imgs.shape[0]):
                input_img_torch_resized = imgs_reverse[idx,:,:,:].permute(1, 2, 0)
                input_img_np = input_img_torch_resized.detach().cpu().numpy()
                input_img_np = (input_img_np * 255).astype(np.uint8)
                #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_train/checkreverse_view_{idx}.png')                
                image = Image.fromarray((input_img_np).astype(np.uint8))
                image_inverses.append(image)
                # idx = 0
                # if True:
                #     imgs = self.decode_latents(latents)
                #     imgs_reverse = imgs
                #     input_img_torch_resized = imgs[idx,:,:,:].permute(1, 2, 0)
                #     input_img_np = input_img_torch_resized.detach().cpu().numpy()
                #     input_img_np = (input_img_np * 255).astype(np.uint8)
                #     Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_reversion/checkreverse_{i}_{t}.png')
                    

                # if True:
                #     imgs = self.decode_latents(latents_predict)
                #     input_img_torch_resized = imgs[idx,:,:,:].permute(1, 2, 0)
                #     input_img_np = input_img_torch_resized.detach().cpu().numpy()
                #     input_img_np = (input_img_np * 255).astype(np.uint8)
                #     Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_reversion/checkpredict_{i}_{t}.png')

        loss = F.mse_loss( latents_grad , torch.zeros_like(latents_grad ).to(self.device))

        return imgs_reverse, loss, image_inverses  # 16*3*256*256




    # def get_sde_loop_safe(self, pred_rgb, prompts, negative_prompts, back_ratio_list=[0.5], guidance_scale=7.5, as_latent=False):

    #     batch_size = 1
    #     uncond_tokens = [""] * batch_size
    #     uncond_inputs = self.tokenizer(
    #         uncond_tokens,
    #         padding="max_length",
    #         max_length=self.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     )
    #     uncond_input_ids = uncond_inputs.input_ids



    #     pred_rgb_512 = F.interpolate(pred_rgb[:,:,:,:], (512, 512), mode='bilinear', align_corners=False)
    #     latents_target = self.encode_imgs(pred_rgb_512.to(self.dtype))

    #     with torch.no_grad(): 
    #         torch.manual_seed(0)
    #         insert_t = 601
    #         alpha_prod_t = self.scheduler.alphas_cumprod[torch.tensor(insert_t, device = self.device)]
        
    #         randv_param = torch.randn(latents_target.shape[0], 4, 64, 64).to(self.device)
    #         latents = (alpha_prod_t)**(0.5) * (latents_target) + (1 - alpha_prod_t)**(0.5) * ((randv_param))
    #         latents_t = latents.clone().detach()  # latents at timestep t 

    #         # start SDEedit generation 
    #         prompt = prompts
    #         insert_timestep = insert_t
    #         guidance_scale = 7.5


    #         timestep_path = list(range(insert_t, 1, -20))
    #         timestep_path.append(timestep_path[-1])

    #         for i,t in enumerate(timestep_path[:-1]):

    #             #print(latents[0,0,0,0:5])

    #             t = torch.tensor(t, device = self.device)
    #             print(t)

    #             # feed text prompt to text encoder
    #             negative_prompt_embeds = self.text_encoder(
    #                 input_ids = uncond_input_ids.to(self.device),
    #                 attention_mask=None,
    #             )
    #             negative_prompt_embeds = negative_prompt_embeds[0]

    #             prompt = prompt
    #             text_inputs = self.tokenizer(
    #                 prompt ,
    #                 padding="max_length",
    #                 max_length=self.tokenizer.model_max_length,
    #                 truncation=True,
    #                 return_tensors="pt",
    #             )
    #             text_input_ids = text_inputs.input_ids 
    #             guide_prompt_embeds = self.text_encoder(
    #                 input_ids = text_input_ids.to(self.device),
    #                 attention_mask=None,
    #             )
    #             guide_prompt_embeds = guide_prompt_embeds[0]
    #             guide_prompt_embeds = guide_prompt_embeds.to(self.device)
    #             text_prompt_embeds = guide_prompt_embeds

    #             prompt_embeds = torch.cat([negative_prompt_embeds, text_prompt_embeds])

    #             #prompt_embeds = torch.cat([negative_prompt_embeds, text_prompt_embeds])
    #             prompt_embeds = torch.cat([negative_prompt_embeds.repeat(latents_target.shape[0], 1, 1),  \
    #                                             text_prompt_embeds.repeat(latents_target.shape[0], 1, 1)])



    #             latent_model_input = torch.cat([latents] * 2)
    #             ts = torch.cat([t.unsqueeze(0)] * latents_target.shape[0] * 2)

    #             noise_pred = self.unet(
    #                 sample = latent_model_input.to(self.device),
    #                 timestep = ts.to(self.device),
    #                 encoder_hidden_states=prompt_embeds.to(self.device),
    #                 return_dict=False,
    #             )[0]
    #             noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    #             guidance_scale = guidance_scale
                
    #             noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    #             #noise_pred = 1.5* noise_pred_text - 0.5* noise_pred_uncond 
                
    #             #model_output = noise_pred_text 
    #             model_output = noise_pred


    #             X_t = latents    
    #             self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
    #             t_prev = timestep_path[i+1]
    #             alpha_prod_t = self.scheduler.alphas_cumprod[t]
    #             alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else self.scheduler.final_alpha_cumprod
    #             beta_prod_t = 1 - alpha_prod_t
    #             X_t_scale = X_t / (alpha_prod_t)**(0.5)
    #             epsilon_coefficient = (1 - alpha_prod_t)**(0.5) / (alpha_prod_t)**(0.5) - (1 - alpha_prod_t_prev)**(0.5) / (alpha_prod_t_prev)**(0.5)
    #             X_t_prev_scale = X_t_scale - epsilon_coefficient * model_output
    #             X_t_prev = X_t_prev_scale * (alpha_prod_t_prev)**(0.5)
    #             latents = X_t_prev



    #             latents_predict = ( latents - (1 - alpha_prod_t)**(0.5) * model_output ) / (alpha_prod_t)**(0.5)
                

    #         for idx in range(latents_predict.shape[0]):
    #             imgs = self.decode_latents(latents_predict[idx:idx+1,:,:,:])
    #             input_img_torch_resized = imgs[0,:,:,:].permute(1, 2, 0)
    #             input_img_np = input_img_torch_resized.detach().cpu().numpy()
    #             input_img_np = (input_img_np * 255).astype(np.uint8)
    #             Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_train/ablation_view_{idx}.png')                
                   
    #     return 









    def get_sde_loop(self, pred_rgb, prompts, negative_prompts, back_ratio_list=[0.5], index_insert = 20, guidance_scale=7.5, as_latent=False):

        gap = 20
        start = 1
        limit = 999

        num_elements = (limit - start) // gap + 1
        timestep_path = [start + gap * i for i in range(num_elements)]

        back_timeindex_list = []
        for back_ratio in back_ratio_list:
            t = np.round(back_ratio * self.num_train_timesteps)
            closest_index, closest_timestep = min(enumerate(timestep_path), key=lambda x: abs(x[1] - t))
            back_timeindex_list.append(closest_index)


        batch_size = pred_rgb.shape[0]
        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            latents_grad = self.encode_imgs(pred_rgb_256.to(self.dtype))
        
        latents= latents_grad.detach()
        self.get_text_embeds_batch(prompts, negative_prompts, latents)
        dict_latents4predict = {}
        for timeindex in back_timeindex_list:
            dict_latents4predict[timeindex] = None
        dict_latents4predict[1] = latents

 
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        index_insert = index_insert
        t_insert = timestep_path[index_insert]

        t = torch.tensor(t_insert, device = self.device)

        t_back = t + gap
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        alpha_prod_t_back = self.scheduler.alphas_cumprod[t_back] 
        beta_prod_t = 1 - alpha_prod_t
        epsilon_coefficient = (1 - alpha_prod_t_back)**(0.5) / (alpha_prod_t_back)**(0.5) - (1 - alpha_prod_t)**(0.5) / (alpha_prod_t)**(0.5)

        with torch.no_grad():
            latents = (alpha_prod_t_back)**(0.5)*dict_latents4predict[1] + (1 - alpha_prod_t_back)**(0.5) * torch.randn_like(latents).to(self.device)

            for i, t in enumerate(reversed(timestep_path[0:index_insert])):
                t = torch.tensor(t + gap, device = self.device)
                print(t)

                ####### get reverse coefficient 
                t_prev = t - gap
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[t_prev] 
                beta_prod_t = 1 - alpha_prod_t
                epsilon_coefficient = (1 - alpha_prod_t)**(0.5) / (alpha_prod_t)**(0.5) - (1 - alpha_prod_t_prev)**(0.5) / (alpha_prod_t_prev)**(0.5)


                ######## get reverse direction 
                t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
                t_in = torch.cat([t] * 2)
                x_in = torch.cat([latents] * 2)


                noise_pred = self.unet(
                    x_in, t_in, encoder_hidden_states=self.embeddings,
                ).sample

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                ######## get prev node 
                V = noise_pred
                X_t = latents    
                X_t_scale = X_t / (alpha_prod_t)**(0.5)
                epsilon_coefficient = (1 - alpha_prod_t)**(0.5) / (alpha_prod_t)**(0.5) - (1 - alpha_prod_t_prev)**(0.5) / (alpha_prod_t_prev)**(0.5)
                X_t_prev_scale = X_t_scale - epsilon_coefficient * V
                X_t_prev = X_t_prev_scale * (alpha_prod_t_prev)**(0.5)
                latents = X_t_prev

                model_output = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                latents_predict = ( latents - (1 - alpha_prod_t_prev)**(0.5) * model_output ) / (alpha_prod_t_prev)**(0.5)

       
            imgs = self.decode_latents(latents[:,:,:,:])
            imgs_reverse = imgs


            #for idx in range(imgs.shape[0]):
            for idx in range(1):
                input_img_torch_resized = imgs_reverse[idx,:,:,:].permute(1, 2, 0)
                input_img_np = input_img_torch_resized.detach().cpu().numpy()
                input_img_np = (input_img_np * 255).astype(np.uint8)
                #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_train/checksde_view_{idx}.png')                
                image = Image.fromarray((input_img_np).astype(np.uint8))

                # idx = 0
                # if True:
                #     imgs = self.decode_latents(latents)
                #     imgs_reverse = imgs
                #     input_img_torch_resized = imgs[idx,:,:,:].permute(1, 2, 0)
                #     input_img_np = input_img_torch_resized.detach().cpu().numpy()
                #     input_img_np = (input_img_np * 255).astype(np.uint8)
                #     Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_reversion/checkreverse_{i}_{t}.png')
                    

                # if True:
                #     imgs = self.decode_latents(latents_predict)
                #     input_img_torch_resized = imgs[idx,:,:,:].permute(1, 2, 0)
                #     input_img_np = input_img_torch_resized.detach().cpu().numpy()
                #     input_img_np = (input_img_np * 255).astype(np.uint8)
                #     Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_reversion/checkpredict_{i}_{t}.png')

        loss = F.mse_loss( latents_grad  , latents_predict)

        return imgs_reverse, loss, image # 16*3*256*256
















if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument("--fp16", action="store_true", help="use float16 for training")
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
