
import argparse
import itertools
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTokenizer
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
#from diffusers import StableVideoDiffusionPipeline
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
from diffusers.utils import load_image, export_to_video

from diffusers.loaders import LoraLoaderMixin
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.schedulers import EulerDiscreteScheduler, DDIMScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from pipeline_stable_video_diffusion import tensor2vid

import cv2
import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
from PIL import Image

PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }

from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to a tensor and scales to [0, 1]
])


try:
    from safetensors.torch import safe_open
    from safetensors.torch import save_file as safe_save
    safetensors_available = True
except ImportError:
    from .safe_open import safe_open

    def safe_save(
        tensors: Dict[str, torch.Tensor],
        filename: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None:
        raise EnvironmentError(
            "Saving safetensors requires the safetensors library. Please install with pip or similar."
        )
    safetensors_available = False


from lora_diffusion_svd_simo import (inject_trainable_lora,
                                     inject_trainable_lora_extended
                                    )


################################### load svd pipe #################################################################################
# load checkpoint
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None).to(
    "cuda"
)
scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
pipe.scheduler = scheduler

pipe.vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
pipe.image_processor = VaeImageProcessor(vae_scale_factor=pipe.vae_scale_factor)  


######## utils definition #########################
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






######## get trainable parameters #########################
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dir",
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )

    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def train(args):


    ################################### initialize the optimizer ######################################################################
    if True:
        ################################### set the frozen part  ######################################################################
        # We only train the additional adapter LoRA layers
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.unet.requires_grad_(False)

        ################################### set the accelerator and precision  ########################################################################
        # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.

        accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        
        # Move unet, vae and text_encoder to device and cast to weight_dtype
        # The VAE is in float32 to avoid NaN losses.
        pipe.unet.to(accelerator.device, dtype=weight_dtype)
        if args.pretrained_vae_model_name_or_path is None:
            pipe.vae.to(accelerator.device, dtype=torch.float32)
        else:
            pipe.vae.to(accelerator.device, dtype=weight_dtype)
        pipe.text_encoder.to(accelerator.device, dtype=weight_dtype)

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                pipe.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        ################################### set the lora params from diffusers compatile lora layers ##################################

        # # The text encoder comes from 🤗 transformers, so we cannot directly modify it.
        # # So, instead, we monkey-patch the forward calls of its attention-blocks.
        # if args.train_text_encoder:
        #     # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        #     text_lora_parameters_one = LoraLoaderMixin._modify_text_encoder(
        #         text_encoder_one, dtype=torch.float32, rank=args.rank
        #     )
        #     text_lora_parameters_two = LoraLoaderMixin._modify_text_encoder(
        #         text_encoder_two, dtype=torch.float32, rank=args.rank
        #     )

        ################################### set the lora params of unet by simo injecting ############################################
        # from simo codebase, inject_trainable_lora only monkeypatch linear layer, while inject_trainable_lora_extended also monkeypatch conv layer
        lora_unet_target_modules = {"ResnetBlock2D", "Attention", "GEGLU"}
        ## search_class=[nn.Linear, LoraInjectedLinear, nn.Conv2d, LoraInjectedConv2d]))
        ## if isinstance(_child_module, LoraInjectedLinear)
        unet_lora_params, _ = inject_trainable_lora_extended(
            pipe.unet,    
            r=1,     # r=lora_rank,
            target_replace_module=lora_unet_target_modules,
        )

        #print(len(unet_lora_params))  # 652(linear) loras for inject_trainable_lora and 768(linear and conv2d) for inject_trainable_lora_extended
        unet_lr = 1e-4
        params_to_optimize = [
            {"params": itertools.chain(*unet_lora_params), "lr": unet_lr},
        ]


        weight_decay_lora = 0.001
        lora_optimizers = optim.AdamW(params_to_optimize, weight_decay=weight_decay_lora)

        # breakpoint()
        #pipe.unet.train()
        

        lr_scheduler_lora = "linear"
        lr_warmup_steps_lora = 0
        max_train_steps_tuning = 1000
        lr_scheduler_lora = get_scheduler(
            lr_scheduler_lora,
            optimizer=lora_optimizers,
            num_warmup_steps=lr_warmup_steps_lora,
            num_training_steps=max_train_steps_tuning,
        )


        ### preprocess training data
        image_directory = './training_data/images'
        image_count = sum(1 for filename in os.listdir(image_directory) if filename.endswith('.png'))

        text_prompt = 'a red swimming fish'


        optimizer = lora_optimizers
        num_timesteps = 50
        pipe.scheduler.set_timesteps(num_timesteps,  device=pipe.device)
        height = 512
        width = 512

        for iter in range(10):
            
            print(video_count)
            # load training data
            randint = random.randint(0, video_count-1)
            image_path = f'./training_data/images/image_{randint}.png'

            latents = image2latent(image_path)
            # process training data and build loss




            
        #     image_load = pipe.image_processor.preprocess(image_load, height=height, width=width)
        #     noise = randn_tensor(image_load.shape, device=image_load.device, dtype=image_load.dtype)
        #     noise_aug_strength = 0.02
        #     image_load = image_load + noise_aug_strength * noise
        #     needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
        #     # if needs_upcasting:
        #     #     self.vae.to(dtype=torch.float32)
        #     image_latent = pipe._encode_vae_image(image_load, pipe.device, 1, do_classifier_free_guidance=False)
        #     image_latent = image_latent.to(pipe.dtype) # torch.Size([1, 4, 64, 64])
        #     image_latents = image_latent.unsqueeze(1).repeat(1, num_frames, 1, 1, 1) # torch.Size([1, 14, 4, 64, 64])


        #     added_time_ids = pipe._get_add_time_ids(
        #         fps = 7-1,
        #         motion_bucket_id = 127,
        #         noise_aug_strength = 0.02,
        #         dtype = image_embeddings.dtype ,
        #         batch_size = 1,
        #         num_videos_per_prompt = 1,
        #         do_classifier_free_guidance = False,
        #         )
        #     added_time_ids = added_time_ids.to(pipe.device)


           
        #     r = torch.randint(1, 30, (1,), dtype=torch.long, device=pipe.device)
        #     noise = torch.randn_like(latents).to(pipe.device).to(torch.float16)
        #     target = noise # torch.Size([1, 14, 4, 64, 64])
        #     latents_noisy = pipe.scheduler.add_noise(latents, noise, pipe.scheduler.timesteps[r:r+1]) # torch.Size([1, 14, 4, 64, 64]) # t=0 noise;t=999 clean
            

        #     timesteps = pipe.scheduler.timesteps[r:r+1]
        #     t = timesteps[0]

        #     ##latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        #     latent_model_input = latents
        #     latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        #     # Concatenate image_latents over channels dimention
        #     latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        #     # predict the noise residual
        #     noise_pred = pipe.unet(
        #         latent_model_input,
        #         t,
        #         encoder_hidden_states=image_embeddings,
        #         added_time_ids=added_time_ids,
        #         return_dict=False,
        #     )[0]

        #     print(noise_pred.shape)


        #     loss = F.mse_loss(noise_pred.unsqueeze(0).float(), target.float(), reduction="none").mean()
        #     # loss.requires_grad = True
        #     # breakpoint()

        #     # build loss and backpropagation
        #     lr_scheduler_lora.step()
        #     optimizer.zero_grad()

        #     noise = torch.randn((1,4,64,64))
        #     #loss = torch.mean( pipe.unet())
        #     #loss_sum += loss.detach().item()
        #     print(loss)

        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(
        #         itertools.chain(unet.parameters()), 1.0
        #     )
        #     optimizer.step()



if __name__ == "__main__":
    args = parse_args(input_args=None)
    train(args)