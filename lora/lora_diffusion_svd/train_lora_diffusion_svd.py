
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
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

from diffusers.loaders import LoraLoaderMixin
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import json
import math
from itertools import groupby
from typing import Callable, Dict, List, Optional, Set, Tuple, Type, Union

import numpy as np
import PIL
PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }

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
    ################################### load svd pipe #################################################################################
    pipe = StableVideoDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16, variant="fp16")
    pipe.enable_model_cpu_offload()

    ################################### initialize the optimizer ######################################################################
    if True:
        ################################### set the frozen part  ######################################################################
        # We only train the additional adapter LoRA layers
        pipe.vae.requires_grad_(False)
        pipe.image_encoder.requires_grad_(False)
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
        pipe.image_encoder.to(accelerator.device, dtype=weight_dtype)

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






if __name__ == "__main__":
    args = parse_args(input_args=None)
    train(args)