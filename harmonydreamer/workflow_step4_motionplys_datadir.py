import os
import cv2
import time
import tqdm
import numpy as np

import torch
import torch.nn.functional as F

import rembg
import re

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer_4d_step2 import Renderer, MiniCam
from gs_renderer_3d_clean import Renderer_3d, MiniCam_3d

from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

import copy

from PIL import Image
from diffusers.utils import export_to_video, export_to_gif



from PIL import Image
def image_pt2pil(img): # input image tensor shoud be of shape 3,512,512
    input_img_torch_resized = img.permute(1, 2, 0)
    input_img_np = input_img_torch_resized.detach().cpu().numpy()
    input_img_np = (input_img_np * 255).astype(np.uint8)
    return Image.fromarray((input_img_np).astype(np.uint8)) 

def resize_image(image_path, new_width=None, new_height=None):
    try:
        image = Image.open(image_path)
    except Exception as e:
        print("Failed to open image:", e)
        return None

    # Get original dimensions
    image = image.convert("RGB")
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





class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        # self.seed = "random"
        self.seed = 888

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None
        self.guidance_svd = None


        self.enable_sd = False
        self.enable_zero123 = False
        self.enable_svd = False




        self.gaussain_scale_factor = 1
        self.back_ratio_list = None
        self.back_gt_list = None
        self.back_embeddings_list = None


        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        self.input_img_list = None
        self.input_mask_list = None
        self.input_img_torch_list = None
        self.input_mask_torch_list = None

        self.segments_views = None

        self.dict_for_image_list = None 
        self.dict_for_mask_list = None
        self.casenames = None

        # input text
        self.prompt = ""
        self.negative_prompt = ""
        self.lora_params_initial_values = None

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.start_vsd = opt.start_vsd
        
        # # load input data from cmdline
        # if self.opt.input is not None: # True
        #     #self.load_input(self.opt.input) # load imgs, if has bg, then rm bg; or just load imgs
        #     self.load_dir(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None: # None
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None: # None
            self.nagative_prompt = self.opt.negative_prompt



        # renderer
        self.renderer_3d = Renderer_3d(sh_degree=self.opt.sh_degree)

        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.renderer.initialize(self.opt.load_edit)  

        # modify scale 
        xyz = self.renderer.gaussians._xyz
        # Step 1: Compute the centroid
        centroid = xyz.mean(dim=0)
        # Step 2: Translate points to have the centroid at the origin
        translated_xyz = xyz - centroid
        # Step 3: Scale the coordinates
        scale_factor = 1/2.0  # Define your scale factor
        scaled_xyz = translated_xyz * scale_factor
        # Step 4: Optionally translate back to the original centroid
        xyz_ = scaled_xyz + centroid
        # Step 5: feedback to GS xyz
        self.renderer.gaussians._xyz = xyz_
        self.renderer.gaussians.save_ply('./logs/test.ply')
        self.renderer.initialize('./logs/test.ply')


        model_state_path = self.opt.deform_load + "_deformation.pth"  
        print(  model_state_path )
        self.renderer.gaussians._deformation.load_state_dict(torch.load(model_state_path))
        deformation_table_path = self.opt.deform_load + "_deformation_table.pth"    
        self.renderer.gaussians._deformation_table = torch.load(deformation_table_path)
        deformation_accum_path = self.opt.deform_load + "_deformation_accum.pth"
        self.renderer.gaussians_deformation_accum = torch.load(deformation_accum_path)

        self.renderer.save_plys( save_path = self.opt.save_plys )


        # render eval
        image_list =[]
        nframes = 14
        hor = 180
        delta_hor = 360 / nframes
        time = 0
        delta_time = 1
        for _ in range(nframes):
            pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
            cur_cam = MiniCam_3d(
                pose,
                512,
                512,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            self.renderer_3d.initialize( os.path.join( self.opt.save_plys , f'model_{_}.ply') )  
            with torch.no_grad():
                outputs = self.renderer_3d.render(cur_cam)

            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            image_list.append(out)

            time = (time + delta_time) % 14
            hor = (hor+delta_hor) % 360

        export_to_gif(image_list, os.path.join( self.opt.save_plys , f'edit_rotating4d_plys.gif'), )


        self.seed_everything()


    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        print(f'Seed: {seed:d}')
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed




import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="path to the yaml config file")
args, extras = parser.parse_known_args()

args.config = './configs/4d.yaml'

# override default config from cli
opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))


# auto find mesh from stage 1
#opt.load = os.path.join(opt.outdir, opt.save_path + '_model.ply')

# opt.load = os.path.join(opt.outdir, 'flower' + '_model.ply')w
# opt.savename = './visuals/flower_offset_balance'
# opt.input = './data/set/flower'
# opt.save_path = 'flower'

opt.debug = False
opt.data_dir ='/mnt/vita-nas/zy3724/4Dprojects'



opt.load = os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'ply_3d', 'model.ply' )
opt.load_edit = os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'logs_ply', opt.ply_name )

opt.deform_load = os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'pth_4d', 'model' )

opt.save_plys = os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'plys_4d' )


target_dir = os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'backgrounds', 'prompt')
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

opt.bg_path = png_path
opt.case_dir = os.path.join(  opt.data_dir, 'demo_output' , opt.run_name)


#opt.load = None
opt.savename = None
opt.input = None
opt.save_path = None
opt.batch_size = 24
opt.prompt = 'robot'
opt.negative_prompt = 'negative'
opt.start_vsd = 2000



gui = GUI(opt)
