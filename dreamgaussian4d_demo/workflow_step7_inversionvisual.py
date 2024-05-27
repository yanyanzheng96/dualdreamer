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

import torchvision.transforms.functional as TF



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


        # load backgrounds 
        bgpngs = self.opt.bg_dir
        if bgpngs is not None:
            self.bgs = []
            for b_idx in range(self.opt.num_frames):
                #bg_image = Image.open(os.path.join( bgpngs, f'output_{b_idx:04}.png'  ) ).convert("RGB")

                bg_image_path = os.path.join( bgpngs, f'{b_idx:05}.png' )
                new_height = 512
                bg_image_resized, new_height, new_width = resize_image(bg_image_path, new_height = new_height)

                # Convert the PIL image to a PyTorch tensor
                bg_tensor = TF.to_tensor(bg_image_resized)
                # Ensure the tensor is in the shape [3, 512, 512]
                bg_tensor = bg_tensor[:3, :, :].to("cuda")

                print(bg_tensor.shape)

                self.bgs.append( bg_tensor)

        self.new_width = new_width



        self.GSlist = []
        for i in range(self.opt.num_frames):
            renderer_3d = Renderer(sh_degree=self.opt.sh_degree)
            self.GSlist.append(renderer_3d)

            self.GSlist[i].initialize( os.path.join(self.opt.load_plys, f'model_{i}.ply') )






        # renderer
        self.renderer_3d = Renderer_3d(sh_degree=self.opt.sh_degree)

        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.renderer.initialize(self.opt.load_edit)  


        model_state_path = self.opt.deform_load + "_deformation.pth"  
        print(  model_state_path )
        self.renderer.gaussians._deformation.load_state_dict(torch.load(model_state_path))
        deformation_table_path = self.opt.deform_load + "_deformation_table.pth"    
        self.renderer.gaussians._deformation_table = torch.load(deformation_table_path)
        deformation_accum_path = self.opt.deform_load + "_deformation_accum.pth"
        self.renderer.gaussians_deformation_accum = torch.load(deformation_accum_path)

        self.renderer.save_plys( save_path = self.opt.save_plys )


        ############## render input #############################

        nframes = self.opt.num_frames
        hor = -90
        delta_hor = 180 / nframes
        time = 0
        delta_time = 1
        hor_max = 45
        hor_min = -45
        hor = np.random.randint(hor_min, hor_max)

   
        fps = 7
        hors = []
        vers = []
        radii = []
        ts = []


        # Generate 14 uniformly spaced values between -45 and 45
        values = np.linspace(-45, 45, 14)
        # Round the values to the nearest integers
        int_values = np.round(values).astype(int)
        # Convert to a list
        int_list = list(int_values)


        ### ----------------------------------------------------------------------------------------------------
        hors = [44]*14
        vers = [0]*14
        radii = [3]*14
        ts = list(range(14))
        b_limit = 0.6
        sd_prompt = 'robot'
        inverse_steps = 10
        ### ----------------------------------------------------------------------------------------------------




        # hor = hor_min
        # t_start = 0
        # snaps = 20
        # for h_idx in range(snaps):
        #     hors.append(hor)
        #     ts.append(t_start)
        #     vers.append(0)
        #     radii.append(3)


        #     print(list(range(nframes)))
        #     if h_idx == 5:

        #         ts = ts + list(range(t_start, nframes))
        #         ts = ts + list(reversed(range(nframes, t_start)))

        #         add_num = len( list(range(t_start, nframes)) ) + len( list(reversed(range(nframes, t_start))) )

        #         hors = hors + [hor]*add_num
        #         vers = vers + [0]*add_num
        #         radii = radii + [3]*add_num
                
        #     hor = hor + int((hor_max - hor_min)/snaps)




        
        images_pt = []
        front_list =[]
        full_list =[]
        for k in range( len(hors) ):
            ver = vers[k]
            hor = hors[k]
            radius = radii[k]
            t = ts[k]
            print(t)

            pose = orbit_camera(ver, hor, radius)
            cur_cam_3d = MiniCam(pose, 512, 512, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,)


            # with torch.no_grad():
            #     outputs = self.GSlist[t].render(cur_cam_3d)


            self.renderer_3d.initialize( os.path.join( self.opt.save_plys , f'model_{t}.ply') )  
            with torch.no_grad():
                outputs = self.renderer_3d.render(cur_cam_3d)


            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            front_list.append(out)


            # Assuming out["image"] and out["alpha"] are provided and correctly shaped
            image = outputs["image"]  # This should be [3, 512, 512]
            alpha = outputs["alpha"]  # This should be [1, 512, 512]
            # Expand the alpha tensor to match the foreground/background shape for broadcasting
            alpha = alpha.expand_as(image)
            image = image.unsqueeze(0) # [1, 3, H, W] in [0, 1]
            alpha = alpha.unsqueeze(0) # [1, 1, H, W] in [0, 1]

            azimuth_shift = -hor 
            if azimuth_shift <= -90 :
                azimuth_shift = -90
            if azimuth_shift >= 90:
                azimuth_shift = 90
            start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))* (self.new_width - 512)  )
            
            image = alpha * image + (1 - alpha) * self.bgs[t].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
            #image = alpha * image + (1 - alpha) * bgs[t].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
            
            full_list.append( image_pt2pil(image[0,:,:,:]) )
            images_pt.append( image )

        export_to_gif(front_list, os.path.join( self.opt.visual_dir , f'edit3dwithpth_front.gif'), fps = fps) ################################################
        export_to_gif(full_list, os.path.join( self.opt.visual_dir , f'edit3dwithpth_full.gif'), fps = fps) ###################################################


        print(f"[INFO] loading SD...")
        from guidance.sd_utils import StableDiffusion
        self.guidance_sd = StableDiffusion(self.device)
        #self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])
        print(f"[INFO] loaded SD!")


        images_pt = torch.cat(images_pt, dim = 0)
        imgs_reverse = []
        for b_idx in range(images_pt.shape[0]):
            print(f'######### frames {b_idx} #############')
            backratio = b_limit - radii[b_idx]/10
            imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(images_pt[b_idx:b_idx+1], prompts = sd_prompt, negative_prompts = '', back_ratio_list = [backratio])
            imgs_reverse.append( imgs_reverse_batch )

        imgs_reverse = torch.cat(imgs_reverse, dim=0)
        pred_rgb = imgs_reverse

        print(f"[INFO] offloading SD...")
        del self.guidance_sd 
        print(f"[INFO] offloading SD!")

        print(f"[INFO] loading SVD...")
        from guidance.svd_utils_ import StableVideoDiffusion
        self.guidance_svd = StableVideoDiffusion(self.device)
        print(f"[INFO] loaded SVD!")


        print(f'inversion for video')
        self.guidance_svd.get_img_embeds( image_pt2pil(pred_rgb[0,:,:,:]) )
        frames = self.guidance_svd.get_inverse_loop(pred_rgb = pred_rgb, inverse_steps = inverse_steps)

        ### visual check
        frames_uint8 = (frames * 255).astype(np.uint8)
        pngs = [Image.fromarray(frame) for frame in frames_uint8]
        export_to_gif(pngs, os.path.join( self.opt.visual_dir , f'inversionedit3dwithpth_full.gif'), fps = fps) #################################


        print(f"[INFO] offloading SVD...")
        del self.guidance_svd 
        print(f"[INFO] offloaded SVD!")




        front_list =[]
        full_list =[]
        images_pt = []
        for k in range( len(hors) ):
            ver = vers[k]
            hor = hors[k]
            radius = radii[k]
            t = ts[k]

            pose = orbit_camera(ver, hor, radius)
            cur_cam_3d = MiniCam(pose, 512, 512, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,)


            with torch.no_grad():
                outputs = self.GSlist[t].render(cur_cam_3d)


            # self.renderer_3d.initialize( os.path.join( self.opt.save_plys , f'model_{t}.ply') )  
            # with torch.no_grad():
            #     outputs = self.renderer_3d.render(cur_cam_3d)


            out = outputs["image"].cpu().detach().numpy().astype(np.float32)
            out = np.transpose(out, (1, 2, 0))
            out = Image.fromarray(np.uint8(out*255))
            front_list.append(out)


            # Assuming out["image"] and out["alpha"] are provided and correctly shaped
            image = outputs["image"]  # This should be [3, 512, 512]
            alpha = outputs["alpha"]  # This should be [1, 512, 512]
            # Expand the alpha tensor to match the foreground/background shape for broadcasting
            alpha = alpha.expand_as(image)
            image = image.unsqueeze(0) # [1, 3, H, W] in [0, 1]
            alpha = alpha.unsqueeze(0) # [1, 1, H, W] in [0, 1]

            azimuth_shift = -hor 
            if azimuth_shift <= -90 :
                azimuth_shift = -90
            if azimuth_shift >= 90:
                azimuth_shift = 90
            start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))* (self.new_width - 512)  )
            
            image = alpha * image + (1 - alpha) * self.bgs[t].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
            #image = alpha * image + (1 - alpha) * bgs[t].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
            
            full_list.append( image_pt2pil(image[0,:,:,:]) )
            images_pt.append(image)

        export_to_gif(front_list, os.path.join( self.opt.visual_dir , f'edit4dwithpth_front.gif'), fps = fps) #####################################
        export_to_gif(full_list, os.path.join( self.opt.visual_dir , f'edit4dwithpth_full.gif'), fps = fps) ######################################


        print(f"[INFO] loading SD...")
        from guidance.sd_utils import StableDiffusion
        self.guidance_sd = StableDiffusion(self.device)
        #self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])
        print(f"[INFO] loaded SD!")


        images_pt = torch.cat(images_pt, dim = 0)
        imgs_reverse = []
        for b_idx in range(images_pt.shape[0]):
            print(f'######### frames {b_idx} #############')
            backratio = b_limit - radii[b_idx]/10
            imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(images_pt[b_idx:b_idx+1], prompts = sd_prompt, negative_prompts = '', back_ratio_list = [backratio])
            imgs_reverse.append( imgs_reverse_batch )

        imgs_reverse = torch.cat(imgs_reverse, dim=0)
        pred_rgb = imgs_reverse

        print(f"[INFO] offloading SD...")
        del self.guidance_sd 
        print(f"[INFO] offloading SD!")

        print(f"[INFO] loading SVD...")
        from guidance.svd_utils_ import StableVideoDiffusion
        self.guidance_svd = StableVideoDiffusion(self.device)
        print(f"[INFO] loaded SVD!")


        print(f'inversion for video')
        self.guidance_svd.get_img_embeds( image_pt2pil(pred_rgb[0,:,:,:]) )
        frames = self.guidance_svd.get_inverse_loop(pred_rgb = pred_rgb, inverse_steps = inverse_steps)

        ### visual check
        frames_uint8 = (frames * 255).astype(np.uint8)
        pngs = [Image.fromarray(frame) for frame in frames_uint8]
        export_to_gif(pngs, os.path.join( self.opt.visual_dir , f'inversionedit4dwithpth_full.gif'), fps = fps) #################################


        print(f"[INFO] offloading SVD...")
        del self.guidance_svd 
        print(f"[INFO] offloaded SVD!")










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
os.makedirs( os.path.join( '../demo_output' , opt.run_name ), exist_ok = True )
os.makedirs( os.path.join( '../demo_output' , opt.run_name, 'ply_3d' ), exist_ok = True )
os.makedirs( os.path.join( '../demo_output' , opt.run_name, 'pth_4d' ), exist_ok = True )
os.makedirs( os.path.join( '../demo_output' , opt.run_name, 'plys_4d' ), exist_ok = True )

os.makedirs( os.path.join( '../demo_output' , opt.run_name, 'backgrounds', 'prompt' ), exist_ok = True )
os.makedirs( os.path.join( '../demo_output' , opt.run_name, 'backgrounds', 'generates' ), exist_ok = True )

if not os.path.exists( os.path.join( '../demo_output' , opt.run_name, 'backgrounds', 'prompt', 'seed.txt' ) ):
    with open( os.path.join( '../demo_output' , opt.run_name, 'backgrounds', 'prompt', 'seed.txt' ) , 'w') as file:
        file.write(str( 0 ))




opt.load_plys = os.path.join( '../demo_logs_plys', opt.plys_name )


opt.load = os.path.join( '../demo_output' , opt.run_name, 'ply_3d', 'model.ply' )
opt.load_edit = os.path.join( '../demo_output' , opt.run_name, 'logs_ply', opt.ply_name )


opt.deform_load = os.path.join( '../demo_output' , opt.run_name, 'pth_4d', 'model' )

opt.save_plys = os.path.join( '../demo_cache' , opt.run_name )
os.makedirs( opt.save_plys, exist_ok = True )

opt.num_frames = 14


target_dir = os.path.join( '../demo_output' , opt.run_name, 'backgrounds', 'prompt')
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
opt.case_dir = os.path.join( '../demo_output' , opt.run_name)

opt.bg_dir = os.path.join( '../demo_output' , opt.run_name, 'backgrounds', 'generates')


opt.visual_dir = os.path.join( '../demo_output' , opt.run_name, 'final_visuals')
os.makedirs( opt.visual_dir, exist_ok = True )


#opt.load = None
opt.savename = None
opt.input = None
opt.save_path = None
opt.batch_size = 24
opt.prompt = 'robot'
opt.negative_prompt = 'negative'
opt.start_vsd = 2000



gui = GUI(opt)
