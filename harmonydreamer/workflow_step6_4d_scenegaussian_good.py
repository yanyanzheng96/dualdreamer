import os
import cv2
import time
import tqdm
import numpy as np
import itertools

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.lora import LoRALinearLayer
from diffusers.utils import load_image, export_to_video, export_to_gif

#from FastSAM.fastsam import FastSAM, FastSAMPrompt

import torch
import torch.nn.functional as F

import rembg

from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam


from grid_put import mipmap_linear_grid_put_2d
from mesh import Mesh, safe_normalize

import copy
import glob
import wandb
from PIL import Image, ImageDraw
import imageio

import torchvision.transforms.functional as TF

import datetime
import matplotlib.pyplot as plt

import shutil
import sys

from PIL import Image


def compute_intrinsic_matrix(fovx, fovy, width, height):
    """
    Compute the camera intrinsic matrix K from the given parameters.
    
    Parameters:
    fovx (float): Horizontal field of view in radians.
    fovy (float): Vertical field of view in radians.
    width (int): Image width in pixels.
    height (int): Image height in pixels.
    
    Returns:
    np.ndarray: The intrinsic matrix K.
    """
    
    # Compute the focal lengths in pixels
    fx = width / (2 * np.tan(fovx / 2))
    fy = height / (2 * np.tan(fovy / 2))
    
    # Assume the principal point is at the center of the image
    cx = width / 2
    cy = height / 2
    
    # Construct the intrinsic matrix K
    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    
    return K


def save_as_ply(pts_coord_world, pts_colors, filename="output.ply"):
    assert pts_coord_world.shape[1] == pts_colors.shape[0], "Coordinate and color arrays must have a matching number of points."
    
    # Ensuring color values are within the expected integer range
    pts_colors = np.clip(pts_colors*255, 0, 255).astype(np.uint8)
    
    # Transpose coordinates to match colors in shape
    pts_coord_world = pts_coord_world.T

    points_count = pts_colors.shape[0]
    
    # Define the PLY header for a file with vertex colors
    ply_header = '''ply
format ascii 1.0
element vertex {0}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''.format(points_count)
    
    # Combine the coordinates and colors for saving
    data_to_save = np.hstack([pts_coord_world, pts_colors])

    # Writing the PLY file
    with open(filename, 'w') as ply_file:
        ply_file.write(ply_header)
        np.savetxt(ply_file, data_to_save, fmt="%f %f %f %d %d %d")



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
        
        # load input data from cmdline
        if self.opt.input is not None: # True
            #self.load_input(self.opt.input) # load imgs, if has bg, then rm bg; or just load imgs
            self.load_dir(self.opt.input)
        
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


        self.GSlist_depth = []
        for i in range(self.opt.num_frames):
            renderer_3d = Renderer(sh_degree=self.opt.sh_degree)
            self.GSlist_depth.append(renderer_3d)


        os.makedirs( os.path.join( self.opt.data_dir , 'demo_cache' ) , exist_ok = True)
        os.makedirs( os.path.join( self.opt.data_dir , 'demo_logs_4D' ) , exist_ok = True)


        # renderer
        # self.renderer_template = Renderer(sh_degree=self.opt.sh_degree)

        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.renderer_add1 = Renderer(sh_degree=self.opt.sh_degree)

        #self.renderer_last = Renderer(sh_degree=self.opt.sh_degree)

        # override if provide a checkpoint
        for b_idx in range(self.opt.num_frames):

            self.renderer.initialize( os.path.join(self.opt.case_dir, 'plys_4d', f'model_{b_idx}.ply') )  # torch.Size([17290, 3])
            self.renderer_add1.initialize( os.path.join(self.opt.case_dir, 'plys_4d', 'add', f'model_{b_idx}.ply')  ) # 

            xyz = self.renderer.gaussians._xyz
            # Step 1: Compute the centroid
            centroid = xyz.mean(dim=0)
            # Step 2: Translate points to have the centroid at the origin
            translated_xyz = xyz - centroid
            # Step 3: Scale the coordinates
            scale_factor = 1.0  # Define your scale factor
            scaled_xyz = translated_xyz * scale_factor
            # Step 4: Optionally translate back to the original centroid
            xyz_1 = scaled_xyz + centroid
        
            xyz_1[:,0] = xyz_1[:,0]
            xyz_1[:,1] = xyz_1[:,1] 



            xyz = self.renderer_add1.gaussians._xyz
            # Step 1: Compute the centroid
            centroid = xyz.mean(dim=0)
            # Step 2: Translate points to have the centroid at the origin
            translated_xyz = xyz - centroid
            # Step 3: Scale the coordinates
            scale_factor = 1.0  # Define your scale factor
            scaled_xyz = translated_xyz * scale_factor
            # Step 4: Optionally translate back to the original centroid
            xyz_2 = scaled_xyz + centroid
            xyz_2[:,1] = xyz_2[:,1] + 0


            self.renderer.gaussians._xyz = torch.cat([xyz_1, xyz_2], dim = 0)
            self.renderer.gaussians._features_dc = torch.cat([self.renderer.gaussians._features_dc, self.renderer_add1.gaussians._features_dc], dim = 0)
            self.renderer.gaussians._features_rest = torch.cat([self.renderer.gaussians._features_rest, self.renderer_add1.gaussians._features_rest], dim = 0)
            self.renderer.gaussians._scaling = torch.cat([self.renderer.gaussians._scaling, self.renderer_add1.gaussians._scaling], dim = 0)
            self.renderer.gaussians._rotation = torch.cat([self.renderer.gaussians._rotation, self.renderer_add1.gaussians._rotation], dim = 0)
            self.renderer.gaussians._opacity = torch.cat([self.renderer.gaussians._opacity, self.renderer_add1.gaussians._opacity], dim = 0)
            
            self.renderer.gaussians.save_ply( os.path.join( self.opt.data_dir, f'demo_cache/test_{b_idx}.ply') )



            #self.GSlist[b_idx].initialize( os.path.join(self.opt.case_dir, 'plys_4d', f'model_{b_idx}.ply') )

            self.GSlist[b_idx].initialize( os.path.join( self.opt.data_dir, 'demo_logs_plys', 'exp_2024-05-27 11:52:27_4D', 'step_30', f'model_{b_idx}.ply' ) )




            #self.renderer.initialize(num_pts=5000)
            # setup training
            self.GSlist[b_idx].gaussians.training_setup(self.opt)
            # do not do progressive sh-level
            self.GSlist[b_idx].gaussians.active_sh_degree = self.GSlist[b_idx].gaussians.max_sh_degree

        del self.renderer.gaussians
        del self.renderer_add1.gaussians

        self.timestamp = self.opt.timestamp


        self.d_model = torch.hub.load('./ZoeDepth', 'ZoeD_N', source='local', pretrained=True).to('cuda')
        self.K = compute_intrinsic_matrix(self.cam.fovx, self.cam.fovy, width=512, height=512)


        self.seed_everything()



    def d(self, im):
        return self.d_model.infer_pil(im)



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

    def prepare_train(self):

        self.step = 0

        # # setup training
        # self.renderer.gaussians.training_setup(self.opt)

        # # do not do progressive sh-level
        # self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        # self.optimizer = self.renderer.gaussians.optimizer

        # # fix attributes 
        # self.renderer.gaussians._opacity.requires_grad = False
        # self.renderer.gaussians._scaling.required_grad = False
        # self.renderer.gaussians._rotation.requires_grad = False

        # ### loadng segmentation model
        # print(f"[INFO] loading fastsam ...")
        # seg_model = FastSAM('./FastSAM/ckpts/FastSAM-x.pt')
        # print(f"finished loading!")


        # ### yan_vsd to add lora parameters to optimizer
        # print(f"[INFO] loading SD pipeline in GUI initialization for lora preparation...")
        # self.pipe = StableDiffusionPipeline.from_pretrained(
        #     "runwayml/stable-diffusion-v1-5", torch_dtype = torch.float32
        # )
        # print(f"finished loading!")

        with torch.no_grad():


            # prepare inverse 
            self.back_ratio_list = [0.0]

            # self.back_gt_list = self.guidance_zero123.get_inverse((self.input_img_torch).repeat(self.opt.batch_size,1,1,1), [self.opt.elevation]*self.opt.batch_size, [0]*self.opt.batch_size, [self.opt.radius]*self.opt.batch_size, back_ratio_list = self.back_ratio_list)
            
            # #self.back_gt_list = self.guidance_zero123.get_inverse_loop((self.input_img_torch).repeat(self.opt.batch_size,1,1,1), [self.opt.elevation]*self.opt.batch_size, [0]*self.opt.batch_size, [self.opt.radius]*self.opt.batch_size, back_ratio_list = self.back_ratio_list)




    def reset_imageset(self, casename):
        self.input_img_list = self.dict_for_image_list[casename]
        self.input_mask_list = self.dict_for_mask_list[casename]

        self.input_img = self.input_img_list[0]
        self.input_mask = self.input_mask_list[0]

        # input image
        if self.input_img is not None:
            self.input_img_torch = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_img_torch = F.interpolate(self.input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

            self.input_mask_torch = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
            self.input_mask_torch = F.interpolate(self.input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        if self.input_img_list is not None:
            self.input_img_torch_list = [torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device) for input_img in self.input_img_list]
            self.input_img_torch_list = [F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False) for input_img_torch in self.input_img_torch_list]
            
            self.input_mask_torch_list = [torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device) for input_mask in self.input_mask_list]
            self.input_mask_torch_list = [F.interpolate(input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False) for input_mask_torch in self.input_mask_torch_list]




    def train_depth(self, i):

        self.opt.num_frames = 14
        num_depth = 30



        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if True:

            # update lr
            #self.renderer.gaussians.update_learning_rate(self.step)

            ### Novel view with no gradient! use self.renderer_template.render------------------------------------------------------------------------------------
            #np.random.seed(42)  

            render_resolution = 512
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
            for _ in range(num_depth):
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                ver = 0
                #hor = np.random.randint(-180, 180)
                #hor = np.random.randint(-45, 45)
                hor = -45 + 3*_
                #hor = 15

                # radius = 0
                radius = np.random.uniform(3, 4) 
                radius = 3

                # if _ > int(self.opt.batch_size/2):
                #     ver = vers[0] + np.random.randint(-1, 1)
                #     hor = hors[0] + np.random.randint(-3, 3)
                #     radius = radii[0]


                vers.append(ver)
                hors.append(hor)
                radii.append(radius)
                pose = orbit_camera(self.opt.elevation + ver, hor, radius)
                poses.append(pose)



            if i==0:
                render_frames_list = []
                alphatemp_frames_list = []
                for b_idx in range(self.opt.num_frames):
                    images = []
                    alphas = []
                    alphas_template = []
                    for _ in range(num_depth): # this batch_size is for number of views 
                        cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        #out = self.renderer_template.render(cur_cam, bg_color=bg_color)
                        out = self.GSlist[b_idx].render(cur_cam, bg_color=bg_color)

                        # image = out["image"]  # torch.Size([3, 512, 512])
                        # alpha = out["alpha"]  # torch.Size([1, 512, 512])

                        # Assuming out["image"] and out["alpha"] are provided and correctly shaped
                        image = out["image"]  # This should be [3, 512, 512]
                        alpha = out["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha = alpha.expand_as(image)
                        # Blend the images
                        # The formula for blending is: result = alpha * foreground + (1 - alpha) * background

                        image = image.unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        alpha = alpha.unsqueeze(0) # [1, 1, H, W] in [0, 1]


                        hor = hors[_]
                        azimuth_shift = -hor 
                        if azimuth_shift <= -90 :
                            azimuth_shift = -90
                        if azimuth_shift >= 90:
                            azimuth_shift = 90
                        start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))*(self.new_width - 512) )
                        image = alpha * image + (1 - alpha) * self.bgs[b_idx].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
                        #print(self.bgs[b_idx].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512].shape)



                        images.append(image)
                        alphas.append(alpha) # each mask saved in the list is of shape 1, 512, 512


                        #out_template = self.renderer.render(cur_cam, bg_color=bg_color)
                        out_template = out
                        alpha_template = out_template["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha_template = alpha_template.expand_as(image)
                        alphas_template.append(alpha_template)

                    images = torch.cat(images, dim=0)
                    alphas = torch.cat(alphas, dim=0)
                    alphas_template = torch.cat(alphas_template, dim=0)
                    render_frames_list.append(images)   # len 14(self.opt.num_frames),  10*3*512*512 
                    alphatemp_frames_list.append(alphas_template)


            if i > 0:
                render_frames_list = []
                alphatemp_frames_list = []
                for b_idx in range(self.opt.num_frames):
                    images = []
                    alphas = []
                    alphas_template = []
                    for _ in range(num_depth): # this batch_size is for number of views 
                        cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        #out = self.renderer_template.render(cur_cam, bg_color=bg_color)
                        out = self.GSlist_depth[b_idx].render(cur_cam, bg_color=bg_color)

                        # image = out["image"]  # torch.Size([3, 512, 512])
                        # alpha = out["alpha"]  # torch.Size([1, 512, 512])

                        # Assuming out["image"] and out["alpha"] are provided and correctly shaped
                        image = out["image"]  # This should be [3, 512, 512]
                        alpha = out["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha = alpha.expand_as(image)
                        # Blend the images
                        # The formula for blending is: result = alpha * foreground + (1 - alpha) * background

                        image = image.unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        alpha = alpha.unsqueeze(0) # [1, 1, H, W] in [0, 1]


                        # hor = hors[_]
                        # azimuth_shift = -hor 
                        # if azimuth_shift <= -90 :
                        #     azimuth_shift = -90
                        # if azimuth_shift >= 90:
                        #     azimuth_shift = 90
                        # start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))*(self.new_width - 512) )
                        # image = alpha * image + (1 - alpha) * self.bgs[b_idx].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
                        # #print(self.bgs[b_idx].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512].shape)



                        images.append(image)
                        alphas.append(alpha) # each mask saved in the list is of shape 1, 512, 512


                        #out_template = self.renderer.render(cur_cam, bg_color=bg_color)
                        out_template = out
                        alpha_template = out_template["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha_template = alpha_template.expand_as(image)
                        alphas_template.append(alpha_template)

                    images = torch.cat(images, dim=0)
                    alphas = torch.cat(alphas, dim=0)
                    alphas_template = torch.cat(alphas_template, dim=0)
                    render_frames_list.append(images)   # len 14(self.opt.num_frames),  10*3*512*512 
                    alphatemp_frames_list.append(alphas_template)



            ################################ apply image inversion 
            # self.guidance_sd.to("cuda")

            print(f"[INFO] loading SD...")
            from guidance.sd_utils import StableDiffusion
            self.guidance_sd = StableDiffusion(self.device)
            #self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])
            print(f"[INFO] loaded SD!")


            num_reverse = num_depth

            reverse_frames_list = []
            for b_idx in range(self.opt.num_frames):
            
                render_views = render_frames_list[b_idx]


                image_inverses = []
                imgs_reverse = []
                render_views_ = torch.clone(render_views)

                # Calculate the total number of samples
                total_samples = render_views_.size(0)

                with torch.no_grad(): 
                    for t in range(total_samples):
                        print(f'making reverse view {t}/{num_reverse} in frame {b_idx}')
                        if t <= num_reverse - 1:
                        #if b_idx == 0:
                            backratio = 0.6 - radii[_]/10
                            imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(render_views_[t:t+1], prompts = self.opt.prompt, negative_prompts = '', back_ratio_list = [backratio])

                            image_inverses = image_inverses + image_inverses_batch
                            imgs_reverse.append(imgs_reverse_batch)

                            #image_pt2pil(imgs_reverse_batch[0,:,:,:]).save(f'paper_demo/imageinverse_hor15_{b_idx}.png')

    
                        if t > num_reverse - 1:
                        #if b_idx > 0:
                            image_inverses.append( image_pt2pil(render_views_[t,:,:,:])  )
                            imgs_reverse.append( render_views_[t:t+1,:,:,:] )

                    imgs_reverse = torch.cat(imgs_reverse, dim=0)

                reverse_frames_list.append(torch.clone(imgs_reverse).detach())
            
            print(f"[INFO] offloading SD...")
            del self.guidance_sd 
            print(f"[INFO] offloading SD!")


            # self.guidance_sd.to("cpu")

            ################################ apply video inversion 

            ################################ apply video inversion 
            #self.guidance_svd.device = 'cuda'

            print(f"[INFO] loading SVD...")
            from guidance.svd_utils_ import StableVideoDiffusion
            self.guidance_svd = StableVideoDiffusion(self.device)
            print(f"[INFO] loaded SVD!")


            svdinv_frames_list = []

            for _ in range(num_reverse):
                pred_rgb_list = []
                for b_idx in range(self.opt.num_frames):
                    pred_rgb_list.append( reverse_frames_list[b_idx][_:_+1, :, :, :] )
                pred_rgb = torch.cat( pred_rgb_list, dim = 0 )

                print(f'inversion for training step {self.step} view {_} hor {hors[_]}')
                self.guidance_svd.get_img_embeds( image_pt2pil(pred_rgb[0,:,:,:]) )
                frames = self.guidance_svd.get_inverse_loop(pred_rgb = pred_rgb, inverse_steps = 10)

                ### visual check
                frames_uint8 = (frames * 255).astype(np.uint8)
                pngs = [Image.fromarray(frame) for frame in frames_uint8]
                os.makedirs( os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D' ), exist_ok = True )
                export_to_gif(pngs, os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D/inversion_{self.step}_hor_{hors[_]}.gif' ) )
            
                frames_pt = torch.tensor( frames ).permute(0, 3, 1, 2).to("cuda")
                
                for b_idx in range(self.opt.num_frames):
                    reverse_frames_list[b_idx][_:_+1, :, :, :] = frames_pt[b_idx:b_idx+1, :, :, :]

            # self.guidance_svd.device='cpu'

            print(f"[INFO] offloading SVD...")
            del self.guidance_svd 
            print(f"[INFO] offloaded SVD!")


                # while total_samples > 0:
                #     batch_size = min(total_samples, 6)
                #     batch = render_views_[:batch_size]
                #     render_views_ = render_views_[batch_size:]
                #     total_samples -= batch_size
                #     print("Batch shape:", batch.shape)

                #     imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(batch, prompts = self.opt.prompt, negative_prompts = '', back_ratio_list = [0.3])
                #     image_inverses = image_inverses + image_inverses_batch
                #     imgs_reverse.append(imgs_reverse_batch)

            
            # # GS init generation
            # H, W, K = 512, 512, self.K

            # x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy') # pixels
            # edgeN = 2
            # edgemask = np.ones((H-2*edgeN, W-2*edgeN))
            # edgemask = np.pad(edgemask, ((edgeN,edgeN),(edgeN,edgeN)))


            # # render_frames_list.append(images)   # len 14(self.opt.num_frames),  10*3*512*512 
            # for images_pt in reverse_frames_list:
            #     for c in range(images_pt.shape[0]):
            #         image_curr = image_pt2pil( images_pt[c,:,:,:] )
            #         depth_curr = self.d(image_curr)

            #         R0, T0 = poses[c][:3,:3], poses[c][:3,3:4]

            #         pts_coord_cam = np.matmul(np.linalg.inv(K), np.stack((x*depth_curr, y*depth_curr, 1*depth_curr), axis=0).reshape(3,-1))
            #         new_pts_coord_world2 = (np.linalg.inv(R0).dot(pts_coord_cam) - np.linalg.inv(R0).dot(T0)).astype(np.float32) ## new_pts_coord_world2
            #         new_pts_colors2 = (np.array(image_curr).reshape(-1,3).astype(np.float32)/255.) ## new_pts_colors2
            #         pts_coord_world, pts_colors = new_pts_coord_world2.copy(), new_pts_colors2.copy()

            #         save_as_ply(pts_coord_world, pts_colors, f"./test_visual/depth.ply")

            #         break

            #         #print(pts_coord_world.shape) # (3, 262144), numpy.ndarray
            #         #breakpoint()

            #         # # Normalize the depth array to the range 0-255 for visualization
            #         # normalized_depth = (depth_curr - np.min(depth_curr)) / (np.max(depth_curr) - np.min(depth_curr))
            #         # depth_image = ((1.0 - normalized_depth) * 255).astype(np.uint8)
            #         # # Create a PIL image
            #         # depth_pil_image = Image.fromarray(depth_image, 'L')  # 'L' mode for (8-bit pixels, black and white)
            #         # depth_pil_image.save('./test_visual/test_depth.png')
        
            #         # breakpoint()





            ###########################################################################
            train_iters = 6001

            for iii in range(train_iters):
                # # update lr
                # self.renderer.gaussians.update_learning_rate(iii)
                print(iii)
                self.step += 1

                if iii == 0:
                    from gaussian_model import GaussianModel, BasicPointCloud
                    from sh_utils import eval_sh, SH2RGB, RGB2SH

                    if i == 0:
                        for b_idx in range(self.opt.num_frames):
                            xyz_ = self.GSlist[b_idx].gaussians._xyz.detach().cpu().numpy()

                            extra_pts = 2000
                            x = np.random.uniform(-1, 1, extra_pts)
                            y = np.random.uniform(-1, 1, extra_pts)
                            z = np.random.uniform(-1.5, -1, extra_pts)
                            # Stack the points to create an array of shape (num_pts, 3)
                            xyz = np.stack((x, y, z), axis=1)

                            xyz_ = np.vstack((xyz, xyz_))

                            num_pts = xyz_.shape[0]
                            shs = np.random.random((num_pts, 3)) / 255.0
                            pcd = BasicPointCloud(
                                points=xyz_, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
                            )
                            self.GSlist_depth[b_idx].gaussians.create_from_pcd(pcd, 1)

                            #self.renderer.initialize(num_pts = 5000)

                            # setup training
                            self.GSlist_depth[b_idx].gaussians.training_setup(self.opt)
                            # do not do progressive sh-level
                            self.GSlist_depth[b_idx].gaussians.active_sh_degree = self.GSlist_depth[b_idx].gaussians.max_sh_degree
                            #self.optimizer = self.renderer.gaussians.optimizer
                    
                    if i > 0:
                        for b_idx in range(self.opt.num_frames):
                            xyz_ = self.GSlist_depth[b_idx].gaussians._xyz.detach().cpu().numpy()

                            num_pts = xyz_.shape[0]
                            shs = np.random.random((num_pts, 3)) / 255.0
                            pcd = BasicPointCloud(
                                points=xyz_, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
                            )
                            self.GSlist_depth[b_idx].gaussians.create_from_pcd(pcd, 1)

                            #self.renderer.initialize(num_pts = 5000)

                            # setup training
                            self.GSlist_depth[b_idx].gaussians.training_setup(self.opt)
                            # do not do progressive sh-level
                            self.GSlist_depth[b_idx].gaussians.active_sh_degree = self.GSlist_depth[b_idx].gaussians.max_sh_degree
                            #self.optimizer = self.renderer.gaussians.optimizer




                ### known view ------------------------------------------------------------------------------------
                #if self.input_img_torch is not None:
                loss = 0
                losses = []

                ### novel view with gradient! use self.renderer.render------------------------------------------------------------------------------------
                render_frames_list_ = []
                alpha_frames_list_ = []
                #front_views_list = []

                for b_idx in range(self.opt.num_frames):
                    images = []
                    alphas = []
                    fronts = []

                    # this loop is for rendering 
                    for _ in range(num_depth):

                        # if i > 0:
                        #     breakpoint()
                        cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        out = self.GSlist_depth[b_idx].render(cur_cam, bg_color=bg_color)



                        # Assuming out["image"] and out["alpha"] are provided and correctly shaped
                        image = out["image"]  # This should be [3, 512, 512]
                        alpha = out["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha = alpha.expand_as(image)
                        # Blend the images
                        # The formula for blending is: result = alpha * foreground + (1 - alpha) * background

                        image = image.unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        alpha = alpha.unsqueeze(0) # [1, 1, H, W] in [0, 1]

                        fronts.append(image)

                        hor = hors[_]
                        azimuth_shift = -hor
                        # if azimuth_shift <= -90:
                        #     azimuth_shift = -90
                        # if azimuth_shift >= 90:
                        #     azimuth_shift = 90
                        # start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))* (self.new_width - 512)  )
                        # image = alpha * image + (1 - alpha) * self.bgs[b_idx].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]


                        images.append(image)
                        alphas.append(alpha) # each mask saved in the list is of shape 1, 512, 512
                

                    images = torch.cat(images, dim=0)
                    alphas = torch.cat(alphas, dim=0)
                    render_frames_list_.append(images)
                    alpha_frames_list_.append(alphas)


                for b_idx in range(self.opt.num_frames):
                    loss =  loss + F.mse_loss(render_frames_list_[b_idx], reverse_frames_list[b_idx].detach().to(torch.float32))
                    #loss = loss + F.mse_loss(alpha_frames_list_[b_idx], alphatemp_frames_list[b_idx].detach().to(torch.float32))


                loss_colorconsist = 0
                #loss_xyzconsist = 0
                for b_idx in range(1, self.opt.num_frames):
                    loss_colorconsist = loss_colorconsist + 1000*F.mse_loss( self.GSlist_depth[b_idx].gaussians._features_dc, torch.clone(self.GSlist_depth[0].gaussians._features_dc).detach()) 
                    #loss_colorconsist = loss_colorconsist + 1000*F.mse_loss( self.GSlist[b_idx].gaussians._features_dc, torch.clone(self.GSlist[b_idx-1].gaussians._features_dc).detach()) 

                    #loss_xyzconsist = loss_xyzconsist + 1000*F.mse_loss( self.GSlist[i].gaussians._xyz, torch.clone(self.GSlist[i-1].gaussians._xyz).detach()) 


                print(f'exp {self.opt.timestamp}, step {self.step}, iter {iii}', 'mse:', loss.item(), 'consist:', loss_colorconsist.item())
                #print(f'exp {self.opt.timestamp}, step {self.step}, iter {iii}', 'mse:', loss.item())


                loss = loss + loss_colorconsist

                loss.backward()


                for b_idx in range(self.opt.num_frames):
                    self.GSlist_depth[b_idx].gaussians.optimizer.step()
                    self.GSlist_depth[b_idx].gaussians.optimizer.zero_grad()


                    # # densify and prune
                    # if self.GSlist_depth[b_idx].gaussians._xyz.shape[0] < 100000 and iii%100==0:
                    #     print(f'number of GS in frame {b_idx} is {self.GSlist_depth[b_idx].gaussians._xyz.shape[0]}')
                    #     #if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                    #     viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                    #     self.GSlist_depth[b_idx].gaussians.max_radii2D[visibility_filter] = torch.max(self.GSlist_depth[b_idx].gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    #     self.GSlist_depth[b_idx].gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    #     # if self.step % self.opt.densification_interval == 0:
                    #     #     # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                        
                    #     self.GSlist_depth[b_idx].gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                        
                    #     # if self.step % self.opt.opacity_reset_interval == 0:
                    #     #     self.GSlist_depth[b_idx].gaussians.reset_opacity()







                if iii % 50 == 0 :
                    # render eval
                    front_list =[]
                    full_list =[]
                    nframes = self.opt.num_frames
                    hor = -90
                    delta_hor = 180 / nframes
                    time = 0
                    delta_time = 1
                    hor = np.random.randint(-45, 45)
                    for t in range(nframes):
                        pose = orbit_camera(0, hor, 3)
                        cur_cam_3d = MiniCam(pose, 512, 512, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,)

                        with torch.no_grad():
                            outputs = self.GSlist_depth[t].render(cur_cam_3d)

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

                        # azimuth_shift = -hor 
                        # if azimuth_shift <= -90 :
                        #     azimuth_shift = -90
                        # if azimuth_shift >= 90:
                        #     azimuth_shift = 90
                        # start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))* (self.new_width - 512)  )
                       
                        # image = alpha * image + (1 - alpha) * self.bgs[t].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
                        # #image = alpha * image + (1 - alpha) * bgs[t].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
                       
                        full_list.append( image_pt2pil(image[0,:,:,:]) )


                        time = (time + delta_time) % self.opt.num_frames
                        #hor = (hor+delta_hor) % 360

                    os.makedirs( os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D' ), exist_ok=True )
                    export_to_gif(full_list, os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D/train_full_{self.step}_iter_{iii}_hor_{hor}.gif' ) )

                    self.GSlist_depth[0].gaussians.save_ply( os.path.join( './test_visual', f'model_test.ply' ) )
                    
            step_dir = os.path.join( self.opt.data_dir, f'demo_logs_plys/exp_{self.opt.timestamp}_4D' , f'step_{i}')
            os.makedirs( step_dir, exist_ok=True )
            for b_idx in range(self.opt.num_frames):
                self.GSlist_depth[b_idx].gaussians.save_ply( os.path.join( step_dir, f'model_{b_idx}.ply' ) )
                export_to_gif(full_list, os.path.join( step_dir, f'train_full_step_{i}_hor_{hor}.gif' ) )

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            self.need_update = True















    def train_step(self, i):
        wandb_logdict = {}
        wandb_logdict["iter"] = i


        #self.opt.batch_size = 1
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()



        for _ in range(self.train_steps): # 1

            self.step += 1 # self.step starts from 0
            #step_ratio = min(1, self.step / self.opt.iters) # 1, step / 500
            step_ratio = min(1, self.step / 500) # 1, step / 500

            # update lr
            #self.renderer.gaussians.update_learning_rate(self.step)



            ### Novel view with no gradient! use self.renderer_template.render------------------------------------------------------------------------------------
            #np.random.seed(42)  

            render_resolution = 512
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
            for _ in range(self.opt.batch_size):
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                ver = 0
                #hor = np.random.randint(-180, 180)
                hor = np.random.randint(-45, 45)
                #hor = 15

                # radius = 0
                radius = np.random.uniform(3, 4) 
                radius = 3

                # if _ > int(self.opt.batch_size/2):
                #     ver = vers[0] + np.random.randint(-1, 1)
                #     hor = hors[0] + np.random.randint(-3, 3)
                #     radius = radii[0]


                vers.append(ver)
                hors.append(hor)
                radii.append(radius)
                pose = orbit_camera(self.opt.elevation + ver, hor, radius)
                poses.append(pose)


            from PIL import Image

            # # Load the background image
            # #bg_image_path = './test_bg/ocean_1.png'
            # #bg_image_path = './test_bg_moving/ocean_3_images/output_0000.png'
            # #bg_image_path = './test_bg/room_1.png'
            # bg_image_path = './test_bg_moving/sand.jpg'
            # #bg_image_path = './test_bg_moving/ocean_3.png'
            # #bg_image_path = '../demo_data/ocean_fish_robot/background/images_generate/00000.png'

            # #bg_image_path = f'./test_bg/oceans/oceans_{i%100}.png'

            # # if i%3 == 0:
            # #     bg_image_path = './test_bg/ocean_1.png'
            # # if i%3 == 1:
            # #     bg_image_path = './test_bg/ocean_2.png'
            # # if i%3 == 2:
            # #     bg_image_path = './test_bg/ocean_3.png'

            # # bg_image = Image.open(bg_image_path).convert("RGB")
            # # # Resize the image to the desired size (512x512)
            # # bg_image_resized = bg_image.resize((1024, 512))

            # new_height = 512
            # bg_image_resized, new_height, new_width = resize_image(bg_image_path, new_height = new_height)

            # # Convert the PIL image to a PyTorch tensor
            # bg_tensor = TF.to_tensor(bg_image_resized)
            # # Ensure the tensor is in the shape [3, 512, 512]
            # bg_tensor = bg_tensor[:3, :, :].to("cuda")


            if True:
                render_frames_list = []
                alphatemp_frames_list = []
                for b_idx in range(self.opt.num_frames):
                    images = []
                    alphas = []
                    alphas_template = []
                    for _ in range(self.opt.batch_size): # this batch_size is for number of views 
                        cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        #out = self.renderer_template.render(cur_cam, bg_color=bg_color)
                        out = self.GSlist[b_idx].render(cur_cam, bg_color=bg_color)


                        # image = out["image"]  # torch.Size([3, 512, 512])
                        # alpha = out["alpha"]  # torch.Size([1, 512, 512])


                        # Assuming out["image"] and out["alpha"] are provided and correctly shaped
                        image = out["image"]  # This should be [3, 512, 512]
                        alpha = out["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha = alpha.expand_as(image)
                        # Blend the images
                        # The formula for blending is: result = alpha * foreground + (1 - alpha) * background

                        image = image.unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        alpha = alpha.unsqueeze(0) # [1, 1, H, W] in [0, 1]


                        hor = hors[_]
                        azimuth_shift = -hor 
                        if azimuth_shift <= -90 :
                            azimuth_shift = -90
                        if azimuth_shift >= 90:
                            azimuth_shift = 90
                        start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))*(self.new_width - 512) )
                        image = alpha * image + (1 - alpha) * self.bgs[b_idx].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]
                        #print(self.bgs[b_idx].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512].shape)



                        images.append(image)
                        alphas.append(alpha) # each mask saved in the list is of shape 1, 512, 512


                        #out_template = self.renderer.render(cur_cam, bg_color=bg_color)
                        out_template = out
                        alpha_template = out_template["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha_template = alpha_template.expand_as(image)
                        alphas_template.append(alpha_template)


                    images = torch.cat(images, dim=0)
                    alphas = torch.cat(alphas, dim=0)
                    alphas_template = torch.cat(alphas_template, dim=0)
                    render_frames_list.append(images)
                    alphatemp_frames_list.append(alphas_template)


                 

            # ################################ apply image inversion 
            # # self.guidance_sd.to("cuda")

            # print(f"[INFO] loading SD...")
            # from guidance.sd_utils import StableDiffusion
            # self.guidance_sd = StableDiffusion(self.device)
            # #self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])
            # print(f"[INFO] loaded SD!")


            # num_reverse = 4

            # reverse_frames_list = []
            # for b_idx in range(self.opt.num_frames):
            
            #     render_views = render_frames_list[b_idx]


            #     image_inverses = []
            #     imgs_reverse = []
            #     render_views_ = torch.clone(render_views)

            #     # Calculate the total number of samples
            #     total_samples = render_views_.size(0)

            #     with torch.no_grad(): 
            #         for t in range(total_samples):
            #             print(f'making reverse view {t}/{num_reverse} in frame {b_idx}')
            #             if t <= num_reverse - 1:
            #             #if b_idx == 0:
            #                 backratio = 0.7 - radii[_]/10
            #                 imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(render_views_[t:t+1], prompts = self.opt.prompt, negative_prompts = '', back_ratio_list = [backratio])

            #                 image_inverses = image_inverses + image_inverses_batch
            #                 imgs_reverse.append(imgs_reverse_batch)

            #                 #image_pt2pil(imgs_reverse_batch[0,:,:,:]).save(f'paper_demo/imageinverse_hor15_{b_idx}.png')

    
            #             if t > num_reverse - 1:
            #             #if b_idx > 0:
            #                 image_inverses.append( image_pt2pil(render_views_[t,:,:,:])  )
            #                 imgs_reverse.append( render_views_[t:t+1,:,:,:] )

            #         imgs_reverse = torch.cat(imgs_reverse, dim=0)

            #     reverse_frames_list.append(torch.clone(imgs_reverse).detach())
            
            # print(f"[INFO] offloading SD...")
            # del self.guidance_sd 
            # print(f"[INFO] offloading SD!")


            # # self.guidance_sd.to("cpu")

            # ################################ apply video inversion 

            # ################################ apply video inversion 
            # #self.guidance_svd.device = 'cuda'

            # print(f"[INFO] loading SVD...")
            # from guidance.svd_utils_ import StableVideoDiffusion
            # self.guidance_svd = StableVideoDiffusion(self.device)
            # print(f"[INFO] loaded SVD!")


            # svdinv_frames_list = []

            # for _ in range(num_reverse):
            #     pred_rgb_list = []
            #     for b_idx in range(self.opt.num_frames):
            #         pred_rgb_list.append( reverse_frames_list[b_idx][_:_+1, :, :, :] )
            #     pred_rgb = torch.cat( pred_rgb_list, dim = 0 )

            #     print(f'inversion for training step {self.step} view {_} hor {hors[_]}')
            #     self.guidance_svd.get_img_embeds( image_pt2pil(pred_rgb[0,:,:,:]) )
            #     frames = self.guidance_svd.get_inverse_loop(pred_rgb = pred_rgb, inverse_steps = 10)

            #     ### visual check
            #     frames_uint8 = (frames * 255).astype(np.uint8)
            #     pngs = [Image.fromarray(frame) for frame in frames_uint8]
            #     os.makedirs( os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D' ), exist_ok = True )
            #     export_to_gif(pngs, os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D/inversion_{self.step}_hor_{hors[_]}.gif' ) )
            
            #     frames_pt = torch.tensor( frames ).permute(0, 3, 1, 2).to("cuda")
                
            #     for b_idx in range(self.opt.num_frames):
            #         reverse_frames_list[b_idx][_:_+1, :, :, :] = frames_pt[b_idx:b_idx+1, :, :, :]

            # # self.guidance_svd.device='cpu'

            # print(f"[INFO] offloading SVD...")
            # del self.guidance_svd 
            # print(f"[INFO] offloaded SVD!")


            #     # while total_samples > 0:
            #     #     batch_size = min(total_samples, 6)
            #     #     batch = render_views_[:batch_size]
            #     #     render_views_ = render_views_[batch_size:]
            #     #     total_samples -= batch_size
            #     #     print("Batch shape:", batch.shape)

            #     #     imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(batch, prompts = self.opt.prompt, negative_prompts = '', back_ratio_list = [0.3])

            #     #     image_inverses = image_inverses + image_inverses_batch
            #     #     imgs_reverse.append(imgs_reverse_batch)

            


            # # if i == 0:
            # #     self.opt.load = None
            # #     self.renderer.initialize(num_pts=5000)
            # #     # setup training
            # #     self.renderer.gaussians.training_setup(self.opt)
            # #     # do not do progressive sh-level
            # #     self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
            # #     self.optimizer = self.renderer.gaussians.optimizer
            # #self.renderer.gaussians._xyz.required_grad = False 


            # # if i % 1 == 0 and i > 0:
            # #     from gaussian_model import GaussianModel, BasicPointCloud
            # #     from sh_utils import eval_sh, SH2RGB, RGB2SH
            # #     num_pts = self.renderer.gaussians._xyz.shape[0]
            # #     shs = np.random.random((num_pts, 3)) / 255.0
            # #     pcd = BasicPointCloud(
            # #         points=self.renderer.gaussians._xyz.detach().cpu().numpy(), colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            # #     )
            # #     self.renderer.gaussians.create_from_pcd(pcd, 1)

            # #     # setup training
            # #     self.renderer.gaussians.training_setup(self.opt)
            # #     # do not do progressive sh-level
            # #     self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
            # #     self.optimizer = self.renderer.gaussians.optimizer









            ###########################################################################
            train_iters = 401

            for iii in range(train_iters):
                # # update lr
                # self.renderer.gaussians.update_learning_rate(iii)
                print(iii)
                self.step += 1

                if iii == 1:
                    from gaussian_model import GaussianModel, BasicPointCloud
                    from sh_utils import eval_sh, SH2RGB, RGB2SH

                    for b_idx in range(self.opt.num_frames):
                        # if i == 0 :
                        #     self.GSlist[b_idx].initialize(num_pts = 10000)
                        if i >= 0:
                            num_pts = self.GSlist[b_idx].gaussians._xyz.shape[0]
                            shs = np.random.random((num_pts, 3)) / 255.0
                            pcd = BasicPointCloud(
                                points=self.GSlist[b_idx].gaussians._xyz.detach().cpu().numpy(), colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
                            )
                            self.GSlist[b_idx].gaussians.create_from_pcd(pcd, 1)


                        #self.renderer.initialize(num_pts = 5000)

                        # setup training
                        self.GSlist[b_idx].gaussians.training_setup(self.opt)
                        # do not do progressive sh-level
                        self.GSlist[b_idx].gaussians.active_sh_degree = self.GSlist[b_idx].gaussians.max_sh_degree
                        
                        #self.optimizer = self.renderer.gaussians.optimizer



                ### known view ------------------------------------------------------------------------------------
                #if self.input_img_torch is not None:
                loss = 0
                losses = []

                ### novel view with gradient! use self.renderer.render------------------------------------------------------------------------------------
                render_frames_list_ = []
                alpha_frames_list_ = []
                #front_views_list = []

                for b_idx in range(self.opt.num_frames):
                    images = []
                    alphas = []
                    fronts = []

                    # this loop is for rendering 
                    for _ in range(self.opt.batch_size):

                        # if i > 0:
                        #     breakpoint()
                        cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        out = self.GSlist[b_idx].render(cur_cam, bg_color=bg_color)



                        # Assuming out["image"] and out["alpha"] are provided and correctly shaped
                        image = out["image"]  # This should be [3, 512, 512]
                        alpha = out["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha = alpha.expand_as(image)
                        # Blend the images
                        # The formula for blending is: result = alpha * foreground + (1 - alpha) * background

                        image = image.unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        alpha = alpha.unsqueeze(0) # [1, 1, H, W] in [0, 1]

                        fronts.append(image)

                        hor = hors[_]
                        azimuth_shift = -hor
                        if azimuth_shift <= -90:
                            azimuth_shift = -90
                        if azimuth_shift >= 90:
                            azimuth_shift = 90
                        start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))* (self.new_width - 512)  )
                        image = alpha * image + (1 - alpha) * self.bgs[b_idx].unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]


                        images.append(image)
                        alphas.append(alpha) # each mask saved in the list is of shape 1, 512, 512
                

                    images = torch.cat(images, dim=0)
                    alphas = torch.cat(alphas, dim=0)
                    render_frames_list_.append(images)
                    alpha_frames_list_.append(alphas)


                for b_idx in range(self.opt.num_frames):
                    loss =  loss + F.mse_loss(render_frames_list_[b_idx], reverse_frames_list[b_idx].detach().to(torch.float32))
                    loss = loss + F.mse_loss(alpha_frames_list_[b_idx], alphatemp_frames_list[b_idx].detach().to(torch.float32))


                loss_colorconsist = 0
                #loss_xyzconsist = 0
                for b_idx in range(1, self.opt.num_frames):
                    loss_colorconsist = loss_colorconsist + 1000*F.mse_loss( self.GSlist[b_idx].gaussians._features_dc, torch.clone(self.GSlist[0].gaussians._features_dc).detach()) 
                    #loss_colorconsist = loss_colorconsist + 1000*F.mse_loss( self.GSlist[b_idx].gaussians._features_dc, torch.clone(self.GSlist[b_idx-1].gaussians._features_dc).detach()) 

                    #loss_xyzconsist = loss_xyzconsist + 1000*F.mse_loss( self.GSlist[i].gaussians._xyz, torch.clone(self.GSlist[i-1].gaussians._xyz).detach()) 


                print(f'exp {self.opt.timestamp}, step {self.step}, iter {iii}', 'mse:', loss.item(), 'consist:', loss_colorconsist.item())


                loss = loss 

                loss.backward()


                for b_idx in range(self.opt.num_frames):
                    self.GSlist[b_idx].gaussians.optimizer.step()
                    self.GSlist[b_idx].gaussians.optimizer.zero_grad()


                if iii % 200 == 0 :
                    # render eval
                    front_list =[]
                    full_list =[]
                    nframes = self.opt.num_frames
                    hor = -90
                    delta_hor = 180 / nframes
                    time = 0
                    delta_time = 1
                    hor = np.random.randint(-45, 45)
                    #hor = 0
                    for t in range(nframes):
                        pose = orbit_camera(0, hor, 3)
                        cur_cam_3d = MiniCam(pose, 512, 512, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far,)

                        with torch.no_grad():
                            outputs = self.GSlist[t].render(cur_cam_3d)

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


                        time = (time + delta_time) % self.opt.num_frames
                        #hor = (hor+delta_hor) % 360

                    os.makedirs( os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D' ), exist_ok=True )
                    export_to_gif(front_list, os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D/train_front_{self.step}_iter_{iii}_hor_{hor}.gif') )
                    export_to_gif(full_list, os.path.join( self.opt.data_dir, f'demo_logs_4D/exp_{self.opt.timestamp}_4D/train_full_{self.step}_iter_{iii}_hor_{hor}.gif' ) )

            step_dir = os.path.join( self.opt.data_dir, f'demo_logs_plys/exp_{self.opt.timestamp}_4D' , f'step_{i}')
            os.makedirs( step_dir, exist_ok=True )
            for b_idx in range(self.opt.num_frames):
                self.GSlist_depth[b_idx].gaussians.save_ply( os.path.join( step_dir, f'model_{b_idx}.ply' ) )
                export_to_gif(full_list, os.path.join( step_dir, f'train_full_step_{i}_hor_{hor}.gif' ) )

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            self.need_update = True



















    
    def load_input(self, file):
        file_list = [file.replace('.png', f'_{x:03d}.png') for x in range(self.opt.batch_size)]
        self.input_img_list, self.input_mask_list = [], []
        for file in file_list:
            # load image
            print(f'[INFO] load image from {file}...')
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            if img.shape[-1] == 3:
                if self.bg_remover is None:
                    self.bg_remover = rembg.new_session()
                img = rembg.remove(img, session=self.bg_remover)
                cv2.imwrite(file.replace('.png', '_rgba.png'), img) 
            img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            input_mask = img[..., 3:]
            # white bg
            input_img = img[..., :3] * input_mask + (1 - input_mask)
            # bgr to rgb
            input_img = input_img[..., ::-1].copy()
            self.input_img_list.append(input_img)
            self.input_mask_list.append(input_mask)

            self.input_img = self.input_img_list[0]
            self.input_mask = self.input_mask_list[0]



        


    # no gui mode
    def train(self, iters=500, ui=False):
        image_list =[]
        from PIL import Image
        from diffusers.utils import export_to_video, export_to_gif
        interval = 1
        nframes = iters // interval # 250
        hor = 180
        delta_hor = 4 * 360 / nframes
        time = 0
        delta_time = 1
        if self.gui:
            from visergui import ViserViewer
            self.viser_gui = ViserViewer(device="cuda", viewer_port=8080)
        if iters > 0:
            self.prepare_train()
            if self.gui:
                self.viser_gui.set_renderer(self.renderer, self.fixed_cam)
            




            for i in tqdm.trange(2000): # 500
                # if i < 200:
                #     self.enable_sd = False 
                # if i >= 200:
                #     self.enable_sd = True
                #print(self.casenames)


                # self.reset_imageset(self.casenames[i % len(self.casenames)])
                # self.renderer.gaussians.training_switch(i % len(self.casenames))

                self.train_depth(i)

                #self.train_step(i)
                if self.gui:
                    self.viser_gui.update()

                # snapshots = [c + s for c in range(len(self.casenames)) for s in snapshots]
                # if i in snapshots:

                #     image_list =[]
                #     from PIL import Image
                #     from diffusers.utils import export_to_video, export_to_gif
                #     nframes = 14 *5
                #     hor = 180
                #     delta_hor = 360 / nframes
                #     time = 0
                #     delta_time = 1
                #     for _ in range(nframes):
                #         pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
                #         cur_cam = MiniCam(
                #             pose,
                #             512,
                #             512,
                #             self.cam.fovy,
                #             self.cam.fovx,
                #             self.cam.near,
                #             self.cam.far,
                #         )

                #         outputs = self.renderer.render(cur_cam)

                #         out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                #         out = np.transpose(out, (1, 2, 0))
                #         out = Image.fromarray(np.uint8(out*255))
                #         image_list.append(out)

                #         time = (time + delta_time) % 14
                #         hor = (hor+delta_hor) % 360

                #     savename = opt.savename
                #     os.makedirs(savename, exist_ok=True)
                #     export_to_gif(image_list, f'{savename}/{opt.save_path}_train_{i}.gif')

                    # if self.opt.debug == False:
                    #     imageio.mimsave('result.gif', image_list, fps=5)  # Adjust fps as needed
                    #     wandb.log({"result_gif": wandb.Image('result.gif')})


                    # ply_mesh = o3d.io.read_triangle_mesh( ply_path )
                    # o3d.io.write_triangle_mesh(f"obj_{self.timestamp}.obj", ply_mesh)
    



        # # save
        # self.save_model(mode='model')
        # self.renderer.gaussians.save_deformation(self.opt.outdir, self.opt.save_path)

        # for t in range(14):
        #     self.save_model(mode='geo+tex', t=t)

        if self.gui:
            while True:
                self.viser_gui.update()

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    args.config = './configs/4d.yaml'


    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    #############################
    opt.data_dir = '/mnt/vita-nas/zy3724/4Dprojects'
    #############################


    opt.case_dir = os.path.join( os.path.join( opt.data_dir, 'demo_output' ) , opt.run_name)


    opt.bg_dir = os.path.join( os.path.join( opt.data_dir, 'demo_output' ) , opt.run_name, 'backgrounds', 'generates')


    # auto find mesh from stage 1
    #opt.load = os.path.join(opt.outdir, opt.save_path + '_model.ply')

    # opt.load = os.path.join(opt.outdir, 'flower' + '_model.ply')w
    # opt.savename = './visuals/flower_offset_balance'
    # opt.input = './data/set/flower'
    # opt.save_path = 'flower'

    opt.load = os.path.join(opt.outdir, 'robot_1' + '_model.ply')

    opt.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs( os.path.join( opt.data_dir, f'demo_logs_4D/exp_{opt.timestamp}_4D' ) , exist_ok=True )
    current_script_path = os.path.abspath(sys.argv[0])
    destination_path = os.path.join( opt.data_dir, f'demo_logs_4D/exp_{opt.timestamp}_4D', os.path.basename(current_script_path))
    shutil.copy2(current_script_path, destination_path)



    #opt.load = None
    opt.savename = './visuals/fishseparate'
    opt.input = None
    opt.save_path = 'robot'
    opt.batch_size = 24
    opt.prompt = 'robot on the moon with another flying pet'
    opt.negative_prompt = 'negative'
    opt.start_vsd = 2000
    opt.num_frames = 14

    # opt.load = os.path.join(opt.outdir, 'cat' + '_model.ply')
    # opt.savename = './visuals/cat_offset_balance'
    # opt.input = './data/set/cat1'
    # opt.save_path = 'cat'

    script_path = os.path.abspath(__file__)



    opt.debug = True 
    if not opt.debug:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="composition",
            name = opt.run_name, 
            # track hyperparameters and run metadata
            config={
            "load": opt.load,
            "savename": opt.savename,
            "input": opt.input,
            "save_path": opt.save_path,
            "batch_size": opt.batch_size, 
            "prompt": opt.prompt,
            "start_vsd": opt.start_vsd,
            },
            settings=wandb.Settings(code_dir=script_path)
        )

        wandb.save(script_path)
        #wandb.save('./guidance/sdmix_utils.py')
        wandb.save('./guidance/sd_utils.py')



    gui = GUI(opt)
    gui.train(opt.iters)