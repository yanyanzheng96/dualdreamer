import os
import cv2
import time
import tqdm
import numpy as np
import itertools

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.lora import LoRALinearLayer

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
        
        # load input data from cmdline
        if self.opt.input is not None: # True
            #self.load_input(self.opt.input) # load imgs, if has bg, then rm bg; or just load imgs
            self.load_dir(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None: # None
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None: # None
            self.nagative_prompt = self.opt.negative_prompt



        # renderer
        self.renderer_template = Renderer(sh_degree=self.opt.sh_degree)

        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.renderer_add1 = Renderer(sh_degree=self.opt.sh_degree)

        self.renderer_last = Renderer(sh_degree=self.opt.sh_degree)
        # override if provide a checkpoint
        if self.opt.load is not None: # not None

            self.renderer.initialize( self.opt.load )
            self.renderer_last.initialize( self.opt.load )


            # # self.renderer.initialize('./logs/robot_1_model.ply')  # torch.Size([17290, 3])
            # # self.renderer_add1.initialize('./logs/fish_model.ply') # torch.Size([7806, 3])

            # self.renderer.initialize('./logs_/robot/model_0.ply')  # torch.Size([17290, 3])
            # self.renderer_add1.initialize('./logs_/fish/model_0.ply') # 

            # #breakpoint()

            # xyz = self.renderer.gaussians._xyz
            # # Step 1: Compute the centroid
            # centroid = xyz.mean(dim=0)
            # # Step 2: Translate points to have the centroid at the origin
            # translated_xyz = xyz - centroid
            # # Step 3: Scale the coordinates
            # scale_factor = 3.0  # Define your scale factor
            # scaled_xyz = translated_xyz * scale_factor
            # # Step 4: Optionally translate back to the original centroid
            # xyz_1 = scaled_xyz + centroid
        
            # xyz_1[:,0] = xyz_1[:,0]
            # xyz_1[:,1] = xyz_1[:,1]

            # self.renderer.gaussians._xyz = xyz_1


            # xyz = self.renderer_add1.gaussians._xyz
            # # Step 1: Compute the centroid
            # centroid = xyz.mean(dim=0)
            # # Step 2: Translate points to have the centroid at the origin
            # translated_xyz = xyz - centroid
            # # Step 3: Scale the coordinates
            # scale_factor = 1.5  # Define your scale factor
            # scaled_xyz = translated_xyz * scale_factor
            # # Step 4: Optionally translate back to the original centroid
            # xyz_2 = scaled_xyz + centroid
            # xyz_2[:,1] = xyz_2[:,1] + 0.5


            # self.renderer.gaussians._xyz = torch.cat([xyz_1, xyz_2], dim = 0)
            # self.renderer.gaussians._features_dc = torch.cat([self.renderer.gaussians._features_dc, self.renderer_add1.gaussians._features_dc], dim = 0)
            # self.renderer.gaussians._features_rest = torch.cat([self.renderer.gaussians._features_rest, self.renderer_add1.gaussians._features_rest], dim = 0)
            # self.renderer.gaussians._scaling = torch.cat([self.renderer.gaussians._scaling, self.renderer_add1.gaussians._scaling], dim = 0)
            # self.renderer.gaussians._rotation = torch.cat([self.renderer.gaussians._rotation, self.renderer_add1.gaussians._rotation], dim = 0)
            # self.renderer.gaussians._opacity = torch.cat([self.renderer.gaussians._opacity, self.renderer_add1.gaussians._opacity], dim = 0)
            
            self.renderer.gaussians.save_ply('./logs/test.ply')
            self.renderer.initialize('./logs/test.ply')
            self.renderer_last.initialize('./logs/test.ply')



            # self.xyz_initialize = torch.clone( self.renderer.gaussians._xyz ).detach()
            # self.renderer.gaussians.load_model(opt.outdir, opt.save_path)             
        else:
            # initialize gaussians to a blob
            self.renderer.initialize(num_pts=self.opt.num_pts)

        #self.renderer.initialize(num_pts=5000)


        self.timestamp = self.opt.timestamp




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

    def prepare_train(self):

        self.step = 0

        # setup training
        self.renderer.gaussians.training_setup(self.opt)

        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        # # fix attributes 
        self.renderer.gaussians._xyz.requires_grad = False
        self.renderer.gaussians._opacity.requires_grad = False
        self.renderer.gaussians._scaling.requires_grad = False
        self.renderer.gaussians._rotation.requires_grad = False

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



        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != "" # False
        self.enable_zero123 = self.opt.lambda_zero123 > 0 # True
        self.enable_svd = self.opt.lambda_svd > 0  #and self.input_img is not None # False

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd: # False
            if self.opt.mvdream: # False
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                # print(f"[INFO] loading SD...")
                # from guidance.sd_utils import StableDiffusion
                # self.guidance_sd = StableDiffusion(self.device)
                # print(f"[INFO] loaded SD!")

                print(f"[INFO] loading VSD...")
                from guidance.vsd_utils import VSD
                self.guidance_vsd = VSD(self.device)
                print(f"[INFO] loaded VSD!")

        



        if self.guidance_svd is None and self.enable_svd: # False
            print(f"[INFO] loading SVD...")
            from guidance.svd_utils import StableVideoDiffusion
            self.guidance_svd = StableVideoDiffusion(self.device)
            print(f"[INFO] loaded SVD!")

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
        
        # prepare embeddings
        with torch.no_grad():

            # if self.enable_sd:
            #self.guidance_vsd.get_text_embeds([self.prompt], [self.negative_prompt])


            # print(f"[INFO] loading zero123...")
            # from guidance.zero123_utils import Zero123
            # self.guidance_zero123 = Zero123(self.device, t_range=[0.02, self.opt.t_max])
            # print(f"[INFO] loaded zero123!")

            print(f"[INFO] loading SD...")
            from guidance.sd_utils import StableDiffusion
            self.guidance_sd = StableDiffusion(self.device)
            self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            print(f"[INFO] loaded SD!")


            # prepare inverse 
            self.back_ratio_list = [0.0]

            # self.back_gt_list = self.guidance_zero123.get_inverse((self.input_img_torch).repeat(self.opt.batch_size,1,1,1), [self.opt.elevation]*self.opt.batch_size, [0]*self.opt.batch_size, [self.opt.radius]*self.opt.batch_size, back_ratio_list = self.back_ratio_list)
            
            # #self.back_gt_list = self.guidance_zero123.get_inverse_loop((self.input_img_torch).repeat(self.opt.batch_size,1,1,1), [self.opt.elevation]*self.opt.batch_size, [0]*self.opt.batch_size, [self.opt.radius]*self.opt.batch_size, back_ratio_list = self.back_ratio_list)


            
            if self.enable_svd:
                self.guidance_svd.get_img_embeds(self.input_img)




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

    def train_step(self, i):
        wandb_logdict = {}
        wandb_logdict["iter"] = i



        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()



        for _ in range(self.train_steps): # 1

            self.step += 1 # self.step starts from 0
            #step_ratio = min(1, self.step / self.opt.iters) # 1, step / 500
            step_ratio = min(1, self.step / 500) # 1, step / 500

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)



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
                hor = np.random.randint(-180, 180)
                #hor = np.random.randint(-45, 45)

                # radius = 0
                radius = np.random.uniform(3, 4) 
                radius = 1

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
            # Load the background image
            #bg_image_path = './test_bg/ocean_1.png'
            #bg_image_path = './test_bg_moving/ocean_3_images/output_0000.png'
            #bg_image_path = './test_bg/room_1.png'
            bg_image_path = self.opt.bg_path
            #bg_image_path = './test_bg_moving/ocean_3.png'
            #bg_image_path = '../demo_data/ocean_fish_robot/background/images_generate/00000.png'



            #bg_image_path = f'./test_bg/oceans/oceans_{i%100}.png'

            # if i%3 == 0:
            #     bg_image_path = './test_bg/ocean_1.png'
            # if i%3 == 1:
            #     bg_image_path = './test_bg/ocean_2.png'
            # if i%3 == 2:
            #     bg_image_path = './test_bg/ocean_3.png'

            # bg_image = Image.open(bg_image_path).convert("RGB")
            # # Resize the image to the desired size (512x512)
            # bg_image_resized = bg_image.resize((1024, 512))

            new_height = 512
            bg_image_resized, new_height, new_width = resize_image(bg_image_path, new_height = new_height)

            # Convert the PIL image to a PyTorch tensor
            bg_tensor = TF.to_tensor(bg_image_resized)
            # Ensure the tensor is in the shape [3, 512, 512]
            bg_tensor = bg_tensor[:3, :, :].to("cuda")


            if True:
                render_views_list = []
                for b in range(len(self.back_ratio_list)):
                    images = []
                    alphas = []
                    alphas_template = []
                    for _ in range(self.opt.batch_size):
                        cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        #out = self.renderer_template.render(cur_cam, bg_color=bg_color)
                        out = self.renderer_last.render(cur_cam, bg_color=bg_color)


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
                        start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))*(new_width - 512) )
                        image = alpha * image + (1 - alpha) * bg_tensor.unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]



                        images.append(image)
                        alphas.append(alpha) # each mask saved in the list is of shape 1, 512, 512


                        out_template = self.renderer.render(cur_cam, bg_color=bg_color)
                        alpha_template = out_template["alpha"]  # This should be [1, 512, 512]
                        # Expand the alpha tensor to match the foreground/background shape for broadcasting
                        alpha_template = alpha_template.expand_as(image)
                        alphas_template.append(alpha_template)

                
                images = torch.cat(images, dim=0)
                alphas = torch.cat(alphas, dim=0)
                alphas_template = torch.cat(alphas_template, dim=0)
                render_views_list.append(images)

                from PIL import Image
                images_images = []
                image_tensor = images
                for r in range(image_tensor.shape[0]):
                    input_img_torch_resized = image_tensor[r,:,:,:].permute(1, 2, 0)
                    input_img_np = input_img_torch_resized.detach().cpu().numpy()
                    input_img_np = (input_img_np * 255).astype(np.uint8)
                    #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversescore/check_view_{iii}_render.png')
                    #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_bg/test_bg_render_{r}.png')
                    images_images.append( Image.fromarray((input_img_np).astype(np.uint8)) )
                
                # bg_tensor_repeated = bg_tensor.unsqueeze(0).repeat(self.opt.batch_size, 1, 1, 1)
                # render_views = torch.clone(render_views_list[0])   
                render_views = images

            # if i > 0:
            #     render_views = self.segments_views
            #     poses = self.poses



            from PIL import Image
            images_render_views = []
            image_tensor = render_views
            for r in range(image_tensor.shape[0]):
                input_img_torch_resized = image_tensor[r,:,:,:].permute(1, 2, 0)
                input_img_np = input_img_torch_resized.detach().cpu().numpy()
                input_img_np = (input_img_np * 255).astype(np.uint8)
                #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversescore/check_view_{iii}_render.png')
                #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_bg/test_bg_render_{r}.png')
                images_render_views.append( Image.fromarray((input_img_np).astype(np.uint8)) )
            


            ############
            #loss_zero123 =  self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio) / (16)
            # loss_zero123 = 0
            # for b, back_ratio in enumerate(self.back_ratio_list):
            #     self.guidance_zero123.embeddings = self.back_embeddings_list[b]
            #     loss_zero123 = loss_zero123 + self.opt.lambda_zero123 * self.guidance_zero123.train_step(render_views_list[b], vers, hors, radii, step_ratio)
            image_inverses = []
            imgs_reverse = []
            render_views_ = torch.clone(render_views)


            # Calculate the total number of samples
            total_samples = render_views_.size(0)

            with torch.no_grad(): 
                for t in range(total_samples):
                    print("Batch sample:", t)

                    if t <= 3:
                        backratio = 0.6 - radii[_]/10
                        imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(render_views_[t:t+1], prompts = self.opt.prompt, negative_prompts = '', back_ratio_list = [backratio])

                        image_inverses = image_inverses + image_inverses_batch
                        imgs_reverse.append(imgs_reverse_batch)
 
                    if t > 3:
                        image_inverses.append( image_pt2pil(render_views_[t,:,:,:])  )
                        imgs_reverse.append( render_views_[t:t+1,:,:,:] )

                imgs_reverse = torch.cat(imgs_reverse, dim=0)


                # while total_samples > 0:
                #     batch_size = min(total_samples, 6)
                #     batch = render_views_[:batch_size]
                #     render_views_ = render_views_[batch_size:]
                #     total_samples -= batch_size
                #     print("Batch shape:", batch.shape)

                #     imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(batch, prompts = self.opt.prompt, negative_prompts = '', back_ratio_list = [0.3])

                #     image_inverses = image_inverses + image_inverses_batch
                #     imgs_reverse.append(imgs_reverse_batch)




                    


            # ### get a larget step reverse view 
            # imgs_reverse_subset, loss_inverse, image_inverses_subset = self.guidance_sd.get_inverse_loop(render_views[0:1,:,:,:], prompts = "swimming fish", negative_prompts = '', back_ratio_list = [0.01])
            # imgs_reverse[0:1,:,:,:] = imgs_reverse_subset



            from PIL import Image
            images_imgs_reverse = []
            image_tensor = imgs_reverse
            for r in range(image_tensor.shape[0]):
                input_img_torch_resized = image_tensor[r,:,:,:].permute(1, 2, 0)
                input_img_np = input_img_torch_resized.detach().cpu().numpy()
                input_img_np = (input_img_np * 255).astype(np.uint8)
                #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversescore/check_view_{iii}_render.png')
                #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_bg/test_bg_inverse_{r}.png')
                images_imgs_reverse.append( Image.fromarray((input_img_np).astype(np.uint8)) )
            

            # if i == 0:
            #     self.opt.load = None
            #     self.renderer.initialize(num_pts=5000)
            #     # setup training
            #     self.renderer.gaussians.training_setup(self.opt)
            #     # do not do progressive sh-level
            #     self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
            #     self.optimizer = self.renderer.gaussians.optimizer
            #self.renderer.gaussians._xyz.required_grad = False 


            # if i % 1 == 0 and i > 0:
            #     from gaussian_model import GaussianModel, BasicPointCloud
            #     from sh_utils import eval_sh, SH2RGB, RGB2SH
            #     num_pts = self.renderer.gaussians._xyz.shape[0]
            #     shs = np.random.random((num_pts, 3)) / 255.0
            #     pcd = BasicPointCloud(
            #         points=self.renderer.gaussians._xyz.detach().cpu().numpy(), colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            #     )
            #     self.renderer.gaussians.create_from_pcd(pcd, 1)

            #     # setup training
            #     self.renderer.gaussians.training_setup(self.opt)
            #     # do not do progressive sh-level
            #     self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
            #     self.optimizer = self.renderer.gaussians.optimizer


            for iii in range(401):
                # # update lr
                # self.renderer.gaussians.update_learning_rate(iii)
                print(self.opt.timestamp, iii)
                self.step += 1
        

                if iii == 1:
                    from gaussian_model import GaussianModel, BasicPointCloud
                    from sh_utils import eval_sh, SH2RGB, RGB2SH
                    num_pts = self.renderer.gaussians._xyz.shape[0]
                    shs = np.random.random((num_pts, 3)) / 255.0
                    pcd = BasicPointCloud(
                        points=self.renderer.gaussians._xyz.detach().cpu().numpy(), colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
                    )
                    self.renderer.gaussians.create_from_pcd(pcd, 1)

                    #self.renderer.initialize(num_pts = 5000)


                    # setup training
                    self.renderer.gaussians.training_setup(self.opt)
                    # do not do progressive sh-level
                    self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
                    self.optimizer = self.renderer.gaussians.optimizer




                ### known view ------------------------------------------------------------------------------------
                #if self.input_img_torch is not None:
                loss_mse = 0

                ### novel view with gradient! use self.renderer.render------------------------------------------------------------------------------------
                render_views_list = []
                front_views_list = []

                for b in range(len(self.back_ratio_list)):
                    images = []
                    alphas = []
                    fronts = []

                    # this loop is for rendering 
                    for _ in range(self.opt.batch_size):

                        # if i > 0:
                        #     breakpoint()
                        cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        out = self.renderer.render(cur_cam, bg_color=bg_color)



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
                        start_pixel = int((azimuth_shift - (-90)) / (90 - (-90))* (new_width - 512)  )
                        image = alpha * image + (1 - alpha) * bg_tensor.unsqueeze(0)[:, :, 0:512, start_pixel:start_pixel + 512]



                        images.append(image)
                        alphas.append(alpha) # each mask saved in the list is of shape 1, 512, 512
                




                    images = torch.cat(images, dim=0)
                    alphas = torch.cat(alphas, dim=0)
                    render_views_list.append(images)

                    fronts = torch.cat(fronts, dim=0)
                    front_views_list.append(fronts)


                # bg_tensor_repeated = bg_tensor.unsqueeze(0).repeat(self.opt.batch_size, 1, 1, 1)
                # render_views = torch.clone(render_views_list[0])   
                render_views = images

                #loss_inverse = F.mse_loss(render_views_list[0], processed_tensor)
                loss_inverse = F.mse_loss(render_views, imgs_reverse.detach().to(torch.float32))
                #loss_inverse = F.mse_loss(fronts, imgs_reverse.detach().to(torch.float32))

                loss_alpha = F.mse_loss(alphas, alphas_template.detach().to(torch.float32)) 


                # pc1 = self.renderer.gaussians._xyz[0:int(self.renderer.gaussians._xyz.shape[0]/2),:]
                # pc2 = self.xyz_initialize
                # n = pc1.size(0)
                # m = pc2.size(0)
                # # Expand pc1 to (n, 1, 3) and pc2 to (1, m, 3) to compute pairwise distance
                # pc1_expanded = pc1.unsqueeze(1).expand(n, m, 3)
                # pc2_expanded = pc2.unsqueeze(0).expand(n, m, 3)
                # # Compute squared distances (n, m)
                # distances_squared = torch.sum((pc1_expanded - pc2_expanded) ** 2, dim=2)
                # # Find the closest points and sum their distances
                # distances_pc1_to_pc2 = torch.min(distances_squared, dim=1)[0]  # Closest points in pc2 for each in pc1
                # distances_pc2_to_pc1 = torch.min(distances_squared, dim=0)[0]  # Closest points in pc1 for each in pc2
                # # Calculate Chamfer Distance
                # chamfer_dist = 1000 * torch.mean(distances_pc1_to_pc2) + torch.mean(distances_pc2_to_pc1)



                # pc1 = self.renderer.gaussians._xyz[0:int(self.renderer.gaussians._xyz.shape[0]/2),:]
                # pc3 = self.renderer.gaussians._xyz[int(self.renderer.gaussians._xyz.shape[0]/2):,:]
                # n = pc1.size(0)
                # m = pc3.size(0)
                # # Expand pc1 to (n, 1, 3) and pc2 to (1, m, 3) to compute pairwise distance
                # pc1_expanded = pc1.unsqueeze(1).expand(n, m, 3)
                # pc3_expanded = pc3.unsqueeze(0).expand(n, m, 3)
                # # Compute squared distances (n, m)
                # distances_squared = torch.sum((pc1_expanded - pc3_expanded) ** 2, dim=2)
                # # Find the closest points and sum their distances
                # distances_pc1_to_pc3 = torch.min(distances_squared, dim=1)[0]  # Closest points in pc2 for each in pc1
                # distances_pc3_to_pc1 = torch.min(distances_squared, dim=0)[0]  # Closest points in pc1 for each in pc2
                # # Calculate Chamfer Distance
                # chamfer_dist_repulsive = -100 * torch.mean(distances_pc1_to_pc3) + torch.mean(distances_pc3_to_pc1)



                # optimize step
                loss= 10000*loss_inverse + 10000*loss_alpha


                #log metrics to wandb
                if self.opt.debug == False :
                    wandb.log({"loss": loss.item(), "loss_inverse":10000*loss_inverse, "loss_alpha":10000*loss_alpha,  "num_points":self.renderer.gaussians._xyz.shape[0]}, step = i*10000 + iii)

                    if iii%200==0:

                        # if iii%20==0:
                        #     wandb.log({"png_renders_whitebg":[wandb.Image(image) for image in images_images], \
                        #                "png_renders_selfbg":[wandb.Image(image) for image in images_render_views], \
                        #                "png_inverses":[wandb.Image(image) for image in images_imgs_reverse], \
                        #                "png_segments":[wandb.Image(image) for image in images_processed_tensor ], \
                        #                "plt_scalingstats": wandb.Image(f'./test_bg/scaling_{self.timestamp}.png') 
                        #                })


                        # if i%5==0:
                        from PIL import Image
                        image_rendersfront = []
                        for r in range(front_views_list[0].shape[0]):
                            input_img_torch_resized = front_views_list[b][r,:,:,:].permute(1, 2, 0)
                            input_img_np = input_img_torch_resized.detach().cpu().numpy()
                            input_img_np = (input_img_np * 255).astype(np.uint8)
                            #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversescore/check_view_{iii}_render.png')
                            image_render = Image.fromarray((input_img_np).astype(np.uint8))
                            image_rendersfront.append(image_render)


                        from PIL import Image
                        image_rendersfull = []
                        for r in range(render_views.shape[0]):
                            input_img_torch_resized = render_views[r,:,:,:].permute(1, 2, 0)
                            input_img_np = input_img_torch_resized.detach().cpu().numpy()
                            input_img_np = (input_img_np * 255).astype(np.uint8)
                            #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversescore/check_view_{iii}_render.png')
                            image_render = Image.fromarray((input_img_np).astype(np.uint8))
                            image_rendersfull.append(image_render)


                        from PIL import Image
                        image_inverses = []
                        for r in range(imgs_reverse.shape[0]):
                            input_img_torch_resized = imgs_reverse[r,:,:,:].permute(1, 2, 0)
                            input_img_np = input_img_torch_resized.detach().cpu().numpy()
                            input_img_np = (input_img_np * 255).astype(np.uint8)
                            #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversescore/check_view_{iii}_render.png')
                            image_render = Image.fromarray((input_img_np).astype(np.uint8))
                            image_inverses.append(image_render)


                        wandb.log({"image_rendersfront": [wandb.Image(image) for image in image_rendersfront] , \
                                   "image_rendersfull": [wandb.Image(image) for image in image_rendersfull] , \
                                   "image_inverse": [wandb.Image(image) for image in image_inverses], \
                                   "image_beforeinverse":[wandb.Image(image) for image in images_render_views], \
                                   }, step = i*10000 + iii)


                        save_dir = os.path.join( self.opt.case_dir, 'logs_ply' )  
                        os.makedirs(save_dir, exist_ok = True)
                        ply_path = os.path.join( save_dir, f'ply_{self.opt.timestamp}_{i}_{i*10000 + iii}.ply' )
                        self.renderer.gaussians.save_ply( ply_path )
                        print("uploading ply")
                        #wandb.save(ply_path)

                        try:
                            wandb.save(ply_path)
                        except:
                            print(f'saving ply_{self.opt.timestamp}_{i}_{i*10000 + iii}.ply fails')
                            pass 



                #breakpoint()
                #loss = 0*loss_mse + loss_inverse + chamfer_dist + chamfer_dist_repulsive

                loss.backward()
                # breakpoint()
                # name_list = []
                # for name,var in self.renderer.gaussians._deformation.deformation_net.grid.named_parameters():
                #     if var.grad is not None and torch.norm(var.grad) > 0:
                #         name_list.append(name)
                # print(name_list)
                # breakpoint()

                self.optimizer.step()

                self.optimizer.zero_grad()

                # if iii>1 and self.renderer.gaussians._xyz.shape[0]<25000:
                #     # densify and prune
                #     #if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                #     if True:
                #         viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                #         self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #         self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #         #if self.step % self.opt.densification_interval == 0:

                #             # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                #             # self.opt.densify_grad_threshold = 0.05
                #         if iii%10 == 0:
                #             print("densify")
                #             self.renderer.gaussians.densify_and_prune(max_grad = 0.05, min_opacity=0.01, extent=0.5, max_screen_size=1)
                        
                #         # #if self.step % self.opt.opacity_reset_interval == 0:
                #         # if iii == 400:
                #         #     print('reset opacity')
                #         #     self.renderer.gaussians.reset_opacity()

            

                #     # Ensure both tensors are on the same device, e.g., CUDA
                #     xyz = self.renderer.gaussians._xyz.to('cuda:0')
                #     xyz_init = self.xyz_initialize.to('cuda:0')
                #     # Compute pairwise squared distances (for efficiency reasons, work with squared distances)
                #     diff = xyz[:, None, :] - xyz_init[None, :, :]
                #     distances_squared = torch.sum(diff ** 2, dim=-1)
                #     # Find the minimum squared distance for each point in xyz to points in xyz_init
                #     min_distances_squared = torch.min(distances_squared, dim=1)[0]
                #     # Define the squared distance threshold (to avoid square root operations for efficiency)
                #     threshold_squared = 0.2 ** 2
                #     # Generate the boolean tensor based on the threshold
                #     thresholded_bools = min_distances_squared < threshold_squared
                #     # Ensure the boolean tensor is on the correct device and of the correct type
                #     thresholded_bools = thresholded_bools.to(dtype=torch.bool).to('cuda:0')

                #     thresholded_bools[int(self.renderer.gaussians._xyz.shape[0]/2):] = True
                #     # Convert to a PyTorch tensor and transfer to CUDA device
                #     semantic_mask = torch.tensor(thresholded_bools, dtype=torch.bool).to('cuda:0') # semantic_mask here is True for object inside and False for object outside, note that in prune func, it is reverse
                #     self.renderer.gaussians.densify_and_prune(max_grad = 0.05, min_opacity=0.01, extent=0.5, max_screen_size=1, semantic_mask = semantic_mask)

            self.renderer_last = copy.deepcopy(self.renderer)




            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            self.need_update = True


            # # #####################################################################################
            # # ########## GS pruning by semantic mask 
            # # #####################################################################################
            # from PIL import Image
            # os.makedirs( f'./cache_sam/cache_{self.opt.timestamp}', exist_ok = True )


            # # Ensure both tensors are on the same device, e.g., CUDA
            # xyz = self.renderer.gaussians._xyz.to('cuda:0')
            # xyz_init = self.xyz_initialize.to('cuda:0')
            # # Compute pairwise squared distances (for efficiency reasons, work with squared distances)
            # diff = xyz[:, None, :] - xyz_init[None, :, :]
            # distances_squared = torch.sum(diff ** 2, dim=-1)
            # # Find the minimum squared distance for each point in xyz to points in xyz_init
            # min_distances_squared = torch.min(distances_squared, dim=1)[0]
            # # Define the squared distance threshold (to avoid square root operations for efficiency)
            # threshold_squared = 0.1 ** 2
            # # Generate the boolean tensor based on the threshold
            # thresholded_bools = min_distances_squared < threshold_squared
            # # Ensure the boolean tensor is on the correct device and of the correct type
            # thresholded_bools = thresholded_bools.to(dtype=torch.bool).to('cuda:0')

            # # Convert to a PyTorch tensor and transfer to CUDA device
            # semantic_mask = torch.tensor(thresholded_bools, dtype=torch.bool).to('cuda:0') # semantic_mask here is True for object inside and False for object outside, note that in prune func, it is reverse
            # self.renderer.gaussians.densify_and_prune(max_grad = 0.05, min_opacity=0.01, extent=0.5, max_screen_size=1, semantic_mask = semantic_mask)


            # # save and upload pruned GS
            # save_dir = './logs_ply'
            # os.makedirs(save_dir, exist_ok = True)
            # ply_path = os.path.join( save_dir, f'ply_{self.opt.timestamp}_{i}_aftersemanticpruning_debug.ply' )
            # self.renderer.gaussians.save_ply( ply_path )
            # print(f'this experiment starts from {self.opt.timestamp}')
            # if self.opt.debug == False :
            #     #wandb.log({"png_segments": [wandb.Image(image) for image in png_segments]}, step = i*10000 + iii + 1)
            #     wandb.save(ply_path)



    
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


                self.train_step(i)
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

    opt.data_dir = '/mnt/vita-nas/zy3724/4Dprojects'

    # auto find mesh from stage 1
    #opt.load = os.path.join(opt.outdir, opt.save_path + '_model.ply')

    # opt.load = os.path.join(opt.outdir, 'flower' + '_model.ply')w
    # opt.savename = './visuals/flower_offset_balance'
    # opt.input = './data/set/flower'
    # opt.save_path = 'flower'

    opt.debug = False
    os.makedirs( os.path.join( opt.data_dir, 'demo_output' , opt.run_name ), exist_ok = True )
    os.makedirs( os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'ply_3d' ), exist_ok = True )
    os.makedirs( os.path.join( opt.data_dir, '../demo_output' , opt.run_name, 'pth_4d' ), exist_ok = True )

    os.makedirs( os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'backgrounds', 'prompt' ), exist_ok = True )
    os.makedirs( os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'backgrounds', 'generates' ), exist_ok = True )

    if not os.path.exists( os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'backgrounds', 'prompt', 'seed.txt' ) ):
        with open( os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'backgrounds', 'prompt', 'seed.txt' ) , 'w') as file:
            file.write(str( 0 ))




    opt.load = os.path.join( opt.data_dir, 'demo_output' , opt.run_name, 'ply_3d', 'model.ply' )


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
    opt.case_dir = os.path.join( opt.data_dir, 'demo_output' , opt.run_name)

    opt.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f'starting experiment {opt.timestamp}')

    #opt.load = None
    opt.savename = None
    opt.input = None
    opt.save_path = None
    opt.batch_size = 24
    opt.prompt = 'running cat'
    opt.negative_prompt = 'negative'
    opt.start_vsd = 2000

    script_path = os.path.abspath(__file__)

    if opt.debug == True:
        pass
    if not opt.debug:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="paper_demo",
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