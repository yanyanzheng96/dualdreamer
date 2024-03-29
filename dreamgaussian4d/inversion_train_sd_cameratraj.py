import os
import cv2
import time
import tqdm
import numpy as np
import itertools

from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.lora import LoRALinearLayer

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
from PIL import Image
import imageio

from scipy.spatial import distance
import numpy as np

import random

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import datetime


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


        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
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
        

        if self.opt.prompt is not None: # None
            self.prompt = self.opt.prompt
        if self.opt.negative_prompt is not None: # None
            self.nagative_prompt = self.opt.negative_prompt


        # override prompt from cmdline
        # load predefined point cloud initialization
        self.selfdefined_shapes = ['hemisphere']
        if self.opt.load in self.selfdefined_shapes:
            self.renderer.initialize_selfdefined(input = self.opt.load, num_pts=self.opt.num_pts)
        else:
            # override if provide a checkpoint
            if self.opt.load is not None: # not None
                self.renderer.initialize(self.opt.load)  
                # self.renderer.gaussians.load_model(opt.outdir, opt.save_path)             
            else:
                # initialize gaussians to a blob
                self.renderer.initialize(num_pts=self.opt.num_pts)


        # set camera history 
        self.opt.radius = 1
        self.history_poses = []
        self.history_targets = []
        self.history_campositions = []
        pose_fix = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.history_poses.append(pose_fix)
        self.history_targets = np.zeros([1, 3], dtype=np.float32)
        self.history_radii = np.zeros([1], dtype=np.float32)
        self.history_vers = np.zeros([0], dtype=np.float32)
        self.history_hors = np.zeros([0], dtype=np.float32)
        self.history_campositions = np.array( [[0, 0, self.opt.radius]] )

        # # vector check for camera trajectories
        # for i in range(50):
        #     self.random_camera() 
        #     breakpoint()


        # seed function
        self.seed_everything()



    def random_camera(self, traj_name = 'ball', seed_num = 20000, prob_interval_list = [0, 0.3, 0.9, 1], K_closest_point_num = 3, prob_interval_num = 10):
        # load predefined camera trajectories 
        # here we should define a function to generate random camera pose with train_step i
        # the intuition is to slice the camera trajectory from a neighborhood to full set 

        if traj_name == 'ball':
            seed_num = seed_num  # Number of samples to generate

            ### generate seeds 
            # Initialize target for all seed_num instances
            target = np.zeros([seed_num, 3], dtype=np.float32)
            # Generate radius, ver (elevation), and hor (azimuth) for all instances
            radius = np.random.uniform(-0.5, 0.5, size=seed_num)
            ver = np.random.randint(-180, 180, size=seed_num)
            hor = np.random.randint(-180, 180, size=seed_num)
            # Convert to radians
            elevation = np.deg2rad(ver)
            azimuth = np.deg2rad(hor)
            # Compute x, y, and z for all instances
            x = radius * np.cos(elevation) * np.sin(azimuth)
            y = -radius * np.sin(elevation)
            z = radius * np.cos(elevation) * np.cos(azimuth)
            # Calculate campos for all instances
            campos = np.stack([x, y, z], axis=1) + target

            ### compare with the history list 
            # Compute pairwise distances between self.history_campositions and campos
            pairwise_distances = distance.cdist(self.history_campositions, campos)
            # Number of top K closest points you want to find
            K = K_closest_point_num
            # Find the indices of the top K closest points in campos for each point in self.history_campos
            top_k_indices = np.argsort(pairwise_distances, axis=1)[:, :K]
            # Convert the indices to a list of lists
            top_k_indices_list = top_k_indices.tolist()

            
            lenhis = len(top_k_indices_list) # short for length of history 
            selection = set()
            for i, p in enumerate(prob_interval_list[:-1]):
                p_next =  prob_interval_list[i+1]
                sublist = top_k_indices_list[int(p*lenhis) : int(p_next*lenhis)+1]

                # Merge the lists in the sublist using a set
                merged_set = set()
                for sublist_item in sublist:
                    merged_set.update(sublist_item)
                # Convert the set back to a list if needed
                merged_list = list(merged_set)
            
                # Randomly select prob_interval_num elements from the merged list
                random_selection = random.sample(merged_list, min(len(merged_list), prob_interval_num))
                random_selection = list(random_selection)
                selection.update(random_selection)

            last_selection = random_selection
            selection = list(selection)

            self.history_campositions = np.concatenate((self.history_campositions, campos[last_selection,:]), axis=0)
            self.history_targets = np.concatenate((self.history_targets, target[last_selection,:]), axis=0)
            self.history_radii = np.concatenate((self.history_radii, radius[last_selection]), axis=0)
            self.history_vers = np.concatenate((self.history_vers, ver[last_selection]), axis=0)
            self.history_hors = np.concatenate((self.history_hors, hor[last_selection]), axis=0)

            return_poses = []
            for s in selection:
                pose = orbit_camera(elevation = ver[s], azimuth = hor[s], radius = radius[s], is_degree = True, target = target[s,:])
                return_poses.append(pose)

            
            ### visualization sanity check
            fig = plt.figure(figsize=(12, 6))
            # Subplot for vectors
            ax1 = fig.add_subplot(121, projection='3d')
            # Get the number of vectors
            num_vectors = len(self.history_targets)
            # Create a colormap based on the index I
            colormap = plt.cm.get_cmap('viridis')
            # Plot the vectors
            for i in range(num_vectors):
                length = np.linalg.norm( self.history_campositions[i,:] - self.history_targets[i, :])
                color = 'red' if i == 0 else colormap(i / num_vectors)  # Red for the first vector, colormap for the rest

                if i == 0:
                    ax1.quiver(self.history_targets[i, 0], self.history_targets[i, 1], self.history_targets[i, 2],
                            self.history_campositions[i, 0], self.history_campositions[i, 1], self.history_campositions[i, 2],
                            color='red', length=0.05*length, normalize=True)
                if i > 0 :
                    ax1.quiver(self.history_targets[i, 0], self.history_targets[i, 1], self.history_targets[i, 2],
                            self.history_campositions[i, 0], self.history_campositions[i, 1], self.history_campositions[i, 2],
                            color=color, length=0.05*length, normalize=True)
            # Set labels and title for the first subplot
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('History Vector Visualization')

            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=colormap)
            sm.set_array([])
            plt.colorbar(sm, ax=ax1, label='Index (I)')

            # Set aspect ratio for the first subplot
            ax1.set_box_aspect([1, 1, 1])

            # Subplot for start and end points
            ax2 = fig.add_subplot(122, projection='3d')

            # Get the number of vectors
            num_vectors = len(last_selection)
            # Plot the vectors
            for i in range(num_vectors):
                length = np.linalg.norm( campos[last_selection,:][i,:] - target[last_selection,:][i, :])

                ax2.quiver(target[last_selection,:][i, 0], target[last_selection,:][i, 1], target[last_selection,:][i, 2],
                        campos[last_selection,:][i, 0], campos[last_selection,:][i, 1], campos[last_selection,:][i, 2],
                        color='darkgreen', length=0.05*length, normalize=True)

            # Set labels and title for the second subplot
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title('sampled new vectors in this batch')
            # Set aspect ratio for the second subplot
            ax2.set_box_aspect([1, 1, 1])

            # Adjust layout
            plt.tight_layout()
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Save the plot as an image
            plt.savefig(f'./test_cam/pose_sanity_{timestamp}.png')
            # Close the plot to free memory
            plt.close()

            img = Image.open(f'./test_cam/pose_sanity_{timestamp}.png')

            if self.opt.debug == False :
                wandb.log({"pose_sanity":wandb.Image(img)})


        return return_poses 






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
        # self.renderer.gaussians._opacity.requires_grad = False
        # self.renderer.gaussians._scaling.required_grad = False
        # self.renderer.gaussians._rotation.requires_grad = False



        ### yan_vsd to add lora parameters to optimizer
        print(f"[INFO] loading SD pipeline in GUI initialization for lora preparation...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype = torch.float32
        )
        print(f"finished loading!")




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


            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device, t_range=[0.02, self.opt.t_max])
            print(f"[INFO] loaded zero123!")

            print(f"[INFO] loading SD...")
            from guidance.sd_utils import StableDiffusion
            self.guidance_sd = StableDiffusion(self.device)
            self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            print(f"[INFO] loaded SD!")

            # this block should happens twice, this is first time
            if True:
                c_list, v_list = [], []
                for _ in range(self.opt.batch_size):
                    for input_img_torch in self.input_img_torch_list:
                        c, v = self.guidance_zero123.get_img_embeds(input_img_torch)  #input_img_torch torch.Size([1, 3, 256, 256])
                        c_list.append(c)
                        v_list.append(v)
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]


            # prepare inverse 
            self.back_ratio_list = [0.0]
            self.back_gt_list = self.guidance_zero123.get_inverse((self.input_img_torch).repeat(self.opt.batch_size,1,1,1), [self.opt.elevation]*self.opt.batch_size, [0]*self.opt.batch_size, [self.opt.radius]*self.opt.batch_size, back_ratio_list = self.back_ratio_list)
            
            #self.back_gt_list = self.guidance_zero123.get_inverse_loop((self.input_img_torch).repeat(self.opt.batch_size,1,1,1), [self.opt.elevation]*self.opt.batch_size, [0]*self.opt.batch_size, [self.opt.radius]*self.opt.batch_size, back_ratio_list = self.back_ratio_list)


            self.back_embeddings_list = []
            for img in self.back_gt_list:
                img = img.unsqueeze(0)
                c_list, v_list = [], []
                for _ in range(self.opt.batch_size):
                    c, v = self.guidance_zero123.get_img_embeds(img)  #input_img_torch torch.Size([1, 3, 256, 256])
                    c_list.append(c)
                    v_list.append(v)
                self.back_embeddings_list.append([torch.cat(c_list, 0), torch.cat(v_list, 0)])

            # this block should happens twice, this is second time
            if True:
                c_list, v_list = [], []
                for _ in range(self.opt.batch_size):
                    for input_img_torch in self.input_img_torch_list:
                        c, v = self.guidance_zero123.get_img_embeds(input_img_torch)  #input_img_torch torch.Size([1, 3, 256, 256])
                        c_list.append(c)
                        v_list.append(v)
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]


            
            if self.enable_svd:
                self.guidance_svd.get_img_embeds(self.input_img)






            # if self.enable_zero123:
            #     dict_embeddings = {}
            #     for casename in self.casenames:

            #         input_img_list = self.dict_for_image_list[casename]

            #         input_img_torch_list = [torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device) for input_img in input_img_list]
            #         input_img_torch_list = [F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False) for input_img_torch in input_img_torch_list]
        
            #         c_list, v_list = [], []
            #         for _ in range(16):
            #             for input_img_torch in input_img_torch_list:
            #                 c, v = self.guidance_zero123.get_img_embeds(input_img_torch)
            #                 c_list.append(c)
            #                 v_list.append(v)
            #         dict_embeddings[casename] = [torch.cat(c_list, 0), torch.cat(v_list, 0)]
                    
            #     self.guidance_zero123.dict_embeddings = dict_embeddings




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


        iii = i % len(self.casenames)
        casename = self.casenames[iii]
        print('----------------------------------------------------------')
        print(f'training iteration {i}, switch to case {self.casenames[iii]}, index {iii}')


        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        for _ in range(self.train_steps): # 1

            self.step += 1 # self.step starts from 0
            #step_ratio = min(1, self.step / self.opt.iters) # 1, step / 500
            step_ratio = min(1, self.step / 500) # 1, step / 500

            # update lr
            self.renderer.gaussians.update_learning_rate(self.step)



            ### Novel view ------------------------------------------------------------------------------------
            
            render_resolution = 512

            if i == 0:
                pose_fix = orbit_camera(self.opt.elevation, 0, self.opt.radius)
                poses = [pose_fix]
                
            if i > 0:
                poses = []
                while len(poses) < self.opt.batch_size:
                    return_poses = self.random_camera(traj_name = 'ball', seed_num = 20000, prob_interval_list = [0, 0.3, 0.9, 1], K_closest_point_num = 3, prob_interval_num = 10)
                    poses = poses + return_poses
                poses = poses[0:self.opt.batch_size]
                
        
            if True:
                render_views_list = []
                images = []
                for _ in range(len(poses)):
                    cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                    bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                    out = self.renderer.render(cur_cam, bg_color=bg_color)

                    image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                    images.append(image)
                images = torch.cat(images, dim=0)
                render_views_list.append(images)


                image_inverses = []
                imgs_reverse = []
                render_views = torch.clone(render_views_list[0])   
                # Calculate the total number of samples
                total_samples = render_views.size(0)
                with torch.no_grad(): 
                    while total_samples > 0:
                        batch_size = min(total_samples, 6)
                        batch = render_views[:batch_size]
                        render_views = render_views[batch_size:]
                        total_samples -= batch_size
                        print("Batch shape:", batch.shape)

                        if i == 0:
                            imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(batch, prompts = "christmas room", negative_prompts = '', back_ratio_list = [1])
                        if i > 0:
                            imgs_reverse_batch, loss_inverse, image_inverses_batch = self.guidance_sd.get_inverse_loop(batch, prompts = "christmas room", negative_prompts = '', back_ratio_list = [0.15])

                        image_inverses = image_inverses + image_inverses_batch
                        imgs_reverse.append(imgs_reverse_batch)

                    imgs_reverse = torch.cat(imgs_reverse, dim=0)
                    image_inverses = image_inverses
            
            

            if i == 0:
                inner_loop = 1000
            if i > 0:
                inner_loop = 20
            for iii in range(inner_loop):
                print(iii)
                self.step += 1

                ### known view ------------------------------------------------------------------------------------
                #if self.input_img_torch is not None:
                loss_mse = 0


                ### novel view ------------------------------------------------------------------------------------
                render_views_list = []
                if True:
                    images = []
                    for _ in range(len(poses)):
                        cur_cam = MiniCam(poses[_], render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)
                        bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                        out = self.renderer.render(cur_cam, bg_color=bg_color)

                        image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)
                    images = torch.cat(images, dim=0)

                    render_views_list.append(images)
                
                loss_inverse = F.mse_loss(render_views_list[0], imgs_reverse.to(torch.float32))


                # if i%5==0:
                from PIL import Image
                image_renders = []
                for r in range(render_views_list[0].shape[0]):
                    input_img_torch_resized = render_views_list[0][r,:,:,:].permute(1, 2, 0)
                    input_img_np = input_img_torch_resized.detach().cpu().numpy()
                    input_img_np = (input_img_np * 255).astype(np.uint8)
                    #Image.fromarray((input_img_np).astype(np.uint8)).save(f'./test_inversescore/check_view_{iii}_render.png')
                    image_render = Image.fromarray((input_img_np).astype(np.uint8))
                    image_renders.append(image_render)




                # optimize step
                # breakpoint()
                loss_inverse= 10000*loss_inverse

                loss_mse = 100000*loss_mse

                #log metrics to wandb
                if self.opt.debug == False :
                    wandb.log({"loss_mse": loss_mse, "loss_inverse":loss_inverse, "num_points":self.renderer.gaussians._xyz.shape[0]})

                    if iii%100==0:
                        wandb.log({"image_inverses":[wandb.Image(image) for image in image_inverses], "image_renders":[wandb.Image(image) for image in image_renders]})

                    if iii%100==0:
                        image_list =[]
                        from PIL import Image
                        from diffusers.utils import export_to_video, export_to_gif
                        nframes = 14 *5
                        hor = 180
                        delta_hor = 360 / nframes
                        time = 0
                        delta_time = 1
                        for _ in range(nframes):
                            pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
                            cur_cam = MiniCam(
                                pose,
                                512,
                                512,
                                self.cam.fovy,
                                self.cam.fovx,
                                self.cam.near,
                                self.cam.far,
                            )

                            outputs = self.renderer.render(cur_cam)

                            outt = outputs["image"].cpu().detach().numpy().astype(np.float32)
                            outt = np.transpose(outt, (1, 2, 0))
                            outt = Image.fromarray(np.uint8(outt*255))
                            image_list.append(outt)


                            time = (time + delta_time) % 14
                            hor = (hor+delta_hor) % 360

                        if self.opt.debug == False:
                            imageio.mimsave('result.gif', image_list, fps=5)  # Adjust fps as needed
                            wandb.log({"result_gif": wandb.Image('result.gif')})




                #breakpoint()
                if i == 0:
                    loss_inverse = loss_inverse * self.opt.batch_size
                
                loss = loss_inverse 
                
                loss.backward()


                self.optimizer.step()

                self.optimizer.zero_grad()

                if iii%1 == 0 and self.renderer.gaussians._xyz.shape[0]<20000:

                    # densify and prune
                    if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                        viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                        self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                        if self.step % self.opt.densification_interval == 0:
                            # size_threshold = 20 if self.step > self.opt.opacity_reset_interval else None
                            self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                        
                        if self.step % self.opt.opacity_reset_interval == 0:
                            self.renderer.gaussians.reset_opacity()



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




    def load_dir(self, dir):
        def extract_number_npy(filename):
            filename = filename.split('/')[-1]
            base, _ = filename.split('.')
            _, number = base.rsplit('_', 1)
            return int(number)

        def extract_number_png(filename):
            base, _ = filename.split('.')
            _, number = base.rsplit('_', 1)
            return int(number)

        dir_list = os.listdir(dir)
        dict_for_image_list = {}
        dict_for_mask_list = {}

        for casename in dir_list:
            temp_input_img_list ,temp_input_mask_list = [], [] #self.input_img_list, self.input_mask_list = [], []

            #print(glob.glob(os.path.join(dir, casename, 'image_*.npy')))
            image_npy_list = [f for f in glob.glob(os.path.join(dir, casename, 'image_*.npy'))]
            mask_npy_list = [f for f in glob.glob(os.path.join(dir, casename, 'mask_*.npy'))]

            if len(image_npy_list)>0 and len(image_npy_list)==len(mask_npy_list):
                image_npy_list = sorted(image_npy_list, key=extract_number_npy)
                image_npy_list = [f for f in image_npy_list]
                print(image_npy_list)
                for npy in image_npy_list:
                    input_img = np.load(npy)
                    temp_input_img_list.append(input_img)
                mask_npy_list = sorted(mask_npy_list, key=extract_number_npy)
                mask_npy_list = [f for f in mask_npy_list]    
                print(mask_npy_list)
                for npy in mask_npy_list:
                    input_mask = np.load(npy)
                    temp_input_mask_list.append(input_mask)

                dict_for_image_list[casename] = temp_input_img_list
                dict_for_mask_list[casename] = temp_input_mask_list

            else:
                file_list = [f for f in os.listdir( os.path.join(dir, casename) ) if not f.endswith('_rgba.png') and f.endswith('.png')]
                file_list = sorted(file_list, key=extract_number_png)
                path_list = [os.path.join(dir, casename, f) for f in file_list]

                for file in path_list:
                    # load image
                    print(f'[INFO] load image from {dir} / {casename} / {file }...')
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
                    np.save(file.rsplit('.', 1)[0] + '.npy', input_img)
                    np.save((file.rsplit('.', 1)[0] + '.npy').replace('image_', 'mask_'), input_mask)


                    temp_input_img_list.append(input_img)
                    temp_input_mask_list.append(input_mask)

                    dict_for_image_list[casename] = temp_input_img_list
                    dict_for_mask_list[casename] = temp_input_mask_list

        self.dict_for_image_list = dict_for_image_list
        self.dict_for_mask_list = dict_for_mask_list

        self.casenames = dir_list

        self.input_img_list = dict_for_image_list[self.casenames[0]]
        self.input_mask_list = dict_for_mask_list[self.casenames[0]]

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

                snapshots = [0,50,150,200,300,400,600,800,980, 1200,1950]
                snapshots = [c + s for c in range(len(self.casenames)) for s in snapshots]
                if i in snapshots:

                    image_list =[]
                    from PIL import Image
                    from diffusers.utils import export_to_video, export_to_gif
                    nframes = 14 *5
                    hor = 180
                    delta_hor = 360 / nframes
                    time = 0
                    delta_time = 1
                    for _ in range(nframes):
                        pose = orbit_camera(self.opt.elevation, hor-180, self.opt.radius)
                        cur_cam = MiniCam(
                            pose,
                            512,
                            512,
                            self.cam.fovy,
                            self.cam.fovx,
                            self.cam.near,
                            self.cam.far,
                        )

                        outputs = self.renderer.render(cur_cam)

                        out = outputs["image"].cpu().detach().numpy().astype(np.float32)
                        out = np.transpose(out, (1, 2, 0))
                        out = Image.fromarray(np.uint8(out*255))
                        image_list.append(out)

                        time = (time + delta_time) % 14
                        hor = (hor+delta_hor) % 360

                    savename = opt.savename
                    os.makedirs(savename, exist_ok=True)
                    export_to_gif(image_list, f'{savename}/{opt.save_path}_train_{i}.gif')

                    if self.opt.debug == False:
                        imageio.mimsave('result.gif', image_list, fps=5)  # Adjust fps as needed
                        wandb.log({"result_gif": wandb.Image('result.gif')})


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
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))


    # auto find mesh from stage 1
    #opt.load = os.path.join(opt.outdir, opt.save_path + '_model.ply')

    # opt.load = os.path.join(opt.outdir, 'flower' + '_model.ply')
    # opt.savename = './visuals/flower_offset_balance'
    # opt.input = './data/set/flower'
    # opt.save_path = 'flower'

    # opt.load = os.path.join(opt.outdir, 'fish' + '_model.ply')
    #opt.load = "sphere"

    opt.load = None

    opt.savename = './visuals/fishseparate'
    opt.input = './data/set/oneonefish'
    opt.save_path = 'fish'
    opt.batch_size = 24
    opt.prompt = 'swimming fish'
    opt.negative_prompt = 'negative'
    opt.start_vsd = 2000

    # opt.load = os.path.join(opt.outdir, 'cat' + '_model.ply')
    # opt.savename = './visuals/cat_offset_balance'
    # opt.input = './data/set/cat1'
    # opt.save_path = 'cat'


    if opt.debug == True:
        pass
    if not opt.debug:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="camgen_inversionscore",
            # track hyperparameters and run metadata
            config={
            "load": opt.load,
            "savename": opt.savename,
            "input": opt.input,
            "save_path": opt.save_path,
            "batch_size": opt.batch_size, 
            "prompt": opt.prompt,
            "start_vsd": opt.start_vsd,
            "idea_try": "blue"
            }
        )

        wandb.save('your_script.py')

    gui = GUI(opt)
    gui.train(opt.iters)