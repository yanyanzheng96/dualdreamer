from PIL import Image
import os

def crop_and_resize_gif(input_path, output_path, crop_box, indices, concatenated_output_path):
    # Create the output directory for frames
    frames_dir = output_path.replace('.gif', '_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Open the original gif
    with Image.open(input_path) as img:
        frames = []
        selected_frames = []
        # Iterate through each frame
        for frame in range(0, img.n_frames):
            img.seek(frame)
            # Crop the frame
            cropped_frame = img.crop(crop_box)
            # Resize the frame
            resized_frame = cropped_frame.resize((512, 512))
            frames.append(resized_frame)
            
            # Save each frame as an image
            frame_path = os.path.join(frames_dir, f'frame_{frame}.png')
            resized_frame.save(frame_path)
            
            # Select frames based on the given indices
            if frame in indices:
                selected_frames.append(resized_frame)
        
        # Save the frames as a new gif
        frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=0)
    
    # Concatenate the selected frames into one image
    if selected_frames:
        total_width = (200 * len(selected_frames)) + (2 * (len(selected_frames) - 1))
        concatenated_image = Image.new('RGB', (total_width, 200), (255, 255, 255))
        x_offset = 0
        for frame in selected_frames:
            resized_for_concat = frame.resize((200, 200))
            concatenated_image.paste(resized_for_concat, (x_offset, 0))
            x_offset += 202  # 200 for the image width + 2 for the gap
        
        concatenated_image.save(concatenated_output_path)




### parameter help 
### Crop box (left, upper, right, lower) ###


################ flower_pot 
#crop_box = (50, 50, 400, 400)  # for flower_pot
# input_gif = '/home/zy3724/4Dprojects/demo_output/flower_pot/final_visuals/inversionedit4dwithpth_full.gif'
# output_gif = '/home/zy3724/4Dprojects/demo_output/flower_pot/final_visuals/inversionedit4dwithpth_full_crop.gif'




################ case_test
crop_box = (0, 0, 500, 500)  # for case_test
input_gif = '/home/zy3724/4Dprojects/demo_output/case_test/final_visuals/edit3dwithpth_full.gif'
output_gif = '/home/zy3724/4Dprojects/demo_output/case_test/final_visuals/edit3dwithpth_full.gif'
concatenated_output_png = '/home/zy3724/4Dprojects/demo_output/case_test/final_visuals/edit3dwithpth_full.png'

indices = [0, 2, 4, 6, 8, 10, 12, 14]


crop_and_resize_gif(input_gif, output_gif, crop_box, indices, concatenated_output_png)
