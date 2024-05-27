

import os
import PIL
from PIL import Image

dir = '../demo_data/sand_fish_robot/background' 

prompt_dir = os.path.join( dir, 'image_prompt' )

png_files = [f for f in os.listdir(prompt_dir) if f.endswith('.png')]
# Function to extract the index from the filename
def extract_index(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else None
# Sort the files by the extracted index
sorted_files = sorted(png_files, key=extract_index)
prompt_path = os.path.join( prompt_dir, sorted_files[0] )


image = Image.open( prompt_path )

for i in range(14):
    image.save( os.path.join( dir , 'images_generate', '' ) )

    





visual_dir = '../demo_data/test_visuals' 
os.makedirs( visual_dir, exist_ok = True )
for i in range(10):
    prompt = "top view of zoom in cloud"
    torch.manual_seed(i)

    image = pipe(prompt).images[0]
    image.save( os.path.join(visual_dir, f'./sdxl_prompt_{prompt}_{i}.png') )