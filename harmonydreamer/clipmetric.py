import torch
import clip
from PIL import Image



image_path = "paperdemo/lucidrobot.png"  # Replace with your image path

text_prompt = "robot in the flower field"  # Replace with your text prompt





# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load and preprocess the image
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# Define the text prompt
text = clip.tokenize([text_prompt]).to(device)

# Calculate the similarity
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (image_features @ text_features.T).item()

print(f"CLIP similarity: {similarity:.4f}")