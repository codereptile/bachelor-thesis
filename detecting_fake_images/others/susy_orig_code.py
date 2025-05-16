import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from torchvision import transforms

# Load the model
model = torch.jit.load("SuSy.pt")

# Load the image
# image = crop_to_square_and_resize(Image.open("flux_detection_reference_image_03.jpg"))
image = Image.open("../flux-inpaint/auto_comfyui/results/shop_0011_transparent_1_seed2_steps20.png")
# image = crop_to_square_and_resize(Image.open("../image_text_box/shops/shop_0005.jpg"))

# Set Parameters
top_k_patches = 5
patch_size = 224

# Get the image dimensions
width, height = image.size

# Calculate the number of patches
num_patches_x = width // patch_size
num_patches_y = height // patch_size

# Divide the image in patches
patches = np.zeros((num_patches_x * num_patches_y, patch_size, patch_size, 3), dtype=np.uint8)
for i in range(num_patches_x):
    for j in range(num_patches_y):
        x = i * patch_size
        y = j * patch_size
        patch = image.crop((x, y, x + patch_size, y + patch_size))
        patches[i * num_patches_y + j] = np.array(patch)

# Compute the most relevant patches (optional)
dissimilarity_scores = []
for patch in patches:
    transform_patch = transforms.Compose([transforms.PILToTensor(), transforms.Grayscale()])
    grayscale_patch = transform_patch(Image.fromarray(patch)).squeeze(0)
    glcm = graycomatrix(grayscale_patch, [5], [0], 256, symmetric=True, normed=True)
    dissimilarity_scores.append(graycoprops(glcm, "contrast")[0, 0])

# Sort patch indices by their dissimilarity score
sorted_indices = np.argsort(dissimilarity_scores)[::-1]

# Extract top k patches and convert them to tensor
top_patches = patches[sorted_indices[:top_k_patches]]
top_patches = torch.from_numpy(np.transpose(top_patches, (0, 3, 1, 2))) / 255.0

# Predict patches
model.eval()
with torch.no_grad():
    preds = model(top_patches)

# Print results
classes = ['authentic', 'dalle-3-images', 'diffusiondb', 'midjourney-images', 'midjourney_tti', 'realisticSDXL']
result = pd.DataFrame(preds.numpy(), columns=classes)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", None)
print(result)