import rasterio
import numpy as np
from PIL import Image
import os

# Input paths
RGB_TIF = RGB_TIF = RGB_TIF = r"C:\Users\Sumedha\Downloads\glacial_rgb.tif"


MASK_TIF = MASK_TIF =r"C:\Users\Sumedha\Downloads\glacial_water_mask.tif"


# Output directories
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/masks", exist_ok=True)

# --- Convert RGB image ---
with rasterio.open(RGB_TIF) as src:
    rgb = src.read([1, 2, 3])  # R, G, B
    rgb = np.transpose(rgb, (1, 2, 0))  # CHW -> HWC
    rgb = np.clip(rgb / 3000 * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    img = img.resize((256, 256))  # Resize for model input
    img.save("data/images/lake1.png")

# --- Convert binary mask ---
with rasterio.open(MASK_TIF) as src:
    mask = src.read(1)
    mask = (mask > 0).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize((256, 256))  # Resize to match
    mask_img.save("data/masks/lake1.png")

print("✅ Converted and saved as PNG.")
# Note: Ensure that the input TIF files are in the same directory as this script or provide the correct paths.
# This script assumes the TIF files are named "glacial_rgb.tif" and "glacial_water_mask.tif"