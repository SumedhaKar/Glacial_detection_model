import cv2
import torch
import numpy as np
from torchvision import transforms
from model_loader import load_model

# Load image
image_path = "data/images/lake1.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_resized = cv2.resize(image, (256, 256)) / 255.0

# Prepare tensor
transform = transforms.ToTensor()
input_tensor = transform(image_resized).unsqueeze(0).float()

# Predict
model = load_model()
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.sigmoid(output).squeeze().numpy()

# Threshold
binary_mask = (pred_mask > 0.5).astype(np.uint8) * 255
cv2.imwrite("outputs/predicted_mask.png", binary_mask)
print("✅ Mask saved to outputs/predicted_mask.png")
