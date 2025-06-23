import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from utils import load_model

st.title("Glacial Lake Segmentation")

uploaded_file = st.file_uploader("Upload a satellite image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    transform = A.Compose([
        A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
        ToTensorV2()
    ])

    augmented = transform(image=image_np)
    input_tensor = augmented['image'].unsqueeze(0).to(config.DEVICE)

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    model = load_model(model, "model.pth", config.DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    st.image((pred_mask > 0.5) * 255, caption="Predicted Mask", use_column_width=True)
    st.image(image_np, caption="Original Image", use_column_width=True)
