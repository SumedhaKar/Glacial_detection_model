import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from dataset import GlacierDataset
import config
from utils import save_model, calculate_metrics

train_transform = A.Compose([
    A.Resize(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
    ToTensorV2()
])

dataset = GlacierDataset("data/images", "data/masks", transform=train_transform)
dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
model.to(config.DEVICE)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LR)

for epoch in range(config.EPOCHS):
    model.train()
    total_loss = 0
    ious, dices = [], []

    for images, masks in dataloader:
        images = images.to(config.DEVICE)
        masks = masks.to(config.DEVICE)

        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            iou, dice = calculate_metrics(torch.sigmoid(preds), masks)
            ious.append(iou)
            dices.append(dice)

    print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Loss: {total_loss/len(dataloader):.4f} | IoU: {sum(ious)/len(ious):.4f} | Dice: {sum(dices)/len(dices):.4f}")

save_model(model, "model.pth")
