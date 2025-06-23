import torch
import segmentation_models_pytorch as smp

def load_model(weights_path="best_model.pth"):
    model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    model.eval()
    return model
