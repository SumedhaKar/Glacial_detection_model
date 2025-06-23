import torch

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def calculate_metrics(preds, masks):
    preds = (preds > 0.5).float()
    masks = masks.float()
    intersection = (preds * masks).sum()
    union = preds.sum() + masks.sum() - intersection
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (preds.sum() + masks.sum() + 1e-6)
    return iou.item(), dice.item()
