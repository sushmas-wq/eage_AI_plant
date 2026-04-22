import os
import logging
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkptc")


class CropClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = models.mobilenet_v3_large(weights=None)
        in_f = self.net.classifier[3].in_features
        self.net.classifier[3] = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.net(x)


class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = models.mobilenet_v3_large(weights=None)
        in_f = self.net.classifier[3].in_features
        self.net.classifier[3] = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.net(x)


def load_crop_model():
    path = os.path.join(CHECKPOINT_DIR, "best_crop_model.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Crop model not found at {path}")
    ckpt  = torch.load(path, map_location=DEVICE)
    model = CropClassifier(len(ckpt["classes"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Crop model loaded — classes: %s", ckpt["classes"])
    return model, ckpt["classes"]


def load_disease_model(crop: str):
    path = os.path.join(CHECKPOINT_DIR, f"best_disease_{crop}.pth")
    if not os.path.exists(path):
        logger.warning("Disease model not found for crop '%s' at %s", crop, path)
        return None, None
    ckpt  = torch.load(path, map_location=DEVICE)
    model = DiseaseClassifier(len(ckpt["classes"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info("Disease model loaded for '%s' — classes: %s", crop, ckpt["classes"])
    return model, ckpt["classes"]