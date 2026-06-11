import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

from pathlib import Path

device = torch.device("cpu")

# === Load model ===
_PROJECT_DIR = Path(__file__).resolve().parent.parent
variety_model_path = str(_PROJECT_DIR.joinpath("models", "banana_variety.pth"))
variety_class_path = str(_PROJECT_DIR.joinpath("models", "banana_variety_class_names.pth"))

_MODEL = None
_CLASS_NAMES = None

def load_model():
    global _MODEL, _CLASS_NAMES
    if _MODEL is not None:
        return _MODEL, _CLASS_NAMES

    if not os.path.exists(variety_model_path) or not os.path.exists(variety_class_path):
        raise FileNotFoundError(f"Model weights or class names not found at: {variety_model_path} or {variety_class_path}")

    _CLASS_NAMES = torch.load(variety_class_path)
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(_CLASS_NAMES))
    model.load_state_dict(torch.load(variety_model_path, map_location=device))
    _MODEL = model.to(device)
    _MODEL.eval()
    return _MODEL, _CLASS_NAMES

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def run_banana_variety(image_path):
    model_obj, class_names = load_model()
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model_obj(input_tensor)
        _, pred_idx = torch.max(outputs, 1)
        pred_class = class_names[pred_idx.item()]
    return pred_class, None
