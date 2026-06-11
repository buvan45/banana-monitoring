import os
from pathlib import Path

# === Paths ===
_THIS_DIR = Path(__file__).resolve().parent
variety_model_path = str(_THIS_DIR.parent.joinpath("models", "banana_variety.pth"))
variety_class_path = str(_THIS_DIR.parent.joinpath("models", "banana_variety_class_names.pth"))

model = None
variety_class_names = None
preprocess = None

def load_variety_model():
    global model, variety_class_names, preprocess
    if model is not None:
        return model

    import torch
    import torch.nn as nn
    from torchvision import transforms, models

    device = torch.device("cpu")
    if os.path.exists(variety_model_path) and os.path.exists(variety_class_path):
        variety_class_names = torch.load(variety_class_path)
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(variety_class_names))
        model.load_state_dict(torch.load(variety_model_path, map_location=device))
        model = model.to(device)
        model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return model

def get_model():
    load_variety_model()
    return model

def run_banana_variety(image_path):
    load_variety_model()
    if model is None:
        return "Model not found", None
    
    import torch
    from PIL import Image

    device = torch.device("cpu")
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, pred_idx = torch.max(outputs, 1)
        pred_class = variety_class_names[pred_idx.item()]
    return pred_class, None
