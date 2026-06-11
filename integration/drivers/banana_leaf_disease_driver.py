# drivers/banana_leaf_disease_driver.py
import os
import tempfile
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Edit these paths to point to your actual model file if different
_THIS_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(_THIS_DIR.parent.joinpath("models", "banana_leaf_disease.pth"))
MODEL_NAME = "efficientnet_b0"
IMG_SIZE = 224

def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Black Sigatoka Disease",
    "Bract Mosaic Virus Disease",
    "Healthy Leaf",
    "Insect Pest Disease",
    "Moko Disease",
    "Panama Disease",
    "Yellow Sigatoka Disease"
]

def make_transform(img_size):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])

def load_model(model_path=MODEL_PATH, model_name=MODEL_NAME, num_classes=len(CLASS_NAMES), device=None):
    # Lazy-load model and cache it on the module
    global _MODEL
    if "_MODEL" in globals():
        return _MODEL

    import torch
    import timm

    if device is None:
        device = get_device()

    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(model_path, map_location=device)

    # unwrap possible wrapper keys
    if isinstance(state, dict):
        for key in ("state_dict","model_state_dict","model_state","weights"):
            if key in state:
                state = state[key]
                break

    # remove "module." prefix if present
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    _MODEL = model
    return _MODEL

def predict(model, transform, image_path, device=None):
    import torch
    import torch.nn.functional as F

    if device is None:
        device = get_device()

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    data = transform(image=arr)
    tensor = data["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)
        probs = F.softmax(out, dim=1)[0].cpu().numpy()

    top_idx = int(probs.argmax())
    top_prob = float(probs[top_idx])
    return top_idx, top_prob

def annotate(image_path, label, prob):
    img = cv2.imread(image_path)
    text = f"{label} ({prob*100:.1f}%)"
    # Put text at top-left and scale depending on image width
    scale = max(0.6, img.shape[1] / 1000.0)
    cv2.putText(img, text, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), int(2*scale))
    return img

def run_banana_leaf_disease(image_path):
    """
    Main driver function for Streamlit front-end.
    Input: path to image file
    Returns: (pred_class_name, prob_float, annotated_image_bgr_numpy)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = load_model()
    transform = make_transform(IMG_SIZE)

    idx, prob = predict(model, transform, image_path, get_device())
    class_name = CLASS_NAMES[idx]

    annotated = annotate(image_path, class_name, prob)
    return class_name, float(prob), annotated
