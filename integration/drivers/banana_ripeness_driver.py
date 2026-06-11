import cv2
from ultralytics import YOLO
from pathlib import Path

# ----------------------------
# banana_ripeness_driver.py
# ----------------------------
from pathlib import Path
import os

_PROJECT_DIR = Path(__file__).resolve().parent.parent
ripeness_model_path = str(_PROJECT_DIR.joinpath("models", "banana ripeness classification-2.pt"))

_YOLO_MODEL = None
ripeness_class_names = []

def get_ripeness_model():
    global _YOLO_MODEL, ripeness_class_names
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(ripeness_model_path)
        # Try to move the model to CUDA for faster inference, fallback to CPU
        try:
            _YOLO_MODEL.to("cuda:0")
            try:
                _YOLO_MODEL.model.half()  # optional fp16
            except Exception:
                pass
        except Exception:
            print("Warning: could not move ripeness model to CUDA. Running on CPU.")
        ripeness_class_names = list(_YOLO_MODEL.names.values())
        print("Ripeness model classes:", ripeness_class_names)
    return _YOLO_MODEL

class_info = {
    "freshunripe": {"days": "0-2 days", "ready": "Not yet (very early)"},
    "unripe":      {"days": "2-4 days", "ready": "Not yet (still green)"},
    "freshripe":   {"days": "4-6 days", "ready": "Perfectly ripe"},
    "ripe":        {"days": "6-8 days", "ready": "Ready to eat"},
    "overripe":    {"days": "8-10 days", "ready": "Very soft, use for baking"},
    "rotten":      {"days": "10+ days", "ready": "Do not eat"},
}

def run_banana_ripeness(image_path: str):
    """
    Returns (pred_class, info, vis_img)
    vis_img = image with YOLO's boxes/labels drawn (or original if none).
    """
    if not os.path.exists(ripeness_model_path):
        raise FileNotFoundError(f"Model weights not found at: {ripeness_model_path}")

    model = get_ripeness_model()
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.10,
        iou=0.5,
        device=0,      # use GPU 0 for inference (falls back if not available)
        verbose=False
    )
    r = results[0]

    pred_class = None
    conf = 0.0

    # === classification model ===
    if getattr(r, "probs", None) is not None:
        probs = r.probs
        top1_idx = int(probs.top1)
        conf = float(probs.top1conf)
        pred_class = ripeness_class_names[top1_idx]
        vis_img = r.plot()  # YOLO renders label on image

    # === detection model ===
    else:
        boxes = getattr(r, "boxes", None)
        num_boxes = 0 if boxes is None else len(boxes)
        print(f"[ripeness] detections: {num_boxes}")

        if boxes is None or len(boxes) == 0:
            # Fallback: assume unripe
            pred_class = "unripe"
            info = class_info[pred_class]
            vis_img = cv2.imread(image_path)
            h, w = vis_img.shape[:2]
            cv2.rectangle(vis_img, (10, 10), (w - 10, 70), (0, 0, 0), -1)
            cv2.putText(vis_img, f"Fallback: {pred_class}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2, cv2.LINE_AA)
            return pred_class, info, vis_img

        confs = boxes.conf.cpu().numpy()
        clss  = boxes.cls.cpu().numpy().astype(int)

        best_i = confs.argmax()
        cls_idx = int(clss[best_i])
        conf = float(confs[best_i])

        pred_class = ripeness_class_names[cls_idx]
        vis_img = r.plot()

    info = class_info.get(
        pred_class,
        {"days": "N/A", "ready": f"Predicted: {pred_class} ({conf:.2f})"}
    )

    return pred_class, info, vis_img
