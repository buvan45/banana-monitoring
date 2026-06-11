import cv2
from pathlib import Path

# ----------------------------
# banana_ripeness_driver.py
# ----------------------------
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
ripeness_model_path = str(_THIS_DIR.parent.joinpath("models", "banana ripeness classification-2.pt"))

# ----------------------------
# LOAD MODEL
# ----------------------------
yolo_model = None
ripeness_class_names = None

def load_ripeness_model():
    global yolo_model, ripeness_class_names
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO(ripeness_model_path)
        try:
            yolo_model.to("cuda:0")
            try:
                yolo_model.model.half()  # optional fp16
            except Exception:
                pass
        except Exception:
            print("Warning: could not move ripeness model to CUDA. Running on CPU.")
        ripeness_class_names = list(yolo_model.names.values())
    return yolo_model

def get_model():
    return load_ripeness_model()

# ----------------------------
# CLASS INFO
# ----------------------------
class_info = {
    "freshunripe": {"days": "0-2 days", "ready": "Not yet (very early)"},
    "unripe":      {"days": "2-4 days", "ready": "Not yet (still green)"},
    "freshripe":   {"days": "4-6 days", "ready": "Perfectly ripe"},
    "ripe":        {"days": "6-8 days", "ready": "Ready to eat"},
    "overripe":    {"days": "8-10 days", "ready": "Very soft, use for baking"},
    "rotten":      {"days": "10+ days", "ready": "Do not eat"},
}

# ----------------------------
# CLASS-WISE COLORS (BGR)
# ----------------------------
CLASS_COLORS = {
    "freshunripe": (0, 120, 0),    # dark green
    "unripe":      (0, 180, 0),    # green
    "freshripe":   (0, 200, 200),  # yellow
    "ripe":        (0, 165, 255),  # orange
    "overripe":    (0, 0, 255),    # red
    "rotten":      (0, 0, 120),    # dark red
}
DEFAULT_COLOR = (200, 200, 200)

# ----------------------------
# DRAWING CONFIG
# ----------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75
TEXT_THICKNESS = 1
BOX_THICKNESS = 2
LABEL_BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)   # WHITE TEXT

# ----------------------------
# RIPENESS FUNCTION
# ----------------------------
def run_banana_ripeness(image_path: str):
    """
    Returns (pred_class, info, vis_img)
    vis_img = image with boxes + class-wise colors drawn
    """
    model = load_ripeness_model()
    results = model.predict(
        source=image_path,
        imgsz=640,
        conf=0.10,
        iou=0.5,
        device=0,
        verbose=False
    )

    r = results[0]
    img = cv2.imread(image_path)

    pred_class = None
    best_conf = 0.0

    # ==========================
    # CLASSIFICATION MODEL
    # ==========================
    if getattr(r, "probs", None) is not None:
        probs = r.probs
        top1_idx = int(probs.top1)
        best_conf = float(probs.top1conf)
        pred_class = ripeness_class_names[top1_idx]

        color = CLASS_COLORS.get(pred_class, DEFAULT_COLOR)

        h, w = img.shape[:2]
        cv2.rectangle(img, (10, 10), (w - 10, 55), LABEL_BG_COLOR, -1)
        cv2.putText(
            img,
            f"{pred_class} ({best_conf:.2f})",
            (20, 40),
            FONT,
            0.6,
            TEXT_COLOR,
            2,
            cv2.LINE_AA
        )

        info = class_info.get(pred_class, {})
        return pred_class, info, img

    # ==========================
    # DETECTION MODEL
    # ==========================
    boxes = getattr(r, "boxes", None)
    if boxes is None or len(boxes) == 0:
        pred_class = "unripe"
        info = class_info[pred_class]
        return pred_class, info, img

    confs = boxes.conf.cpu().numpy()
    clss  = boxes.cls.cpu().numpy().astype(int)

    best_i = confs.argmax()
    pred_class = ripeness_class_names[int(clss[best_i])]
    best_conf = float(confs[best_i])

    for box in boxes:
        cls_id = int(box.cls.item())
        conf_i = float(box.conf.item())
        label = ripeness_class_names[cls_id]

        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # draw bounding box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            BOX_THICKNESS
        )

        # label text
        text = f"{label} ({conf_i:.2f})"
        (tw, th), _ = cv2.getTextSize(
            text,
            FONT,
            FONT_SCALE,
            TEXT_THICKNESS
        )

        # label background
        cv2.rectangle(
            img,
            (x1, y1 - th - 6),
            (x1 + tw + 4, y1),
            LABEL_BG_COLOR,
            -1
        )

        # draw text (WHITE)
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 3),
            FONT,
            FONT_SCALE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA
        )

    info = class_info.get(
        pred_class,
        {"days": "N/A", "ready": f"Predicted: {pred_class} ({best_conf:.2f})"}
    )

    return pred_class, info, img
