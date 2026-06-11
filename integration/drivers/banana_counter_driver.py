# integration/drivers/banana_counter_driver.py
"""
Banana counter driver (lazy-load friendly).

Public API kept:
- run_banana_counter(image_path, conf_thresh=0.25, show_ids=True) -> (banana_count, annotated_bgr_image)

Internals:
- get_banana_model() loads the YOLO model on demand. This prevents import-time crashes
  if the weights (.pt) are missing on a new machine.
"""

from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2

# ultralytics import moved to get_banana_model to support lazy loading

# Candidate model paths (add any other known paths here)
CANDIDATE_MODEL_PATHS = [
    Path(r"E:\SDP\integration\models\banana_counter.pt"),
    Path(__file__).resolve().parent.parent.joinpath("models", "banana_counter.pt"),  # project_root/integration/models
    Path(__file__).resolve().parent.joinpath("models", "banana_counter.pt"),         # integration/drivers/models
    Path.cwd().joinpath("models", "banana_counter.pt"),
    Path.cwd().joinpath("banana_counter.pt"),
]

# Module-level cache for the loaded model
_BANANA_MODEL = None  # type: Optional['YOLO']


def _find_model_path() -> Optional[Path]:
    for p in CANDIDATE_MODEL_PATHS:
        if p.exists():
            return p
    return None


def get_banana_model(weights_path: Optional[str] = None) -> 'YOLO':
    """
    Return a loaded YOLO model. Load lazily on first call.
    If weights_path not provided, searches CANDIDATE_MODEL_PATHS.
    Raises FileNotFoundError if weights not found.
    """
    global _BANANA_MODEL
    if _BANANA_MODEL is not None:
        return _BANANA_MODEL

    from ultralytics import YOLO

    # determine path
    if weights_path:
        p = Path(weights_path)
        if not p.exists():
            raise FileNotFoundError(f"Provided weights_path does not exist: {weights_path}")
        model_path = p
    else:
        model_path = _find_model_path()
        if model_path is None:
            tried = "\n".join(str(p) for p in CANDIDATE_MODEL_PATHS)
            raise FileNotFoundError(
                "Could not find banana_counter.pt weights. Tried these paths:\n" + tried +
                "\n\nPlace your weights in one of the above locations, or call get_banana_model(weights_path='...')"
            )

    # load YOLO model
    _BANANA_MODEL = YOLO(str(model_path))
    return _BANANA_MODEL


# --- Original-style runner kept intact ---
def run_banana_counter(image_path: str, conf_thresh: float = 0.25, show_ids: bool = True) -> Tuple[int, np.ndarray]:
    """
    Run inference on the image at image_path and return (count, annotated_bgr_image).
    Behaves the same as your original driver.
    """
    # Try loading the model (this may raise FileNotFoundError if weights not found)
    model = get_banana_model()

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model.predict(
        source=image_path,
        conf=conf_thresh,
        iou=0.5,
        imgsz=960,
        device="cpu"  # keep default CPU here; caller may choose GPU in realtime pipeline
    )

    r = results[0]
    banana_centers: List[tuple] = []

    # Use masks if available (segmentation model)
    if getattr(r, "masks", None) is not None and getattr(r, "boxes", None) is not None and len(getattr(r.masks, "xy", []) or []) > 0:
        boxes_conf = r.boxes.conf.cpu().numpy() if getattr(r, "boxes", None) is not None else np.array([])
        for mask_pts, score in zip(getattr(r.masks, "xy", []), boxes_conf):
            if score < conf_thresh:
                continue
            pts = np.array(mask_pts, dtype=np.int32)
            M = cv2.moments(pts)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            banana_centers.append((cx, cy))

    else:
        # Fallback: detection model
        if getattr(r, "boxes", None) is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            boxes_conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), score in zip(boxes_xyxy, boxes_conf):
                if score < conf_thresh:
                    continue
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                banana_centers.append((cx, cy))

    banana_count = len(banana_centers)

    # --- draw tiny red dots + optional numbers ---
    dot_radius = 3           # smaller dot
    font_scale = 0.4         # smaller numbers
    thickness = 1

    for idx, (cx, cy) in enumerate(banana_centers, start=1):
        # tiny red dot
        cv2.circle(image, (cx, cy), dot_radius, (0, 0, 255), -1)

        if show_ids:
            # small yellow id, slightly offset
            cv2.putText(image, str(idx), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 255), thickness, cv2.LINE_AA)

    # total count (keep big and clear)
    cv2.putText(image, f"Bananas detected = {banana_count}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    return banana_count, image
