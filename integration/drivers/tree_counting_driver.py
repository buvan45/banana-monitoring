import cv2

def run_tree_counter(image_path):
    from pathlib import Path
    from ultralytics import YOLO
    _THIS_DIR = Path(__file__).resolve().parent
    model = YOLO(str(_THIS_DIR.parent.joinpath("models", "tree_counter.pt")))

    results = model.predict(
        source=image_path,
        conf=0.6,
        iou=0.45,
        classes=[0],
        max_det=500
    )

    boxes = results[0].boxes.xywh.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    filtered_boxes = []
    for (x, y, w, h), score in zip(boxes, scores):
        if 30 < w < 300 and 30 < h < 300:
            filtered_boxes.append((x, y, w, h))

    num_trees = len(filtered_boxes)
    image = cv2.imread(image_path)

    for i, (x, y, w, h) in enumerate(filtered_boxes, start=1):
        x_center = int(x)
        y_center = int(y)
        cv2.circle(image, (x_center, y_center), radius=5, color=(0, 255, 0), thickness=-1)
        cv2.putText(image, str(i), (x_center + 8, y_center - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(image, f"Count: {num_trees}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return num_trees, image
