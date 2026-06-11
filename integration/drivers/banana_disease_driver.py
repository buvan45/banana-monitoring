# drivers/banana_disease_driver.py

from ultralytics import YOLO

from pathlib import Path
import os

_PROJECT_DIR = Path(__file__).resolve().parent.parent
disease_model_path = str(_PROJECT_DIR.joinpath("models", "banana_disease__classification.pt"))

_DISEASE_MODEL = None
disease_class_names = []

def get_disease_model():
    global _DISEASE_MODEL, disease_class_names
    if _DISEASE_MODEL is None:
        _DISEASE_MODEL = YOLO(disease_model_path)
        disease_class_names = list(_DISEASE_MODEL.names.values())
        print("Banana disease classes:", disease_class_names)
    return _DISEASE_MODEL

# Optional: info/description for each disease class
class_info = {
    "Banana healthy": {
        "severity": "None",
        "advice": "Plant is healthy. Keep current care and monitoring."
    },
    "Anthracnose": {
        "severity": "High",
        "advice": "Fungal disease. Remove infected tissue and apply fungicide."
    },
    "Black Sigatoka": {
        "severity": "High",
        "advice": "Serious leaf spot disease. Consider fungicide and sanitation."
    },
    "Yellow Sigatoka": {
        "severity": "Medium",
        "advice": "Leaf spot disease. Improve airflow and consider fungicide."
    },
    "Panama Disease": {
        "severity": "Very High",
        "advice": "Soil-borne wilt. Avoid moving contaminated soil and tools."
    },
    "Bacterial Soft Rot": {
        "severity": "High",
        "advice": "Bacterial infection. Remove rotting tissue and improve drainage."
    },
    "Banana Aphids": {
        "severity": "Medium",
        "advice": "Sap-sucking insects. Consider biological or chemical control."
    },
    "Banana Fruit- Scarring Beetle": {
        "severity": "Medium",
        "advice": "Fruit quality issue. Use field sanitation and trapping."
    },
    "Potassium Deficiency": {
        "severity": "Nutrient issue",
        "advice": "Adjust fertilization with potassium-rich fertilizer."
    },
    "Psudostem Weevil": {
        "severity": "High",
        "advice": "Stem borer. Destroy heavily infested plants and manage adults."
    },
    "Unlabeled": {
        "severity": "Unknown",
        "advice": "Class marked as 'Unlabeled' in dataset. Check image manually."
    },
}


def run_banana_disease(image_path: str):
    """
    Run banana disease classification on a single image.

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    pred_class : str
        Predicted disease class name (e.g., 'Anthracnose', 'Banana healthy').
    info : dict
        Extra info about the class, including severity/advice and confidence.
    """
    if not os.path.exists(disease_model_path):
        raise FileNotFoundError(f"Model weights not found at: {disease_model_path}")

    model = get_disease_model()
    # YOLO classification prediction
    results = model.predict(
        source=image_path,
        imgsz=224,
        verbose=False
    )

    r = results[0]

    # Safety check: classification models should always have r.probs
    if r.probs is None:
        return "unknown", {
            "severity": "Unknown",
            "advice": "Model returned no probabilities.",
            "confidence": "N/A",
        }

    top1_idx = int(r.probs.top1)
    top1_conf = float(r.probs.top1conf)

    pred_class = disease_class_names[top1_idx]

    info = class_info.get(
        pred_class,
        {
            "severity": "Unknown",
            "advice": f"No description stored for class '{pred_class}'.",
        },
    )
    # add confidence string
    info = {
        **info,
        "confidence": f"{top1_conf * 100:.2f}%",
    }

    return pred_class, info


# Optional: quick local test
if __name__ == "__main__":
    test_image = r"C:\Users\buvan\Downloads\Banana Disease.v2i.folder\test\Banana healthy\some_image.jpg"
    pred, details = run_banana_disease(test_image)
    print("Prediction:", pred)
    print("Details:", details)
