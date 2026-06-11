# integration/drivers/__init__.py
"""
Auto-discover driver modules and adapt them to a stable API:
  - get_model(weights_path=None) -> model_obj (or module)
  - run_on_image(model, image: np.ndarray, *, conf_thresh=0.25, imgsz=416) -> dict

The dict returned by run_on_image should be:
  - detection:
      {
        "type": "detection",
        "centers": [(cx,cy), ...],   # may be empty if driver only returns a count
        "vis": bgr_numpy_image_or_None,
        "count": int_or_None        # optional; used when centers list is empty
      }
  - classification:
      {
        "type": "classification",
        "label": "class name",
        "confidence": float,
        "vis": bgr_numpy_image_or_None
      }

We DO NOT modify driver internals or paths; we just adapt their outputs.
"""

import importlib
import pkgutil
from pathlib import Path
import tempfile
import inspect
import os
from typing import Dict

import numpy as np
import cv2

_THIS_DIR = Path(__file__).resolve().parent

# discover driver modules (files ending with _driver.py)
_module_names = []
for _finder, _name, _ispkg in pkgutil.iter_modules([str(_THIS_DIR)]):
    if _name.endswith("_driver") and _name != "__init__":
        _module_names.append(_name)

DRIVERS: Dict[str, object] = {}


def _save_tmp_image_and_get_path(img: np.ndarray) -> str:
    """Save BGR numpy image to a temp JPG and return path."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    path = tf.name
    cv2.imwrite(path, img)
    tf.close()
    return path


def _make_adapter(mod, display_name=None):
    """
    Build an adapter object with:
      - get_model(weights_path=None)
      - run_on_image(model, image: np.ndarray, *, conf_thresh=0.25, imgsz=416)
    """

    class Adapter:
        DRIVER_NAME = getattr(mod, "DRIVER_NAME", display_name or mod.__name__.replace("_driver", ""))
        DRIVER_TYPE = getattr(mod, "DRIVER_TYPE", None)
        __original_module__ = mod

        @staticmethod
        def get_model(weights_path=None):
            # Prefer explicit factory functions if present
            for name in ("get_model", "get_banana_model", "load_model"):
                if hasattr(mod, name):
                    fn = getattr(mod, name)
                    try:
                        return fn(weights_path) if weights_path is not None else fn()
                    except TypeError:
                        return fn()
            # Otherwise return the module itself (drivers using global models)
            return mod

        @staticmethod
        def run_on_image(model, image: np.ndarray, *, conf_thresh=0.25, imgsz=416):
            """
            Generic adapter. For drivers that expect an image path, we save to a
            temporary file and call their run_* function.
            """
            # If the module already has run_on_image, call that directly.
            if hasattr(mod, "run_on_image"):
                try:
                    return mod.run_on_image(model, image, conf_thresh=conf_thresh, imgsz=imgsz)
                except TypeError:
                    return mod.run_on_image(image, conf_thresh=conf_thresh, imgsz=imgsz)

            def call_with_image_path(fn):
                tmp_path = _save_tmp_image_and_get_path(image)
                try:
                    return fn(tmp_path)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            # Pick a run_* function
            run_names = [n for n in dir(mod) if n.startswith("run_")]
            preferred = [
                "run_banana_counter",
                "run_tree_counter",
                "run_banana_ripeness",
                "run_banana_variety",
                "run_banana_leaf_disease",
                "run_banana_disease",
                "run",
            ]
            chosen = None
            for p in preferred:
                if p in run_names:
                    chosen = p
                    break
            if chosen is None and run_names:
                chosen = run_names[0]

            if chosen is None:
                # nothing obvious to call
                return {"type": "detection", "centers": [], "vis": image, "count": 0}

            fn = getattr(mod, chosen)

            # Try calling with array, else with path, else with (model, path)
            try:
                try:
                    res = fn(image)
                except Exception:
                    res = call_with_image_path(fn)
            except TypeError:
                try:
                    res = call_with_image_path(fn)
                except Exception:
                    tmp_path = _save_tmp_image_and_get_path(image)
                    try:
                        res = fn(model, tmp_path)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

            # ---- Normalize outputs ----
            if isinstance(res, dict):
                # assume dict already matches our schema
                return res

            # Tuples/lists from your drivers
            if isinstance(res, (tuple, list)):
                # 2-element forms
                if len(res) == 2:
                    a, b = res
                    # (count, vis_image) for detectors like tree_counter
                    if isinstance(a, int):
                        count = int(a)
                        vis = b if isinstance(b, np.ndarray) else None
                        return {
                            "type": "detection",
                            "centers": [],
                            "vis": vis,
                            "count": count,
                        }
                    # (label, vis_image_or_none) or (label, info_dict)
                    if isinstance(a, str):
                        label = a
                        vis = b if isinstance(b, np.ndarray) else None
                        return {
                            "type": "classification",
                            "label": label,
                            "confidence": 1.0,
                            "vis": vis,
                        }

                # 3-element forms
                if len(res) == 3:
                    a, b, c = res
                    # (label, info_dict, vis)
                    if isinstance(a, str) and (isinstance(b, dict) or b is None):
                        label = a
                        conf = 0.0
                        if isinstance(b, dict):
                            # try to extract confidence if present
                            for k in ("prob", "probability", "confidence"):
                                if k in b:
                                    try:
                                        conf = float(b[k])
                                        break
                                    except Exception:
                                        pass
                        vis = c if isinstance(c, np.ndarray) else None
                        return {
                            "type": "classification",
                            "label": label,
                            "confidence": conf,
                            "vis": vis,
                        }
                    # (label, prob_number, vis)
                    if isinstance(a, str) and isinstance(b, (float, int)):
                        label = a
                        conf = float(b)
                        vis = c if isinstance(c, np.ndarray) else None
                        return {
                            "type": "classification",
                            "label": label,
                            "confidence": conf,
                            "vis": vis,
                        }

                # unknown tuple structure: safest is to just return original image
                return {
                    "type": "detection",
                    "centers": [],
                    "vis": image,
                    "count": 0,
                }

            # Single string => classification label
            if isinstance(res, str):
                return {
                    "type": "classification",
                    "label": res,
                    "confidence": 1.0,
                    "vis": image,
                }

            # Fallback
            return {
                "type": "detection",
                "centers": [],
                "vis": image,
                "count": 0,
            }

    return Adapter


# Discover and register all driver adapters
for name in _module_names:
    try:
        full = f"{__package__}.{name}"
        mod = importlib.import_module(full)
        display = getattr(mod, "DRIVER_NAME", name.replace("_driver", "").replace("_", " ").title())
        adapter = _make_adapter(mod, display)
        DRIVERS[display] = adapter
    except Exception:
        # Skip broken drivers but keep going
        continue
