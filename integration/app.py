# app.py
"""
Flask backend replacing the Streamlit front-end.
Supports:
- Web Camera Realtime (base64 frame processing via AJAX)
- Drone Mode (RTMP MJPEG streaming)
- Video Demo (MJPEG streaming of uploaded videos)
- Image Detection (Instant upload and display)
- Snapshots gallery and dynamic telemetry monitoring.
"""

import sys
import os
import time
import base64
import threading
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename

# --- Paths & Imports ---
_THIS_FILE = Path(__file__).resolve()
_PROJECT_DIR = _THIS_FILE.parent

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

# Import drivers
try:
    from drivers import DRIVERS as AVAILABLE_DRIVERS
except Exception:
    if str(_PROJECT_DIR.joinpath("drivers")) not in sys.path:
        sys.path.insert(0, str(_PROJECT_DIR.joinpath("drivers")))
    from drivers import DRIVERS as AVAILABLE_DRIVERS  # type: ignore

try:
    import banana_counter_driver
except Exception:
    try:
        from drivers import banana_counter_driver  # type: ignore
    except Exception:
        banana_counter_driver = None

# Tree model path
TREE_MODEL_PATH = r"C:\Users\buvan\Downloads\integration\integration\models\tree_counter.pt"
if not Path(TREE_MODEL_PATH).exists():
    # Fallback to local search
    TREE_MODEL_PATH = str(_PROJECT_DIR.joinpath("models", "tree_counter.pt"))

# Setup directories
SNAPSHOT_DIR = _PROJECT_DIR.joinpath("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

UPLOAD_DIR = _PROJECT_DIR.joinpath("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# --- Global States ---
model_states = {}
loaded_models = {}
loaded_models_lock = threading.Lock()

RTMP_DRONE_URL = "rtmp://127.0.0.1:1935/live/stream"

def get_model_state(model_name):
    if model_name not in model_states:
        model_states[model_name] = {
            "current_count": 0,
            "cumulative_total": 0,
            "prev_max_seen": 0,
            "last_snapshot_time": 0.0,
            "_last_autosave": 0.0,
            "last_label": None,
            "last_confidence": 0.0,
            "fps": 0.0,
            "last_processed_image": None
        }
    return model_states[model_name]

def _is_banana_page(name: str) -> bool:
    nl = name.lower()
    return "banana" in nl and "counter" in nl

def _is_tree_page(name: str) -> bool:
    return "tree" in name.lower()

def get_loaded_model(model_name: str, device: str = "cpu"):
    global loaded_models
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    
    cache_key = f"{model_name}::{device}"
    
    with loaded_models_lock:
        if cache_key in loaded_models:
            return loaded_models[cache_key]
        
        dev_arg = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        
        # 1. Banana special-case
        if _is_banana_page(model_name) and banana_counter_driver is not None:
            try:
                model = banana_counter_driver.get_banana_model()
                try:
                    model.to(dev_arg)
                except Exception:
                    pass
                loaded_models[cache_key] = model
                return model
            except Exception as e:
                raise FileNotFoundError(f"Failed to load Banana Counter: {str(e)}")
                
        # 2. Tree special-case
        if _is_tree_page(model_name):
            if not Path(TREE_MODEL_PATH).exists():
                raise FileNotFoundError(f"Tree model file not found at: {TREE_MODEL_PATH}")
            try:
                model = YOLO(TREE_MODEL_PATH)
                try:
                    model.to(dev_arg)
                except Exception:
                    pass
                loaded_models[cache_key] = model
                return model
            except Exception as e:
                raise FileNotFoundError(f"Failed to load Tree Counter YOLO model: {str(e)}")
                
        # 3. Generic Drivers
        selected_mod = AVAILABLE_DRIVERS.get(model_name)
        if selected_mod is not None:
            try:
                model = selected_mod.get_model()
                if hasattr(model, "to"):
                    try:
                        model.to(dev_arg)
                    except Exception:
                        pass
                loaded_models[cache_key] = model
                return model
            except Exception as e:
                raise FileNotFoundError(f"Failed to load driver for {model_name}: {str(e)}")
                
        raise FileNotFoundError(f"Model driver '{model_name}' not registered.")

# --- YOLO Results Centroid Helper ---
def _yolo_results_to_centers_vis(r, conf_thresh: float):
    centers = []
    vis = getattr(r, "orig_img", None)
    masks = getattr(r, "masks", None)
    boxes = getattr(r, "boxes", None)

    if masks is not None and boxes is not None:
        mask_xy = getattr(masks, "xy", None) or []
        boxes_conf = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.array([])
        if mask_xy and len(mask_xy) > 0:
            for mask_pts, score in zip(mask_xy, boxes_conf):
                if score < conf_thresh:
                    continue
                pts = np.array(mask_pts, dtype=np.int32)
                M = cv2.moments(pts)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

    if not centers and boxes is not None and len(boxes) > 0:
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        boxes_conf = boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), score in zip(boxes_xyxy, boxes_conf):
            if score < conf_thresh:
                continue
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy))

    return centers, vis

# --- Unified Prediction Runner ---
def run_model_inference(model_name: str, img: np.ndarray, settings: dict, state: dict) -> Tuple[np.ndarray, List[Tuple[int, int]], int, Optional[str], float, str]:
    """
    Runs model inference on an image and draws annotations.
    Returns: (annotated_image, centers, count, classification_label, classification_conf, device_used)
    """
    import torch
    device = settings.get("device", "cpu")
    imgsz = int(settings.get("imgsz", 640))
    conf_thresh = float(settings.get("conf", 0.25))
    show_ids = settings.get("show_ids", True)
    dot_radius = int(settings.get("dot_radius", 3))
    font_scale = float(settings.get("font_scale", 0.6))
    
    # Load cached model
    model_obj = get_loaded_model(model_name, device)
    dev_arg = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    device_used = dev_arg
    print(f"[DEBUG] run_model_inference: model={model_name}, device_setting={device}, cuda_avail={torch.cuda.is_available()}, dev_arg={dev_arg}")
    
    vis = None
    centers = []
    count = 0
    label = None
    confidence = 0.0

    with torch.no_grad():
        # 1. Banana Counter Special Case
        if _is_banana_page(model_name):
            try:
                results = model_obj.predict(
                    source=img,
                    imgsz=imgsz,
                    conf=conf_thresh,
                    iou=0.5,
                    device=dev_arg,
                    verbose=False
                )
                device_used = dev_arg
            except Exception as e:
                print(f"[DEBUG] Banana CUDA prediction failed, falling back to CPU. Error: {str(e)}")
                results = model_obj.predict(
                    source=img,
                    imgsz=imgsz,
                    conf=conf_thresh,
                    iou=0.5,
                    device="cpu",
                    verbose=False
                )
                device_used = "cpu"
            r = results[0]
            centers, vis = _yolo_results_to_centers_vis(r, conf_thresh)
            if vis is None:
                vis = img.copy()
            else:
                vis = vis.copy()
            count = len(centers)
            
        # 2. Tree Counter Special Case
        elif _is_tree_page(model_name):
            try:
                results = model_obj.predict(
                    source=img,
                    conf=0.6,
                    iou=0.45,
                    classes=[0],
                    max_det=500,
                    imgsz=imgsz,
                    device=dev_arg,
                    verbose=False
                )
                device_used = dev_arg
            except Exception as e:
                print(f"[DEBUG] Tree CUDA prediction failed, falling back to CPU. Error: {str(e)}")
                results = model_obj.predict(
                    source=img,
                    conf=0.6,
                    iou=0.45,
                    classes=[0],
                    max_det=500,
                    imgsz=imgsz,
                    device="cpu",
                    verbose=False
                )
                device_used = "cpu"
            r = results[0]
            boxes = r.boxes.xywh.cpu().numpy() if getattr(r, "boxes", None) is not None else np.zeros((0, 4))
            scores = r.boxes.conf.cpu().numpy() if getattr(r, "boxes", None) is not None else np.zeros((0,))
            centers = []
            for (x, y, w, h), _score in zip(boxes, scores):
                if 30 < w < 300 and 30 < h < 300:
                    centers.append((int(x), int(y)))
            vis = getattr(r, "orig_img", None)
            if vis is None:
                vis = img.copy()
            else:
                vis = vis.copy()
            count = len(centers)
    
        # 3. Generic Drivers
        else:
            selected_mod = AVAILABLE_DRIVERS[model_name]
            results = selected_mod.run_on_image(model_obj, img, conf_thresh=conf_thresh, imgsz=imgsz)
            
            if results.get("type") == "detection":
                vis = results.get("vis")
                if vis is None:
                    vis = img.copy()
                else:
                    vis = vis.copy()
                centers = results.get("centers", []) or []
                total_override = results.get("count", None)
                count = int(total_override) if total_override is not None else len(centers)
                
            elif results.get("type") == "classification":
                vis = results.get("vis")
                if vis is None:
                    vis = img.copy()
                else:
                    vis = vis.copy()
                label = results.get("label", "unknown")
                confidence = float(results.get("confidence", 0.0))
                count = 1
                state["last_label"] = label
                state["last_confidence"] = confidence

    # Draw generic overlays for detection results
    if not _is_banana_page(model_name) and not _is_tree_page(model_name) and label is None:
        # Draw centers & IDs for generic detectors if returned
        if centers:
            for idx, (cx, cy) in enumerate(centers, start=1):
                cv2.circle(vis, (cx, cy), dot_radius, (0, 0, 255), -1)
                if show_ids:
                    cv2.putText(vis, str(idx), (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(vis, (10, 10), (420, 72), (0, 0, 0), -1)
        cv2.putText(vis, f"Detected = {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    elif _is_banana_page(model_name) or _is_tree_page(model_name):
        # Draw centroids + IDs for special cases
        for idx, (cx, cy) in enumerate(centers, start=1):
            cv2.circle(vis, (cx, cy), dot_radius, (0, 0, 255), -1)
            if show_ids:
                cv2.putText(vis, str(idx), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(vis, (10, 10), (420, 72), (0, 0, 0), -1)
        cv2.putText(vis, f"Detected = {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    elif label is not None:
        # Classification layout
        cv2.rectangle(vis, (10, 10), (600, 72), (0, 0, 0), -1)
        cv2.putText(vis, f"{label} ({confidence:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    # Update cumulative state metrics
    state["current_count"] = count
    prev_max = state.get("prev_max_seen", 0)
    if count > prev_max:
        delta = count - prev_max
        state["cumulative_total"] = state.get("cumulative_total", 0) + delta
        state["prev_max_seen"] = count
    
    state["last_processed_image"] = vis

    # Auto-save Snapshot Logic
    auto_save_enabled = settings.get("auto_save", False)
    auto_save_threshold = int(settings.get("auto_save_thresh", 20))
    now = time.time()
    
    if auto_save_enabled and count >= auto_save_threshold and now - state.get("_last_autosave", 0.0) > 2.0:
        try:
            snap_path = SNAPSHOT_DIR.joinpath(f"auto_{int(now)}.jpg")
            cv2.imwrite(str(snap_path), vis)
            state["_last_autosave"] = now
        except Exception as e:
            print("Auto-save failed:", e)

    return vis, centers, count, label, confidence, device_used

# --- Thread-Safe RTMP Stream Singleton ---
class RTMPGrabber:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, url):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RTMPGrabber, cls).__new__(cls)
                cls._instance.url = url
                cls._instance.cap = None
                cls._instance.frame = None
                cls._instance.running = False
                cls._instance.thread = None
                cls._instance.ref_count = 0
            return cls._instance

    def start(self):
        with self._lock:
            self.ref_count += 1
            if not self.running:
                # Use DirectShow on windows or FFMPEG backend if possible
                self.cap = cv2.VideoCapture(self.url + "?fflags=nobuffer&flags=low_delay&rtmp_live=live", cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)
                self.running = True
                self.thread = threading.Thread(target=self._update, daemon=True)
                self.thread.start()
                print(f"Started RTMP Grabber for {self.url}")

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.005)

    def get_latest(self):
        return self.frame

    def stop(self):
        with self._lock:
            self.ref_count -= 1
            if self.ref_count <= 0 and self.running:
                self.running = False
                if self.cap:
                    self.cap.release()
                self.frame = None
                print(f"Stopped RTMP Grabber for {self.url}")

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/available_models', methods=['GET'])
def available_models():
    print("[DEBUG] AVAILABLE_DRIVERS keys:", list(AVAILABLE_DRIVERS.keys()), flush=True)
    models = [name for name, mod in AVAILABLE_DRIVERS.items() if mod is not None]
    return jsonify(models)

@app.route('/process_frame/<model_name>', methods=['POST'])
def process_frame(model_name):
    """Handles real-time webcam frame processing via AJAX."""
    if 'frame' not in request.files:
        return jsonify({"status": "error", "message": "No frame file uploaded."}), 400
        
    file = request.files['frame']
    # Read settings from request form
    settings = {
        "device": request.form.get("device", "cpu"),
        "imgsz": request.form.get("imgsz", 640),
        "conf": request.form.get("conf", 0.25),
        "show_ids": request.form.get("show_ids") == 'true',
        "dot_radius": request.form.get("dot_radius", 3),
        "font_scale": request.form.get("font_scale", 0.6),
        "auto_save": request.form.get("auto_save") == 'true',
        "auto_save_thresh": request.form.get("auto_save_thresh", 20)
    }
    
    # Read file stream directly to OpenCV
    try:
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Could not decode uploaded frame image."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Frame read failure: {str(e)}"}), 500

    state = get_model_state(model_name)
    
    try:
        start_time = time.time()
        vis, centers, count, label, conf, device_used = run_model_inference(model_name, img, settings, state)
        latency = time.time() - start_time
        state["fps"] = 1.0 / latency if latency > 0 else 0.0
        
        # Encode vis frame back to JPG base64
        ret, buffer = cv2.imencode('.jpg', vis)
        if not ret:
            return jsonify({"status": "error", "message": "Could not encode annotated output image."}), 500
            
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "image": base64_jpg,
            "current_count": state["current_count"],
            "cumulative_total": state["cumulative_total"],
            "fps": round(state["fps"], 1),
            "label": label,
            "confidence": conf,
            "device_used": device_used
        })
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"Model inference error: {str(e)}"}), 500

@app.route('/image_upload', methods=['POST'])
def image_upload():
    """Processes static uploaded images and returns annotations immediately."""
    if 'image' not in request.files:
        return jsonify({"status": "error", "message": "No image file uploaded."}), 400
        
    file = request.files['image']
    model_name = request.form.get("model_name")
    
    settings = {
        "device": request.form.get("device", "cpu"),
        "imgsz": request.form.get("imgsz", 640),
        "conf": request.form.get("conf", 0.25),
        "show_ids": request.form.get("show_ids") == 'true',
        "dot_radius": request.form.get("dot_radius", 3),
        "font_scale": request.form.get("font_scale", 0.6),
        "auto_save": False,
        "auto_save_thresh": 20
    }
    
    try:
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"status": "error", "message": "Could not decode image."}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Image decode failure: {str(e)}"}), 500

    # Image upload doesn't affect cumulative webcam dashboard metrics,
    # so we process with a local isolated state.
    dummy_state = {
        "current_count": 0,
        "cumulative_total": 0,
        "prev_max_seen": 0,
        "last_snapshot_time": 0.0,
        "_last_autosave": 0.0,
        "last_label": None,
        "last_confidence": 0.0,
        "fps": 0.0,
        "last_processed_image": None
    }
    
    try:
        vis, centers, count, label, conf, device_used = run_model_inference(model_name, img, settings, dummy_state)
        ret, buffer = cv2.imencode('.jpg', vis)
        base64_jpg = base64.b64encode(buffer).decode('utf-8')
        
        # Save last processed image to this model's main state for snapshots
        state = get_model_state(model_name)
        state["last_processed_image"] = vis
        
        return jsonify({
            "status": "success",
            "image": base64_jpg,
            "count": count,
            "label": label,
            "confidence": conf,
            "device_used": device_used
        })
    except FileNotFoundError as e:
        return jsonify({"status": "error", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"Inference execution error: {str(e)}"}), 500

@app.route('/video_upload', methods=['POST'])
def video_upload():
    """Uploads a video file and saves it locally for demonstration streaming."""
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file uploaded."}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename."}), 400
        
    filename = secure_filename(file.filename)
    filepath = UPLOAD_DIR.joinpath(filename)
    file.save(str(filepath))
    
    return jsonify({"status": "success", "filename": filename})

@app.route('/video_feed/demo/<filename>/<model_name>')
def video_feed_demo(filename, model_name):
    """Serves the MJPEG processed stream of the uploaded video."""
    filepath = UPLOAD_DIR.joinpath(secure_filename(filename))
    if not filepath.exists():
        return "Video not found", 404
        
    settings = {
        "device": request.args.get("device", "cpu"),
        "imgsz": request.args.get("imgsz", 640),
        "conf": request.args.get("conf", 0.25),
        "show_ids": request.args.get("show_ids") == 'true',
        "dot_radius": request.args.get("dot_radius", 3),
        "font_scale": request.args.get("font_scale", 0.6),
        "auto_save": request.args.get("auto_save") == 'true',
        "auto_save_thresh": request.args.get("auto_save_thresh", 20)
    }

    def gen_video_frames():
        cap = cv2.VideoCapture(str(filepath))
        state = get_model_state(model_name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        delay = 1.0 / fps
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                start_time = time.time()
                vis, centers, count, label, conf, device_used = run_model_inference(model_name, frame, settings, state)
                
                latency = time.time() - start_time
                state["fps"] = 1.0 / latency if latency > 0 else 0.0
                
                ret, buffer = cv2.imencode('.jpg', vis)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Match native framerate delay
                elapsed = time.time() - start_time
                sleep_time = max(0.001, delay - elapsed)
                time.sleep(sleep_time)
        except GeneratorExit:
            # Browser terminated video connection
            pass
        finally:
            cap.release()

    return Response(gen_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/drone/<model_name>')
def video_feed_drone(model_name):
    """Serves the MJPEG processed stream of the Drone RTMP grabber."""
    settings = {
        "device": request.args.get("device", "cpu"),
        "imgsz": request.args.get("imgsz", 640),
        "conf": request.args.get("conf", 0.25),
        "show_ids": request.args.get("show_ids") == 'true',
        "dot_radius": request.args.get("dot_radius", 3),
        "font_scale": request.args.get("font_scale", 0.6),
        "auto_save": request.args.get("auto_save") == 'true',
        "auto_save_thresh": request.args.get("auto_save_thresh", 20)
    }
    
    grabber = RTMPGrabber(RTMP_DRONE_URL)
    grabber.start()
    
    def gen_drone_frames():
        state = get_model_state(model_name)
        try:
            while True:
                img = grabber.get_latest()
                if img is None:
                    # Serve an empty frame or sleep while waiting
                    time.sleep(0.03)
                    continue
                    
                start_time = time.time()
                vis, centers, count, label, conf, device_used = run_model_inference(model_name, img, settings, state)
                
                latency = time.time() - start_time
                state["fps"] = 1.0 / latency if latency > 0 else 0.0
                
                ret, buffer = cv2.imencode('.jpg', vis)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.01)
        except GeneratorExit:
            pass
        finally:
            grabber.stop()

    return Response(gen_drone_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/metrics/<model_name>', methods=['GET'])
def metrics(model_name):
    state = get_model_state(model_name)
    return jsonify({
        "current_count": state["current_count"],
        "cumulative_total": state["cumulative_total"],
        "fps": round(state["fps"], 1),
        "label": state.get("last_label"),
        "confidence": state.get("last_confidence", 0.0)
    })

@app.route('/reset_counts/<model_name>', methods=['POST'])
def reset_counts(model_name):
    state = get_model_state(model_name)
    state["current_count"] = 0
    state["cumulative_total"] = 0
    state["prev_max_seen"] = 0
    state["fps"] = 0.0
    state["last_label"] = None
    state["last_confidence"] = 0.0
    return jsonify({"status": "success"})

@app.route('/capture_snapshot', methods=['POST'])
def capture_snapshot():
    """Captures the last processed image in memory for a model and saves to disk."""
    data = request.json or {}
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"status": "error", "message": "Missing model_name parameter"}), 400
        
    state = get_model_state(model_name)
    last_img = state.get("last_processed_image")
    
    if last_img is not None:
        filename = f"manual_{int(time.time())}.jpg"
        filepath = SNAPSHOT_DIR.joinpath(filename)
        try:
            cv2.imwrite(str(filepath), last_img)
            return jsonify({"status": "success", "filename": filename})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Write failed: {str(e)}"}), 500
            
    return jsonify({"status": "error", "message": "No processed frame available to capture."}), 400

@app.route('/snapshots', methods=['GET'])
def list_snapshots():
    files = sorted([f.name for f in SNAPSHOT_DIR.glob("*.jpg")], key=lambda x: os.path.getmtime(SNAPSHOT_DIR.joinpath(x)), reverse=True)
    return jsonify(files)

@app.route('/snapshots/<filename>')
def get_snapshot(filename):
    return send_from_directory(str(SNAPSHOT_DIR), secure_filename(filename))

@app.route('/snapshots/delete/<filename>', methods=['POST', 'DELETE'])
def delete_snapshot(filename):
    filepath = SNAPSHOT_DIR.joinpath(secure_filename(filename))
    if filepath.exists():
        try:
            filepath.unlink()
            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "error", "message": "Snapshot file not found."}), 404


if __name__ == '__main__':
    # Get port from environment (Hugging Face uses PORT=7860)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
