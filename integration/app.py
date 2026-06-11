# realtime_integration.py
"""
Realtime Banana/Model Counter + Video Demo (multi-driver aware; banana & tree special-cased)

- Banana counter uses YOLO.predict(device=...) directly (fast CUDA when available).
- Tree counter uses its YOLO weights directly (once) and mirrors your filtering logic.
- Other drivers go through the adapter registry in integration/drivers/__init__.py.
- Classification labels are drawn on the video for both Realtime and Video Demo.
"""

import sys
from pathlib import Path
import time
import cv2
import numpy as np
import tempfile
import av
import streamlit as st
import torch
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO  # for banana & tree special cases

# --- Paths & imports ---
_THIS_FILE = Path(__file__).resolve()
_PROJECT_DIR = _THIS_FILE.parent

if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

try:
    from integration.drivers import DRIVERS as AVAILABLE_DRIVERS
except Exception:
    if str(_PROJECT_DIR.joinpath("drivers")) not in sys.path:
        sys.path.insert(0, str(_PROJECT_DIR.joinpath("drivers")))
    from drivers import DRIVERS as AVAILABLE_DRIVERS  # type: ignore

# try to import raw banana and tree drivers to detect pages
try:
    from integration.drivers import banana_counter_driver
except Exception:
    try:
        import banana_counter_driver  # type: ignore
    except Exception:
        banana_counter_driver = None

try:
    from integration.drivers import tree_counting_driver  # noqa: F401
except Exception:
    try:
        import tree_counting_driver  # type: ignore
    except Exception:
        tree_counting_driver = None

# Tree model path (same as in your tree_counting_driver)
import os
TREE_MODEL_PATH = str(_PROJECT_DIR.joinpath("models", "tree_counter.pt"))
if not os.path.exists(TREE_MODEL_PATH):
    TREE_MODEL_PATH = r"E:\SDP\integration\models\tree_counter.pt"

# --- Streamlit config ---
st.set_page_config(page_title="🌿 Banana & Models Realtime", layout="wide")
st.title("🌿 Banana & Models Realtime / Video Demo")

SNAPSHOT_DIR = _PROJECT_DIR.joinpath("snapshots")
SNAPSHOT_DIR.mkdir(exist_ok=True)

# Choose model page
model_names = [name for name, mod in AVAILABLE_DRIVERS.items() if mod is not None]
if not model_names:
    st.error("No drivers found in integration/drivers.")
    st.stop()

selected_model_name = st.sidebar.selectbox("Select model page", options=model_names, index=0)
selected_mod = AVAILABLE_DRIVERS[selected_model_name]

# --- Model Weights Existence & Uploader Check ---
MODEL_WEIGHTS_INFO = {
    "Banana Counter": {
        "filename": "banana_counter.pt",
        "path": _PROJECT_DIR.joinpath("models", "banana_counter.pt"),
        "help": "YOLO model weights for counting bananas."
    },
    "Tree Counting": {
        "filename": "tree_counter.pt",
        "path": _PROJECT_DIR.joinpath("models", "tree_counter.pt"),
        "help": "YOLO model weights for counting trees."
    },
    "Banana Ripeness": {
        "filename": "banana ripeness classification-2.pt",
        "path": _PROJECT_DIR.joinpath("models", "banana ripeness classification-2.pt"),
        "help": "YOLO model weights for banana ripeness classification."
    },
    "Banana Variety": {
        "filename": "banana_variety.pth",
        "path": _PROJECT_DIR.joinpath("models", "banana_variety.pth"),
        "help": "ResNet model weights for banana variety classification.",
        "extra_files": [
            {
                "filename": "banana_variety_class_names.pth",
                "path": _PROJECT_DIR.joinpath("models", "banana_variety_class_names.pth"),
                "help": "Class names file for banana variety."
            }
        ]
    },
    "Banana Disease": {
        "filename": "banana_disease__classification.pt",
        "path": _PROJECT_DIR.joinpath("models", "banana_disease__classification.pt"),
        "help": "YOLO model weights for banana disease classification."
    },
    "Banana Leaf Disease": {
        "filename": "banana_leaf_disease.pth",
        "path": _PROJECT_DIR.joinpath("models", "banana_leaf_disease.pth"),
        "help": "EfficientNet model weights for banana leaf disease classification."
    }
}

info = MODEL_WEIGHTS_INFO.get(selected_model_name)
weights_missing = False
if info:
    target_path = Path(info["path"])
    target_path.parent.mkdir(exist_ok=True)
    
    # Check main weight file
    if not target_path.exists():
        weights_missing = True
        st.sidebar.warning(f"⚠️ Model weights file `{info['filename']}` is missing.")
        uploaded_file = st.file_uploader(f"Upload `{info['filename']}` ({info['help']})", type=["pt", "pth"], key=f"upload_{selected_model_name}")
        if uploaded_file is not None:
            with open(target_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded `{info['filename']}` successfully! Reloading...")
            st.rerun()

    # Check extra files
    for extra in info.get("extra_files", []):
        extra_path = Path(extra["path"])
        if not extra_path.exists():
            weights_missing = True
            st.sidebar.warning(f"⚠️ Required extra file `{extra['filename']}` is missing.")
            uploaded_extra = st.file_uploader(f"Upload `{extra['filename']}` ({extra['help']})", type=["pth"], key=f"upload_extra_{extra['filename']}")
            if uploaded_extra is not None:
                with open(extra_path, "wb") as f:
                    f.write(uploaded_extra.getbuffer())
                st.success(f"Uploaded `{extra['filename']}` successfully! Reloading...")
                st.rerun()

if weights_missing:
    st.info("The application requires the model weights to perform inference. Please upload the required file(s) in the sidebar or upload widget above to proceed.")
    st.stop()

# Per-model state
MODEL_KEY = f"model_state::{selected_model_name}"
if MODEL_KEY not in st.session_state:
    st.session_state[MODEL_KEY] = {
        "current_count": 0,
        "cumulative_total": 0,
        "prev_max_seen": 0,
        "last_snapshot_time": 0.0,
        "_last_autosave": 0.0,
        "last_label": None,
        "last_confidence": 0.0,
    }
MODEL_STATE = st.session_state[MODEL_KEY]

# Sidebar controls
st.sidebar.header("Realtime / Video Demo Controls")
cuda_avail = torch.cuda.is_available()
DEVICE = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0 if cuda_avail else 1)
IMGSZ = st.sidebar.selectbox("Inference image size (imgsz)", [320, 416, 640, 960], index=1)
CONF_THRESH = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
SHOW_IDS = st.sidebar.checkbox("Show per-object IDs (detection)", value=True)
DOT_RADIUS = st.sidebar.slider("Dot radius (px)", 1, 8, 3)
FONT_SCALE = st.sidebar.slider("Font scale", 0.3, 1.5, 0.6, 0.1)
AUTO_SAVE_SNAPSHOTS = st.sidebar.checkbox("Auto-save snapshot on threshold", value=False)
AUTO_SAVE_THRESHOLD = st.sidebar.number_input("Auto-save threshold", 1, 1000, 20, 1)

if DEVICE == "cuda" and not cuda_avail:
    st.sidebar.error("CUDA not available — using CPU instead.")

# Layout
left_col, right_col = st.columns([1, 2])
cur_card_ph = left_col.empty()
cum_card_ph = left_col.empty()
debug_ph = left_col.empty()
right_col.caption(f"Live output — {selected_model_name}")

def card_html(title: str, value: int, bgcolor: str):
    return f"""
      <div style="padding:16px;border-radius:10px;background:{bgcolor};color:white;text-align:center;">
        <div style="font-size:12px;opacity:0.85">{title}</div>
        <div style="font-size:36px;font-weight:700;margin-top:8px">{value}</div>
      </div>
    """

cur_card_ph.markdown(card_html("CURRENT COUNT", MODEL_STATE["current_count"], "#2b6cb0"), unsafe_allow_html=True)
cum_card_ph.markdown(card_html("CUMULATIVE COUNT", MODEL_STATE["cumulative_total"], "#dd6b20"), unsafe_allow_html=True)
debug_ph.text(f"Device: {DEVICE}  |  imgsz: {IMGSZ}  |  conf: {CONF_THRESH:.2f}")

def annotate_and_update(shared_state: dict, vis_bgr: np.ndarray, centers: list, fps_est: float, total_override: int | None = None):
    """
    Annotate vis_bgr with dots/ids and overlay counts/fps; update shared_state.
    If centers list is empty but total_override is not None, we use that as the total.
    """
    if centers:
        for idx, (cx, cy) in enumerate(centers, start=1):
            cv2.circle(vis_bgr, (cx, cy), DOT_RADIUS, (0, 0, 255), -1)
            if SHOW_IDS:
                cv2.putText(vis_bgr, str(idx), (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), 1, cv2.LINE_AA)

    if total_override is not None:
        total = int(total_override)
    else:
        total = len(centers)

    cv2.rectangle(vis_bgr, (10, 10), (420, 72), (0, 0, 0), -1)
    cv2.putText(vis_bgr, f"Detected = {total}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(vis_bgr, f"FPS: {fps_est:.1f}", (10, vis_bgr.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    shared_state["current_count"] = total
    prev_max = shared_state.get("prev_max_seen", 0)
    if total > prev_max:
        delta = total - prev_max
        shared_state["cumulative_total"] = shared_state.get("cumulative_total", 0) + delta
        shared_state["prev_max_seen"] = total
    else:
        shared_state["prev_max_seen"] = prev_max

# Helpers to recognize banana/tree pages
def _is_banana_page(name: str) -> bool:
    nl = name.lower()
    return "banana" in nl and "counter" in nl

def _is_tree_page(name: str) -> bool:
    return "tree" in name.lower()

_IS_BANANA = _is_banana_page(selected_model_name)
_IS_TREE = _is_tree_page(selected_model_name)

# Helper for YOLO results -> centers & vis (for banana)
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

# Mode selector
mode = st.radio(f"{selected_model_name} mode", ["Realtime", "Video Demo","Drone"], index=0, horizontal=True)

# ----------------- REALTIME -----------------
if mode == "Realtime":
    class GenericProcessor(VideoProcessorBase):
        def __init__(self, shared_state: dict, device: str, imgsz: int, conf_thresh: float):
            self.shared_state = shared_state
            self.device = device
            self.imgsz = imgsz
            self.conf_thresh = conf_thresh
            self.prev_time = time.time()
            self.fps = None

            self.banana_model = None
            self.tree_model = None

            dev_arg = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"

            if _IS_BANANA and banana_counter_driver is not None:
                try:
                    self.banana_model = banana_counter_driver.get_banana_model()
                    try:
                        self.banana_model.to(dev_arg)
                    except Exception:
                        pass
                except Exception:
                    self.banana_model = None

            if _IS_TREE and TREE_MODEL_PATH and Path(TREE_MODEL_PATH).exists():
                try:
                    self.tree_model = YOLO(TREE_MODEL_PATH)
                    try:
                        self.tree_model.to(dev_arg)
                    except Exception:
                        pass
                except Exception:
                    self.tree_model = None

        def _update_fps(self):
            now = time.time()
            dt = now - self.prev_time if now - self.prev_time > 1e-6 else 1e-6
            fps = 1.0 / dt
            self.prev_time = now
            self.fps = fps if self.fps is None else (0.85 * self.fps + 0.15 * fps)
            return self.fps, now

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            dev_arg = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"

            # Banana special-case
            if _IS_BANANA and self.banana_model is not None:
                try:
                    results = self.banana_model.predict(
                        source=img,
                        imgsz=self.imgsz,
                        conf=self.conf_thresh,
                        iou=0.5,
                        device=dev_arg,
                        verbose=False,
                    )
                except Exception:
                    try:
                        results = self.banana_model.predict(
                            source=img,
                            imgsz=self.imgsz,
                            conf=self.conf_thresh,
                            iou=0.5,
                            device="cpu",
                            verbose=False,
                        )
                    except Exception:
                        return av.VideoFrame.from_ndarray(img, format="bgr24")

                r = results[0]
                centers, vis = _yolo_results_to_centers_vis(r, self.conf_thresh)
                if vis is None:
                    vis = img.copy()
                fps, now = self._update_fps()
                annotate_and_update(self.shared_state, vis, centers, fps)

                if AUTO_SAVE_SNAPSHOTS and self.shared_state.get("current_count", 0) >= AUTO_SAVE_THRESHOLD and now - self.shared_state.get("_last_autosave", 0.0) > 2.0:
                    try:
                        snap = SNAPSHOT_DIR.joinpath(f"auto_{int(now)}.jpg")
                        cv2.imwrite(str(snap), vis)
                        self.shared_state["_last_autosave"] = now
                    except Exception:
                        pass

                return av.VideoFrame.from_ndarray(vis, format="bgr24")

            # Tree special-case
            if _IS_TREE and self.tree_model is not None:
                try:
                    results = self.tree_model.predict(
                        source=img,
                        conf=0.6,
                        iou=0.45,
                        classes=[0],
                        max_det=500,
                        imgsz=self.imgsz,
                        device=dev_arg,
                        verbose=False,
                    )
                except Exception:
                    try:
                        results = self.tree_model.predict(
                            source=img,
                            conf=0.6,
                            iou=0.45,
                            classes=[0],
                            max_det=500,
                            imgsz=self.imgsz,
                            device="cpu",
                            verbose=False,
                        )
                    except Exception:
                        return av.VideoFrame.from_ndarray(img, format="bgr24")

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
                fps, now = self._update_fps()
                annotate_and_update(self.shared_state, vis, centers, fps)
                return av.VideoFrame.from_ndarray(vis, format="bgr24")

            # Generic adapter path
            try:
                model_obj = selected_mod.get_model()
                results = selected_mod.run_on_image(model_obj, img, conf_thresh=self.conf_thresh, imgsz=self.imgsz)
            except Exception:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            fps, now = self._update_fps()
            if results is None:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            if results.get("type") == "detection":
                vis = results.get("vis")
                if vis is None:
                    vis = img.copy()
                centers = results.get("centers", []) or []
                total_override = results.get("count", None)
                annotate_and_update(self.shared_state, vis, centers, fps, total_override=total_override)
                return av.VideoFrame.from_ndarray(vis, format="bgr24")

            if results.get("type") == "classification":
                vis = results.get("vis")
                if vis is None:
                    vis = img.copy()
                label = results.get("label", "unknown")
                conf = float(results.get("confidence", 0.0))
                self.shared_state["last_label"] = label
                self.shared_state["last_confidence"] = conf
                cv2.rectangle(vis, (10, 10), (600, 72), (0, 0, 0), -1)
                cv2.putText(vis, f"{label} ({conf:.2f})", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                return av.VideoFrame.from_ndarray(vis, format="bgr24")

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    with right_col:
        webrtc_ctx = webrtc_streamer(
            key=f"webrtc-{selected_model_name}",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: GenericProcessor(MODEL_STATE, DEVICE, IMGSZ, CONF_THRESH),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    try:
        time.sleep(0.2)
        while True:
            current = int(MODEL_STATE.get("current_count", 0))
            cumulative = int(MODEL_STATE.get("cumulative_total", 0))
            cur_card_ph.markdown(card_html("CURRENT COUNT", current, "#2b6cb0"), unsafe_allow_html=True)
            cum_card_ph.markdown(card_html("CUMULATIVE COUNT", cumulative, "#dd6b20"), unsafe_allow_html=True)
            debug_ph.text(f"Device: {DEVICE}  |  imgsz: {IMGSZ}  |  conf: {CONF_THRESH:.2f}")
            time.sleep(0.2)
    except Exception as e:
        st.error(f"Realtime dashboard loop error: {e}")

# ----------------- VIDEO DEMO -----------------
elif mode == "Video Demo":
    with right_col:
        st.subheader(f"{selected_model_name} — upload a file and press Play")
        uploaded = st.file_uploader("Upload a video file (mp4, avi, mkv, mov)", type=["mp4", "avi", "mkv", "mov"])
        c1, c2 = st.columns(2)
        play_btn = c1.button("Play", key=f"play_{selected_model_name}")
        reset_btn = c2.button("Reset counts", key=f"reset_counts_{selected_model_name}")
        frame_ph = st.empty()
        status_ph = st.empty()

    if reset_btn:
        MODEL_STATE["cumulative_total"] = 0
        MODEL_STATE["prev_max_seen"] = 0
        MODEL_STATE["current_count"] = 0
        cur_card_ph.markdown(card_html("CURRENT COUNT", MODEL_STATE["current_count"], "#2b6cb0"), unsafe_allow_html=True)
        cum_card_ph.markdown(card_html("CUMULATIVE COUNT", MODEL_STATE["cumulative_total"], "#dd6b20"), unsafe_allow_html=True)
        status_ph.info("Counts reset.")

    # single Stop button widget (unique key to avoid duplicate-element errors)
    stop_key = f"stop_play_{selected_model_name}"
    stop_btn = st.button("Stop", key=stop_key)

    if uploaded is not None and play_btn:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tfile.write(uploaded.read())
        tfile.flush()
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        if not cap.isOpened():
            st.error("Could not open uploaded video.")
        else:
            dev_arg = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"

            # Prepare model
            model_obj = None
            if _IS_BANANA and banana_counter_driver is not None:
                try:
                    model_obj = banana_counter_driver.get_banana_model()
                    try:
                        model_obj.to(dev_arg)
                    except Exception:
                        pass
                except Exception as e:
                    status_ph.error(str(e))
            elif _IS_TREE and TREE_MODEL_PATH and Path(TREE_MODEL_PATH).exists():
                try:
                    model_obj = YOLO(TREE_MODEL_PATH)
                    try:
                        model_obj.to(dev_arg)
                    except Exception:
                        pass
                except Exception as e:
                    status_ph.error(str(e))
            else:
                try:
                    model_obj = selected_mod.get_model()
                except Exception as e:
                    status_ph.error(f"Could not load model: {e}")

            if model_obj is None:
                status_ph.error("Model not loaded — cannot run prediction.")
            else:
                status_ph.info(f"Model loaded. Device preference: {DEVICE} (torch.cuda.is_available()={cuda_avail})")
                prev_time = time.time()
                fps_smooth = None
                vid_fps = cap.get(cv2.CAP_PROP_FPS)
                if not vid_fps or vid_fps <= 0:
                    vid_fps = 25.0
                frame_delay = 1.0 / vid_fps
                frame_idx = 0

                # Attempt optimized streaming predict for Ultralytics YOLO models
                used_stream_path = False
                try:
                    if _IS_BANANA and banana_counter_driver is not None:
                        try:
                            results_gen = model_obj.predict(
                                source=tfile.name,
                                imgsz=IMGSZ,
                                conf=CONF_THRESH,
                                iou=0.5,
                                device=dev_arg,
                                stream=True,
                                verbose=False,
                            )
                            prev_time = time.time()
                            fps_smooth = None
                            frame_idx = 0
                            for r in results_gen:
                                frame_idx += 1
                                # check Stop
                                if stop_btn:
                                    status_ph.info("Stopped by user.")
                                    break

                                centers, vis = _yolo_results_to_centers_vis(r, CONF_THRESH)
                                if vis is None:
                                    vis = getattr(r, "orig_img", None)
                                if vis is None:
                                    vis = np.zeros((480, 640, 3), dtype=np.uint8)

                                now = time.time()
                                dt = now - prev_time if now - prev_time > 1e-6 else 1e-6
                                fps = 1.0 / dt
                                prev_time = now
                                fps_smooth = fps if fps_smooth is None else (0.85 * fps_smooth + 0.15 * fps)

                                annotate_and_update(MODEL_STATE, vis, centers, fps_smooth)

                                frame_ph.image(cv2.cvtColor(vis.astype("uint8"), cv2.COLOR_BGR2RGB), channels="RGB")
                                cur_card_ph.markdown(card_html("CURRENT COUNT", int(MODEL_STATE.get("current_count", 0)), "#2b6cb0"), unsafe_allow_html=True)
                                cum_card_ph.markdown(card_html("CUMULATIVE COUNT", int(MODEL_STATE.get("cumulative_total", 0)), "#dd6b20"), unsafe_allow_html=True)
                                status_ph.text(f"Frame {frame_idx} (stream)  |  FPS(est): {fps_smooth:.1f}")

                            used_stream_path = True
                            status_ph.info("Finished playback (stream mode).")
                        except Exception as e:
                            status_ph.warning(f"Streaming predict failed, falling back to frame-by-frame (error: {e})")
                            used_stream_path = False

                    if not used_stream_path and _IS_TREE and model_obj is not None:
                        try:
                            results_gen = model_obj.predict(
                                source=tfile.name,
                                conf=0.6,
                                iou=0.45,
                                classes=[0],
                                max_det=500,
                                imgsz=IMGSZ,
                                device=dev_arg,
                                stream=True,
                                verbose=False,
                            )
                            prev_time = time.time()
                            fps_smooth = None
                            frame_idx = 0
                            for r in results_gen:
                                frame_idx += 1
                                # check Stop
                                if stop_btn:
                                    status_ph.info("Stopped by user.")
                                    break

                                boxes = r.boxes.xywh.cpu().numpy() if getattr(r, "boxes", None) is not None else np.zeros((0, 4))
                                scores = r.boxes.conf.cpu().numpy() if getattr(r, "boxes", None) is not None else np.zeros((0,))
                                centers = []
                                for (x, y, w, h), _score in zip(boxes, scores):
                                    if 30 < w < 300 and 30 < h < 300:
                                        centers.append((int(x), int(y)))

                                vis = getattr(r, "orig_img", None)
                                if vis is None:
                                    vis = np.zeros((480, 640, 3), dtype=np.uint8)

                                now = time.time()
                                dt = now - prev_time if now - prev_time > 1e-6 else 1e-6
                                fps = 1.0 / dt
                                prev_time = now
                                fps_smooth = fps if fps_smooth is None else (0.85 * fps_smooth + 0.15 * fps)

                                annotate_and_update(MODEL_STATE, vis, centers, fps_smooth)

                                frame_ph.image(cv2.cvtColor(vis.astype("uint8"), cv2.COLOR_BGR2RGB), channels="RGB")
                                cur_card_ph.markdown(card_html("CURRENT COUNT", int(MODEL_STATE.get("current_count", 0)), "#2b6cb0"), unsafe_allow_html=True)
                                cum_card_ph.markdown(card_html("CUMULATIVE COUNT", int(MODEL_STATE.get("cumulative_total", 0)), "#dd6b20"), unsafe_allow_html=True)
                                status_ph.text(f"Frame {frame_idx} (stream)  |  FPS(est): {fps_smooth:.1f}")

                            used_stream_path = True
                            status_ph.info("Finished playback (stream mode).")
                        except Exception as e:
                            status_ph.warning(f"Streaming predict for tree failed, falling back (error: {e})")
                            used_stream_path = False

                except Exception:
                    # If anything unexpected happens in the streaming attempt, fall back to frame loop
                    used_stream_path = False

                # If streaming path wasn't used (or failed), fallback to the original per-frame processing loop
                if not used_stream_path:
                    frame_idx = 0
                    prev_time = time.time()
                    fps_smooth = None
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    while cap.isOpened():
                        # check stop button each iteration
                        if stop_btn:
                            status_ph.info("Stopped by user.")
                            break

                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_idx += 1
                        now = time.time()

                        # --- inference depending on page ---
                        if _IS_BANANA and banana_counter_driver is not None:
                            try:
                                results = model_obj.predict(
                                    source=frame,
                                    imgsz=IMGSZ,
                                    conf=CONF_THRESH,
                                    iou=0.5,
                                    device=dev_arg,
                                    verbose=False,
                                )
                            except Exception:
                                try:
                                    results = model_obj.predict(
                                        source=frame,
                                        imgsz=IMGSZ,
                                        conf=CONF_THRESH,
                                        iou=0.5,
                                        device="cpu",
                                        verbose=False,
                                    )
                                except Exception as e:
                                    status_ph.error(f"Prediction failed on frame {frame_idx}: {e}")
                                    frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                                    continue

                            r = results[0]
                            centers, vis = _yolo_results_to_centers_vis(r, CONF_THRESH)
                            if vis is None:
                                vis = frame.copy()

                            dt = now - prev_time if now - prev_time > 1e-6 else 1e-6
                            fps = 1.0 / dt
                            prev_time = now
                            fps_smooth = fps if fps_smooth is None else (0.85 * fps_smooth + 0.15 * fps)
                            annotate_and_update(MODEL_STATE, vis, centers, fps_smooth)

                        elif _IS_TREE and model_obj is not None:
                            try:
                                results = model_obj.predict(
                                    source=frame,
                                    conf=0.6,
                                    iou=0.45,
                                    classes=[0],
                                    max_det=500,
                                    imgsz=IMGSZ,
                                    device=dev_arg,
                                    verbose=False,
                                )
                            except Exception:
                                try:
                                    results = model_obj.predict(
                                        source=frame,
                                        conf=0.6,
                                        iou=0.45,
                                        classes=[0],
                                        max_det=500,
                                        imgsz=IMGSZ,
                                        device="cpu",
                                        verbose=False,
                                    )
                                except Exception as e:
                                    status_ph.error(f"Prediction failed on frame {frame_idx}: {e}")
                                    frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                                    continue

                            r = results[0]
                            boxes = r.boxes.xywh.cpu().numpy() if getattr(r, "boxes", None) is not None else np.zeros((0, 4))
                            scores = r.boxes.conf.cpu().numpy() if getattr(r, "boxes", None) is not None else np.zeros((0,))
                            centers = []
                            for (x, y, w, h), _score in zip(boxes, scores):
                                if 30 < w < 300 and 30 < h < 300:
                                    centers.append((int(x), int(y)))
                            vis = getattr(r, "orig_img", None)
                            if vis is None:
                                vis = frame.copy()

                            dt = now - prev_time if now - prev_time > 1e-6 else 1e-6
                            fps = 1.0 / dt
                            prev_time = now
                            fps_smooth = fps if fps_smooth is None else (0.85 * fps_smooth + 0.15 * fps)
                            annotate_and_update(MODEL_STATE, vis, centers, fps_smooth)

                        else:
                            try:
                                results = selected_mod.run_on_image(model_obj, frame, conf_thresh=CONF_THRESH, imgsz=IMGSZ)
                            except Exception as e:
                                status_ph.error(f"Prediction failed on frame {frame_idx}: {e}")
                                frame_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                                continue

                            dt = now - prev_time if now - prev_time > 1e-6 else 1e-6
                            fps = 1.0 / dt
                            prev_time = now
                            fps_smooth = fps if fps_smooth is None else (0.85 * fps_smooth + 0.15 * fps)

                            if results.get("type") == "detection":
                                vis = results.get("vis")
                                if vis is None:
                                    vis = frame.copy()
                                centers = results.get("centers", []) or []
                                total_override = results.get("count", None)
                                annotate_and_update(MODEL_STATE, vis, centers, fps_smooth, total_override=total_override)
                            elif results.get("type") == "classification":
                                vis = results.get("vis")
                                if vis is None:
                                    vis = frame.copy()
                                label = results.get("label", "unknown")
                                conf = float(results.get("confidence", 0.0))
                                MODEL_STATE["last_label"] = label
                                MODEL_STATE["last_confidence"] = conf
                                cv2.rectangle(vis, (10, 10), (600, 72), (0, 0, 0), -1)
                                cv2.putText(vis, f"{label} ({conf:.2f})", (20, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                            else:
                                vis = frame.copy()

                        # display frame
                        frame_ph.image(cv2.cvtColor(vis.astype("uint8"), cv2.COLOR_BGR2RGB), channels="RGB")

                        # update cards/status
                        cur_card_ph.markdown(card_html("CURRENT COUNT", int(MODEL_STATE.get("current_count", 0)), "#2b6cb0"), unsafe_allow_html=True)
                        cum_card_ph.markdown(card_html("CUMULATIVE COUNT", int(MODEL_STATE.get("cumulative_total", 0)), "#dd6b20"), unsafe_allow_html=True)
                        status_ph.text(f"Frame {frame_idx}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)}  |  FPS(est): {fps_smooth:.1f}")

                        # slightly under-sleep to let UI update
                        time.sleep(frame_delay * 0.95)

                    cap.release()
                    status_ph.info("Finished playback.")
# ----------------- DRONE -----------------
elif mode == "Drone":
    with right_col:
        st.subheader("🚁 Drone Live Stream (MediaMTX)")

        drone_url = st.text_input(
            "Drone stream URL (RTSP)",
            value="rtsp://10.57.76.135:8554/live/drone"
        )

        c1, c2 = st.columns(2)
        start_btn = c1.button("Start Drone")
        stop_btn = c2.button("Stop Drone")

        frame_ph = st.empty()
        status_ph = st.empty()

    if start_btn:
        st.session_state["drone_running"] = True

    if stop_btn:
        st.session_state["drone_running"] = False

    if st.session_state.get("drone_running", False):
        cap = None

        # ---- RTSP retry logic ----
        for _ in range(10):
            cap = cv2.VideoCapture(drone_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if cap.isOpened():
                break
            time.sleep(0.5)

        if not cap or not cap.isOpened():
            status_ph.error("❌ Could not open drone stream. Start DJI Fly livestream first.")
            st.session_state["drone_running"] = False

        else:
            status_ph.success("✅ Drone stream connected via MediaMTX")

            dev_arg = "cuda" if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"

            # ---- MODEL LOAD (unchanged) ----
            if _IS_BANANA and banana_counter_driver is not None:
                model_obj = banana_counter_driver.get_banana_model()
                try:
                    model_obj.to(dev_arg)
                except Exception:
                    pass

            elif _IS_TREE and TREE_MODEL_PATH and Path(TREE_MODEL_PATH).exists():
                model_obj = YOLO(TREE_MODEL_PATH)
                try:
                    model_obj.to(dev_arg)
                except Exception:
                    pass

            else:
                model_obj = selected_mod.get_model()

            prev_time = time.time()
            fps_smooth = None

            while cap.isOpened() and st.session_state.get("drone_running", False):
                ret, frame = cap.read()
                if not ret:
                    status_ph.warning("⚠️ Drone stream lost")
                    break

                # ---- INFERENCE (unchanged) ----
                if _IS_BANANA and banana_counter_driver is not None:
                    results = model_obj.predict(
                        source=frame,
                        imgsz=IMGSZ,
                        conf=CONF_THRESH,
                        iou=0.5,
                        device=dev_arg,
                        verbose=False,
                    )
                    r = results[0]
                    centers, vis = _yolo_results_to_centers_vis(r, CONF_THRESH)
                    if vis is None:
                        vis = frame.copy()

                elif _IS_TREE:
                    results = model_obj.predict(
                        source=frame,
                        conf=0.6,
                        iou=0.45,
                        classes=[0],
                        imgsz=IMGSZ,
                        device=dev_arg,
                        verbose=False,
                    )
                    r = results[0]
                    boxes = r.boxes.xywh.cpu().numpy()
                    centers = [(int(x), int(y)) for x, y, w, h in boxes if 30 < w < 300]
                    vis = r.orig_img if r.orig_img is not None else frame.copy()

                else:
                    results = selected_mod.run_on_image(
                        model_obj, frame,
                        conf_thresh=CONF_THRESH,
                        imgsz=IMGSZ
                    )
                    vis = results.get("vis", frame.copy())
                    centers = results.get("centers", [])

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                fps_smooth = fps if fps_smooth is None else (0.85 * fps_smooth + 0.15 * fps)

                annotate_and_update(MODEL_STATE, vis, centers, fps_smooth)

                frame_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), channels="RGB")

            cap.release()
            status_ph.info("🛑 Drone stream stopped")

# Bottom reset for this model
st.markdown("---")
if st.button("Reset counts (this page)", key=f"reset_bottom_{selected_model_name}"):
    MODEL_STATE["cumulative_total"] = 0
    MODEL_STATE["prev_max_seen"] = 0
    MODEL_STATE["current_count"] = 0
    cur_card_ph.markdown(card_html("CURRENT COUNT", MODEL_STATE["current_count"], "#2b6cb0"), unsafe_allow_html=True)
    cum_card_ph.markdown(card_html("CUMULATIVE COUNT", MODEL_STATE["cumulative_total"], "#dd6b20"), unsafe_allow_html=True)
    st.success("Counts reset for this model.")
