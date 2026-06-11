# Project Analysis: Division

## Overview
The `division` project is a computer vision application tailored for agricultural monitoring, specifically focused on **banana crops**. It provides a Streamlit-based interface for detecting, counting, and analyzing agricultural assets using deep learning models (YOLO).

## Project Structure
```text
division/
├── drone.py                    # Standalone script for viewing drone RTMP feed
├── integration/
│   ├── app.py                  # Main Streamlit application entry point
│   ├── dependencies.txt        # Python dependencies list
│   ├── drivers/                # Logic for individual models (Banana, Tree, Disease, etc.)
│   │   ├── banana_counter_driver.py
│   │   ├── tree_counting_driver.py
│   │   └── ...
│   ├── models/                 # PyTorch model weights (.pt, .pth)
│   └── snapshots/              # Directory for saving outputs (presumed)
└── .vscode/                    # VS Code settings
```

## Key Components

### 1. Main Application (`integration/app.py`)
This is the core of the project, built with **Streamlit**. It offers a dashboard with three main modes of operation:
*   **Realtime**: Uses `streamlit-webrtc` to process video feeds (e.g., webcam) in real-time within the browser.
*   **Video Demo**: Allows users to upload video files (MP4, AVI, MKV, MOV) for processing.
*   **Drone**: Connects to a live RTMP stream (`rtmp://10.17.175.233:1935/live/drone`) to process drone footage.

**Features:**
*   **Model Selection**: Dynamically loads different models (Banana Counter, Tree Counter, Ripeness, Disease, etc.) based on user selection.
*   **Hardware Acceleration**: Supports switching between CPU and CUDA (GPU) for inference.
*   **Configurable Parameters**: Users can adjust confidence thresholds and inference image size (`imgsz`).
*   **Visualization**: Annotates video frames with bounding boxes, segmentation masks, and counts. It also displays real-time statistics (Current Count, Cumulative Count).

### 2. Drone Stream Viewer (`drone.py`)
A simple utility script using **OpenCV** to connect to and display the raw video feed from the drone's RTMP server. It serves as a quick way to verify the stream connectivity without running the full web app.

### 3. Drivers (`integration/drivers/`)
These modules encapsulate the specific logic for each model type.
*   **`banana_counter_driver.py`**:
    *   Implements lazy loading of the `banana_counter.pt` YOLO model.
    *   Contains logic to count bananas, handling both segmentation masks and bounding boxes.
    *   Draws visual indicators (dots and IDs) on the images.
*   Other drivers likely function similarly, abstracting the model-specific post-processing.

### 4. Models (`integration/models/`)
Contains the heavy weight files for the neural networks.
*   `banana_counter.pt`: YOLO model for counting bananas.
*   `tree_counter.pt`: Model for counting trees.
*   Other models for disease detection, variety classification, and ripeness.

## Tech Stack
*   **Language**: Python
*   **Web Framework**: Streamlit
*   **Computer Vision**: OpenCV, PyTorch, Ultralytics YOLO
*   **Real-time Streaming**: `streamlit-webrtc`, PyAV
*   **Data Processing**: NumPy

## Recommendations / Observations
*   **Hardcoded IP**: The RTMP URL `rtmp://10.17.175.233:1935/live/drone` is hardcoded in both `drone.py` and `app.py`. It might be beneficial to move this to a configuration file or a sidebar input field for flexibility.
*   **Error Handling**: The application includes retry logic for the drone stream connection, which is robust.
*   **Modular Design**: The use of "drivers" is a good design choice, separating the model-specific logic from the main UI code.
