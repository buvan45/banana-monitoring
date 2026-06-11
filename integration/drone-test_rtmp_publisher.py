# drone.py
"""
Drone / Camera → MediaMTX RTMP Publisher

- Captures camera (webcam / drone / phone)
- Pushes frames to MediaMTX via RTMP
- No Flask
- No Streamlit
- Safe to launch via subprocess
"""

import cv2
import subprocess
import sys
import time

# ================= CONFIG =================
CAMERA_INDEX = 0            # Change if needed
WIDTH = 640
HEIGHT = 480
FPS = 30

RTMP_URL = "rtmp://127.0.0.1:1935/live/stream"
# ==========================================


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("❌ Could not open camera")
        sys.exit(1)

    # FFmpeg command to push raw frames to RTMP
    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-g", "30",
        "-keyint_min", "30",
        "-f", "flv",
        RTMP_URL
    ]


    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print("🚁 Drone stream publishing to MediaMTX")
    print(f"📡 {RTMP_URL}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            process.stdin.write(frame.tobytes())

    except KeyboardInterrupt:
        print("🛑 Drone stream stopped")

    finally:
        cap.release()
        process.stdin.close()
        process.wait()


if __name__ == "__main__":
    main()
