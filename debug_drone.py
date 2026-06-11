
import cv2
import time
import subprocess
import platform
import re
import sys

def ping_ip(ip):
    """
    Returns True if host (str) responds to a ping request.
    """
    print(f"[INFO] Pinging {ip}...")
    param = '-n' if platform.system().lower() == 'windows' else '-c'
    command = ['ping', param, '1', ip]
    
    try:
        # Run ping command
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output.returncode == 0
    except Exception as e:
        print(f"[ERROR] Ping failed with exception: {e}")
        return False

def check_stream(url):
    print(f"\n[INFO] Attempting to connect to video stream: {url}")
    print("[INFO] This might take up to 10 seconds...")
    
    cap = cv2.VideoCapture(url)
    
    if not cap.isOpened():
        print(f"[FAILURE] Could not open video capture at {url}")
        print("Possible causes:")
        print("  1. The drone is functionality offline or not streaming.")
        print("  2. The IP address is incorrect.")
        print("  3. Firewall blocking the connection.")
        print("  4. Missing ffmpeg or opencv backend support (unlikely if 'pip install opencv-python' was used).")
        return False
    
    print("[SUCCESS] Video capture opened successfully!")
    
    # Try reading a frame
    ret, frame = cap.read()
    if ret:
        print(f"[SUCCESS] Read frame correctly. Resolution: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print("[WARNING] Connection opened but failed to read the first frame.")
    
    cap.release()
    return True

def extract_ip(url):
    # Regex to extract IP from RTMP/HTTP url
    match = re.search(r'//([\d\.]+)', url)
    if match:
        return match.group(1)
    return None

if __name__ == "__main__":
    default_url = "rtmp://10.17.175.233:1935/live/drone"
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        print(f"Usage: python debug_drone.py [url]")
        print(f"Using default URL: {default_url}")
        url = default_url

    # 1. Check Network Connectivity
    ip = extract_ip(url)
    if ip:
        alive = ping_ip(ip)
        if alive:
            print(f"[SUCCESS] Host {ip} is reachable via ping.")
        else:
            print(f"[WARNING] Host {ip} is NOT reachable via ping. Stream might still work if ICMP is blocked, but this is a bad sign.")
    else:
        print("[INFO] Could not extract IP to ping (not an IP-based URL?). Skipping ping.")

    # 2. Check Video Stream
    check_stream(url)
    
    print("\n[INFO] Debug connected finished.")
