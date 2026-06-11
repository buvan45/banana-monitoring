import cv2

cap = cv2.VideoCapture("rtmp://10.17.175.233:1935/live/drone")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame")
        break
    cv2.imshow("Drone Feed", frame)
    if cv2.waitKey(1) == 27:
        break
