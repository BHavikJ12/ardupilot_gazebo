import cv2
import time

VIDEO_PATH = "/home/bhvaik/Downloads/watermarked_preview.mp4"

# Create KCF tracker
tracker = cv2.TrackerKCF_create()

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Error: Cannot read first frame.")
    exit()

# Select ROI (Bounding box)
bbox = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, bbox)

# FPS calculation
fps = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Update tracker
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Display
    cv2.imshow("KCF Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
