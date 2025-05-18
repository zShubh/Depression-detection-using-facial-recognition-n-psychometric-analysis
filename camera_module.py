import cv2 # type: ignore
import time
import os

# Ensure snapshot folder exists
output_folder = "snapshots"
os.makedirs(output_folder, exist_ok=True)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

start_time = time.time()
duration = 60  # 1 minute; increase this if needed

while int(time.time() - start_time) < duration:
    ret, frame = cap.read()
    if not ret:
        break

    filename = os.path.join(output_folder, f"snapshot_{count}.jpg")
    cv2.imwrite(filename, frame)
    count += 1
    time.sleep(5)  # take a snapshot every 5 seconds

cap.release()
cv2.destroyAllWindows()
