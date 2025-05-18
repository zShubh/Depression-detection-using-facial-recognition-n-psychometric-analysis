import cv2 # type: ignore
import threading

running = False
cap = None

def start_camera():
    global cap, running
    running = True
    cap = cv2.VideoCapture(0)

    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Optional: You can save frames or perform emotion detection here
        cv2.imshow("Live Camera - Press Submit to Close", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def stop_camera():
    global running
    running = False
