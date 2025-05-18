import cv2 # type: ignore
import time
from deepface import DeepFace # type: ignore

def capture_emotions():
    cap = cv2.VideoCapture(0)
    captured = 0
    results = []

    if not cap.isOpened():
        print("Error: Cannot access webcam")
        return

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Press 'q' to quit", frame)

        # Capture one frame every 5 seconds (max 5 snapshots)
        if time.time() - start_time > 5 and captured < 5:
            img_path = f"frame_{captured}.jpg"
            cv2.imwrite(img_path, frame)

            try:
                result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
                results.append(result[0]['emotion'])
                print(f"Snapshot {captured + 1} analyzed: {result[0]['emotion']}")
            except Exception as e:
                print("Emotion analysis failed:", e)

            captured += 1
            start_time = time.time()

        # End if user presses 'q' or 5 captures are done
        if cv2.waitKey(1) & 0xFF == ord('q') or captured >= 5:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save results
    with open("emotion_results.txt", "w") as f:
        for idx, r in enumerate(results):
            f.write(f"Snapshot {idx + 1}: {r}\n")

    print("Emotion analysis completed and saved to 'emotion_results.txt'")

if __name__ == "__main__":
    capture_emotions()
