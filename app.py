import streamlit as st
import cv2
import threading
import time
from deepface import DeepFace
import numpy as np

# Global variables
emotion_counts = {
    'happy': 0,
    'sad': 0,
    'angry': 0,
    'surprise': 0,
    'neutral': 0,
    'fear': 0,
    'disgust': 0
}
camera_running = False
last_emotion = ""
camera_thread = None
final_emotions = {}

# Emotion detection function
def detect_emotion():
    global camera_running, emotion_counts, last_emotion
    cap = cv2.VideoCapture(0)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            emotion_counts[emotion] += 1
            last_emotion = emotion
        except:
            pass

        cv2.putText(frame, f"Emotion: {last_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Camera (Press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Streamlit App UI
st.set_page_config(page_title="Depression Detection", page_icon="🧠")
st.title("🧠 Depression Detection Portal")
st.markdown("Welcome! This tool uses facial recognition and psychometric analysis to assess mental well-being.")

# Step 1: Camera Start
st.header("Step 1: Facial Emotion Recognition")

if st.button("📸 Start Emotion Detection Camera"):
    st.warning("Camera started. Press 'q' in the camera window to manually stop it.")
    emotion_counts = {k: 0 for k in emotion_counts}  # Reset
    camera_running = True
    camera_thread = threading.Thread(target=detect_emotion)
    camera_thread.start()

# Step 2: Questionnaire
st.header("Step 2: Psychometric Questionnaire")
questions = [
    ("How often do you feel sad or down without a clear reason?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("How satisfied are you with your current quality of life?", ["Very Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Very Dissatisfied"]),
    ("Do you find it difficult to concentrate on tasks?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("How frequently do you experience feelings of hopelessness?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you feel tired or fatigued even after adequate sleep?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("How often do you enjoy activities that you used to find pleasurable?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you find yourself overthinking past mistakes or failures?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("How would you rate your self-esteem?", ["Very High", "High", "Moderate", "Low", "Very Low"]),
    ("Do you struggle with making decisions, even for small matters?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("How often do you feel restless or unable to relax?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you feel detached or distant from your surroundings?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you have difficulty controlling negative thoughts?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you feel irritable or easily annoyed?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you have a reduced appetite or overeating tendencies?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("How often do you experience physical symptoms (e.g., headaches, muscle tension) without a clear cause?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you feel like you are moving or speaking slower than usual?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("How often do you feel overwhelmed by day-to-day activities?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you find it challenging to feel optimistic about the future?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("How often do you think about self-harm or suicide?", ["Always", "Often", "Sometimes", "Rarely", "Never"]),
    ("Do you feel supported by friends or family when you need help?", ["Always", "Often", "Sometimes", "Rarely", "Never"])
]

option_scores = {
    "Always": 5,
    "Very Dissatisfied": 5,
    "Often": 4,
    "Dissatisfied": 4,
    "Sometimes": 3,
    "Neutral": 3,
    "Rarely": 2,
    "Satisfied": 2,
    "Never": 1,
    "Very Satisfied": 1,
    "Very High": 1,
    "High": 2,
    "Moderate": 3,
    "Low": 4,
    "Very Low": 5
}

responses = []
score = 0

with st.form("psychometric_form"):
    for idx, (q, opts) in enumerate(questions):
        response = st.radio(f"{idx + 1}. {q}", opts, key=idx)
        responses.append(response)
        score += option_scores.get(response, 0)
    submitted = st.form_submit_button("Submit")

# ✅ Handle submission
if submitted:
    camera_running = False  # Stop the camera loop
    if camera_thread:
        camera_thread.join()  # Wait until camera stops
    final_emotions = emotion_counts.copy()  # Capture summary

    st.success("✔️ Questionnaire Submitted Successfully!")

    st.subheader("📊 Analysis")
    st.write(f"**Total Score:** {score}")

    # Depression level interpretation
    if score <= 18:
        level = "No Significant Depression"
        rec = "General wellness monitoring"
    elif score <= 37:
        level = "Mild Depression"
        rec = "Lifestyle changes, counseling"
    elif score <= 62:
        level = "Moderate Depression"
        rec = "Therapy, psychological support"
    elif score <= 82:
        level = "Severe Depression"
        rec = "Clinical intervention, therapy, and potential medication"
    else:
        level = "Critical Depression"
        rec = "Emergency psychiatric care, potential hospitalization"

    st.write(f"**Depression Level:** {level}")
    st.write(f"**Recommendation:** {rec}")
