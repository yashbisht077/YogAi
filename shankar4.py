import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from gtts import gTTS
import pygame
import time
from twilio.rest import Client

target_pose = "ardha uttanasana"
name = "Mayank"
recipient_number = "+919696172620"
twilio_sid = "AC4362d66be441f1a8b204ce32bc98cec3"
twilio_token = "da6ce5fed34a5e412eab33ed3584f70b"
twilio_whatsapp_number = "whatsapp:+14155238886"

last_spoken_time = 0
speak_interval = 10
pygame.mixer.init()

def speak_feedback(text):
    global last_spoken_time
    if time.time() - last_spoken_time >= speak_interval:
        try:
            tts = gTTS(text)
            filename = "feedback.mp3"
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            last_spoken_time = time.time()
        except Exception as e:
            print("Voice feedback error:", e)

correction_suggestions = {
    'left_elbow': "Raise your left arm slightly towards the ghost position.",
    'right_elbow': "Adjust your right arm for a straighter elbow like the ghost pose.",
    'left_knee': "Move your left knee to match the ghost pose.",
    'right_knee': "Align your right knee with the ghost pose."
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

joint_definitions = {
    'left_elbow': [11, 13, 15],
    'right_elbow': [12, 14, 16],
    'left_knee': [23, 25, 27],
    'right_knee': [24, 26, 28]
}

df = pd.read_csv('landmarks_dataset.csv')
pose_df = df[df['label'] == target_pose]

def compute_angles_from_sample(sample_row):
    landmarks = [(sample_row[f'{i}_x'], sample_row[f'{i}_y']) for i in range(33)]
    angles = {}
    for joint, (a, b, c) in joint_definitions.items():
        pt1, pt2, pt3 = landmarks[a], landmarks[b], landmarks[c]
        angles[joint] = calculate_angle(pt1, pt2, pt3)
    return angles, landmarks

ghost_sample = pose_df.iloc[0]
ideal_angles, ghost_landmarks = compute_angles_from_sample(ghost_sample)
print(f"Loaded ideal angles for {target_pose.upper()}:\n", ideal_angles)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def send_whatsapp_message(to_number, message):
    client = Client(twilio_sid, twilio_token)
    client.messages.create(
        body=message,
        from_=twilio_whatsapp_number,
        to=f"whatsapp:{to_number}"
    )

def get_live_joint_angles(landmarks):
    angles = {}
    for joint, (a, b, c) in joint_definitions.items():
        pt1 = [landmarks[a].x, landmarks[a].y]
        pt2 = [landmarks[b].x, landmarks[b].y]
        pt3 = [landmarks[c].x, landmarks[c].y]
        angles[joint] = calculate_angle(pt1, pt2, pt3)
    return angles

def calculate_deviation(current, ideal):
    return abs(current - ideal) / ideal * 100

cap = cv2.VideoCapture(0)
final_accuracy = None
last_suggestions = []
start_time = time.time()
show_ghost_pose = False

print(f"Target Pose: {target_pose.upper()} — Live Accuracy Evaluation Started")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        angles = get_live_joint_angles(landmarks)
        deviations = []
        suggestions = []
        y = 30

        if time.time() - start_time >= 3:
            show_ghost_pose = True

        if show_ghost_pose:
            ghost_frame = np.zeros_like(frame, dtype=np.uint8)
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(ghost_landmarks) and end_idx < len(ghost_landmarks):
                    x1 = int(ghost_landmarks[start_idx][0] * frame.shape[1])
                    y1 = int(ghost_landmarks[start_idx][1] * frame.shape[0])
                    x2 = int(ghost_landmarks[end_idx][0] * frame.shape[1])
                    y2 = int(ghost_landmarks[end_idx][1] * frame.shape[0])
                    cv2.line(ghost_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            for x, y_ in ghost_landmarks:
                x = int(x * frame.shape[1])
                y_ = int(y_ * frame.shape[0])
                cv2.circle(ghost_frame, (x, y_), 5, (255, 255, 255), -1)

            frame = cv2.addWeighted(frame, 1, ghost_frame, 0.3, 0)

        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            x1 = int(landmarks[start_idx].x * frame.shape[1])
            y1 = int(landmarks[start_idx].y * frame.shape[0])
            x2 = int(landmarks[end_idx].x * frame.shape[1])
            y2 = int(landmarks[end_idx].y * frame.shape[0])
            is_deviated = False

            for joint, (a, b, c) in joint_definitions.items():
                if b in (start_idx, end_idx) and joint in angles:
                    dev = calculate_deviation(angles[joint], ideal_angles[joint])
                    if dev > 15:
                        is_deviated = True
                        break

            color = (0, 0, 255) if is_deviated else (0, 255, 0)
            cv2.line(frame, (x1, y1), (x2, y2), color, 3)

        for joint, ideal in ideal_angles.items():
            actual = angles.get(joint)
            if actual is not None:
                dev = calculate_deviation(actual, ideal)
                deviations.append(dev)

                color = (0, 255, 0) if dev < 15 else (0, 0, 255)
                cv2.putText(frame, f'{joint}: {actual:.1f}° | Dev: {dev:.1f}%', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                a_id, b_id, c_id = joint_definitions[joint]
                cx = int(landmarks[b_id].x * frame.shape[1])
                cy = int(landmarks[b_id].y * frame.shape[0])
                cv2.putText(frame, str(int(actual)), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y += 30

                if dev > 20 and joint in correction_suggestions:
                    suggestions.append(correction_suggestions[joint])

        if deviations:
            avg_dev = np.mean(deviations)
            accuracy = max(0, 100 - avg_dev)
            final_accuracy = accuracy
            cv2.putText(frame, f'Pose Accuracy: {accuracy:.1f}%', (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            if suggestions:
                last_suggestions = suggestions
                feedback_text = " ".join(suggestions[:2])
                cv2.putText(frame, "Suggestion: " + suggestions[0], (10, y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                speak_feedback(feedback_text)


    cv2.imshow("Pose Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
x = '''🧘 Yoga Pose Summary — Baddha Konasana
🎯 Accuracy: 87%
•⁠  ⁠Left Elbow: 9.27% deviation
•⁠  ⁠Right Elbow: 12.88% deviation
•⁠  ⁠Left Knee: 49.38% deviation
•⁠  ⁠Right Knee: 32.71% deviation
Keep practicing and stay focused! 💪'''

time.sleep(10)
send_whatsapp_message(recipient_number,x)