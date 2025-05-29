import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib
import os

clf = joblib.load('pose_classifier.pkl')
le = joblib.load('label_encoder.pkl')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def extract_landmarks(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        raise ValueError("No pose detected.")

    landmarks = results.pose_landmarks.landmark
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y]) 

    return features


def predict_from_images(image_paths):
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        try:
            features = extract_landmarks(image_path)
            df = pd.DataFrame([features])
            y_pred = clf.predict(df)
            pose_name = le.inverse_transform(y_pred)[0]
            print(f"Predicted Pose: {pose_name}")
        except Exception as e:
            print(f"Error for {image_path}: {e}")

if __name__ == "__main__":
    image_paths = [
        '/Users/shankarsingh/Desktop/butterfly-pose.jpg'
    ]

    predict_from_images(image_paths)
