
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
from collections import deque, Counter

# ---------------- CONFIGURATION ----------------

CAMERA_INDEX = 0
HAND_MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "models/gesture_model.joblib"

CONFIDENCE_THRESHOLD = 0.65
SMOOTHING_WINDOW = 9
MIN_VOTES = 5

# ---------------- CHECK FILES ----------------

for path in [HAND_MODEL_PATH, CLASSIFIER_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Complete Steps 3-5 and check your models folder."
        )

# ---------------- LOAD CLASSIFIER ----------------

artifact = joblib.load(CLASSIFIER_PATH)

if not isinstance(artifact, dict):
    raise ValueError(
        "The saved model is not in the expected dictionary format. "
        "Run step5_train_model.py again."
    )

if "model" not in artifact:
    raise ValueError(
        "The saved model artifact has no 'model' entry. "
        "Run step5_train_model.py again."
    )

model = artifact["model"]

if artifact.get("feature_count") != 63 or getattr(
    model, "n_features_in_", None
) != 63:
    raise ValueError(
        f"Feature mismatch: artifact feature_count="
        f"{artifact.get('feature_count')}, "
        f"model expects {getattr(model, 'n_features_in_', 'unknown')} "
        "features. Retrain using step5_train_model.py."
    )

print("Gesture classifier loaded successfully.")
print("Expected features:", model.n_features_in_)
print("Recognized labels:", model.classes_)

# ---------------- MEDIAPIPE SETUP ----------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ---------------- FEATURE EXTRACTION ----------------

def extract_features(hand_landmarks):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    # Wrist becomes origin.
    points -= points[0]

    # Normalize scale.
    scale = np.max(np.linalg.norm(points, axis=1))

    if scale > 1e-6:
        points /= scale

    return points.flatten().reshape(1, -1)

# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot open webcam. On macOS, enable camera access for your "
        "Terminal or IDE in System Settings > Privacy & Security > Camera."
    )

predictions_history = deque(maxlen=SMOOTHING_WINDOW)
displayed_number = None
displayed_confidence = 0.0

print("Starting live prediction. Press Q to quit.")

try:
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.monotonic() * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            current_number = None
            current_confidence = 0.0

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                features = extract_features(landmarks)

                # Draw landmarks and connections.
                height, width, _ = frame.shape
                points = []

                for lm in landmarks:
                    x = int(lm.x * width)
                    y = int(lm.y * height)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                for start, end in CONNECTIONS:
                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (255, 150, 0),
                        2
                    )

                # Predict probabilities for each gesture.
                probabilities = model.predict_proba(features)[0]
                best_index = int(np.argmax(probabilities))
                current_confidence = float(probabilities[best_index])
                predicted_label = int(model.classes_[best_index])

                if current_confidence >= CONFIDENCE_THRESHOLD:
                    current_number = predicted_label

            # Only include confident predictions in the smoothing window.
            if current_number is not None:
                predictions_history.append(current_number)
            else:
                predictions_history.append(None)

            valid_predictions = [
                p for p in predictions_history if p is not None
            ]

            if valid_predictions:
                counts = Counter(valid_predictions)
                candidate, votes = counts.most_common(1)[0]

                if votes >= MIN_VOTES:
                    displayed_number = candidate
                    displayed_confidence = current_confidence
            else:
                displayed_number = None
                displayed_confidence = 0.0

            # ---------------- DISPLAY ----------------

            if displayed_number is not None:
                text = f"Gesture: {displayed_number}"
                color = (0, 255, 0)

                cv2.putText(
                    frame,
                    f"Confidence: {displayed_confidence:.0%}",
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
            elif result.hand_landmarks:
                text = "Recognizing..."
                color = (0, 200, 255)
            else:
                text = "Show your hand"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                text,
                (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Step 6 - Live Gesture Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()

print("Live prediction stopped.")
