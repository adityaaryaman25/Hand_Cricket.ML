
import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time

# ---------------- CONFIGURATION ----------------

CAMERA_INDEX = 0
MODEL_PATH = "models/hand_landmarker.task"
DATA_PATH = "data/gestures.csv"

os.makedirs("data", exist_ok=True)

# ---------------- MEDIAPIPE SETUP ----------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
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
    """
    Convert 21 landmarks into 63 normalized values.
    Wrist is the origin; scale is normalized using
    the maximum distance from the wrist.
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    points -= points[0]

    scale = np.max(np.linalg.norm(points, axis=1))

    if scale > 1e-6:
        points /= scale

    return points.flatten().tolist()


# ---------------- CHECK FILE AND MODEL ----------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing model: {MODEL_PATH}\n"
        "Make sure hand_landmarker.task is inside the models folder."
    )

file_exists = os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 0

# ---------------- CAMERA ----------------

camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    raise RuntimeError(
        "Cannot open webcam. On macOS, allow camera access for "
        "your Terminal or IDE in System Settings > Privacy & Security > Camera."
    )

selected_label = 1
saved_this_session = 0
status_message = "Select a label (1-9, 0=10), then press SPACE"
last_save_time = 0.0

try:
    with HandLandmarker.create_from_options(options) as detector:
        with open(DATA_PATH, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)

            # Header contains 63 landmark features and one label.
            if not file_exists:
                header = [f"f{i}" for i in range(63)] + ["label"]
                writer.writerow(header)
                csvfile.flush()

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read camera frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp_ms = int(time.monotonic() * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)

                hand_features = None

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    hand_features = extract_features(landmarks)

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

                    hand_status = "Hand detected"
                    hand_color = (0, 255, 0)
                else:
                    hand_status = "No hand detected"
                    hand_color = (0, 0, 255)

                cv2.putText(
                    frame, f"Selected label: {selected_label}",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )

                cv2.putText(
                    frame, hand_status,
                    (15, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, hand_color, 2
                )

                cv2.putText(
                    frame, f"Saved this session: {saved_this_session}",
                    (15, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2
                )

                cv2.putText(
                    frame, "1-9: label | 0: label 10 | SPACE: save | Q: quit",
                    (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2
                )

                cv2.imshow("Step 4 - Collect Gesture Data", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                elif ord("1") <= key <= ord("9"):
                    selected_label = int(chr(key))
                    status_message = f"Selected label {selected_label}"

                elif key == ord("0"):
                    selected_label = 10
                    status_message = "Selected label 10"

                elif key == ord(" "):
                    if hand_features is None:
                        status_message = "No hand detected - sample not saved"
                    else:
                        # Avoid accidental duplicate snapshots from key repeats.
                        now = time.monotonic()

                        if now - last_save_time >= 0.15:
                            writer.writerow(hand_features + [selected_label])
                            csvfile.flush()

                            saved_this_session += 1
                            last_save_time = now
                            status_message = (
                                f"Saved sample for label {selected_label}"
                            )
                            print(status_message)

                # Show feedback on the next frame.
                cv2.putText(
                    frame, status_message,
                    (15, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2
                )

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Finished. Dataset saved at: {DATA_PATH}")
print(f"Samples collected this session: {saved_this_session}")
