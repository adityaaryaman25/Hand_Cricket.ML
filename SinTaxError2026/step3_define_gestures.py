
import cv2
import mediapipe as mp
import json
import os
import time

# ---------------- CONFIGURATION ----------------

CAMERA_INDEX = 0
MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_PATH = "data/gesture_mapping.json"

os.makedirs("data", exist_ok=True)

# ---------------- CHECK MODEL ----------------

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Download hand_landmarker.task and place it in the models folder."
    )

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

# Draw connections manually to avoid MediaPipe-version differences.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ---------------- LOAD EXISTING MAPPING ----------------

if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as file:
        mapping = json.load(file)
else:
    mapping = {}

selected_number = 1
last_saved_time = 0
message = "Press 1-9 for a number, 0 for 10, S to save label"
camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open the camera. On macOS, allow camera access "
        "for your Terminal/IDE in System Settings > Privacy & Security > Camera."
    )

# ---------------- CAMERA LOOP ----------------

try:
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a camera frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.monotonic() * 1000)

            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Draw detected hand landmarks.
            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    height, width, _ = frame.shape
                    points = []

                    for landmark in hand:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    for start, end in CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (255, 150, 0),
                            2
                        )

                hand_status = "Hand detected"
                status_color = (0, 255, 0)
            else:
                hand_status = "No hand detected"
                status_color = (0, 0, 255)

            # Show current selection and controls.
            cv2.putText(
                frame,
                f"Selected number: {selected_number}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                hand_status,
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )

            cv2.putText(
                frame,
                "Keys: 1-9 = select | 0 = 10 | S = save | Q = quit",
                (15, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow("Step 3 - Gesture Mapping", frame)

            # Keep the GUI responsive: no terminal input() in this loop.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif ord("1") <= key <= ord("9"):
                selected_number = int(chr(key))

            elif key == ord("0"):
                selected_number = 10

            elif key == ord("s"):
                mapping[str(selected_number)] = f"Gesture {selected_number}"

                with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
                    json.dump(mapping, file, indent=2)

                message = f"Saved label for number {selected_number}"
                print(message)

            # Display save status without blocking the camera.
            cv2.putText(
                frame,
                message,
                (15, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

finally:
    camera.release()
    cv2.destroyAllWindows()

print(f"Gesture mapping saved to {OUTPUT_PATH}")
