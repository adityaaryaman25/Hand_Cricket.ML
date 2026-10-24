
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Path to the downloaded pretrained model
MODEL_PATH = Path("models/hand_landmarker.task")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Download hand_landmarker.task into the models folder."
    )

# Connections between the 21 hand landmarks
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),# Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Little finger
    (0, 17),
]

# Configure MediaPipe's pretrained hand detector
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Open webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Could not open webcam. Check camera permissions."
    )


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]
    points = []

    # Convert normalized coordinates to pixel coordinates
    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    # Draw connections
    for start, end in CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2,
        )

    # Draw landmarks
    for x, y in points:
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 0, 255),
            -1,
        )


try:
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the frame like a selfie camera
            frame = cv2.flip(frame, 1)

            # MediaPipe requires RGB images
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # VIDEO mode requires increasing timestamps
            timestamp_ms = time.monotonic_ns() // 1_000_000

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            if result.hand_landmarks:
                for hand_index, landmarks in enumerate(
                    result.hand_landmarks
                ):
                    draw_hand(frame, landmarks)

                    cv2.putText(
                        frame,
                        f"Hand {hand_index + 1}: "
                        f"{len(landmarks)} landmarks",
                        (20, 35 + hand_index * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Hand Detector", frame)

            # Press Q to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
