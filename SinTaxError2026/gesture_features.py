
import numpy as np

# Landmark indices: thumb, index, middle, ring, little fingertips
FINGERTIPS = [4, 8, 12, 16, 20]


def extract_features(hand_landmarks):
    """
    Returns 78 features:
    63 wrist-relative landmark coordinates
    5 wrist-to-fingertip distances
    10 pairwise fingertip distances
    """
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    # Translate landmarks so the wrist is at the origin.
    points -= points[0]

    # Scale consistently so hand size and camera distance
    # have less influence on predictions.
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 1e-6:
        points /= scale

    features = points.flatten().tolist()

    # Wrist-to-fingertip distances.
    for tip in FINGERTIPS:
        features.append(float(np.linalg.norm(points[tip])))

    # Distances between every pair of fingertips.
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            a = points[FINGERTIPS[i]]
            b = points[FINGERTIPS[j]]
            features.append(float(np.linalg.norm(a - b)))

    if len(features) != 78:
        raise ValueError(f"Expected 78 features, got {len(features)}")

    return np.asarray(features, dtype=np.float32)
