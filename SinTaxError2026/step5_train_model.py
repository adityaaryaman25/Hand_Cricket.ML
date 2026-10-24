
import os
import csv
import joblib
import numpy as np

from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- CONFIGURATION ----------------

DATA_PATH = "data/gestures_old_backup.csv"
MODEL_PATH = "models/gesture_model.joblib"

os.makedirs("models", exist_ok=True)

# ---------------- LOAD DATASET ----------------

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Run step4_collect_data.py first."
    )

features = []
labels = []

with open(DATA_PATH, "r", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    expected_features = [f"f{i}" for i in range(63)]
    expected_columns = set(expected_features + ["label"])

    if not reader.fieldnames or not expected_columns.issubset(reader.fieldnames):
        raise ValueError(
            "CSV columns are incorrect. Expected f0-f62 and label. "
            "Check that you used the Step 4 data collection script."
        )

    for row in reader:
        try:
            values = [float(row[f"f{i}"]) for i in range(63)]
            label = int(row["label"])

            if label not in range(1, 11):
                continue

            if not np.all(np.isfinite(values)):
                continue

            features.append(values)
            labels.append(label)

        except (ValueError, TypeError, KeyError):
            continue

X = np.asarray(features, dtype=np.float32)
y = np.asarray(labels, dtype=np.int32)

if len(X) == 0:
    raise ValueError("No valid samples found in the dataset.")

# ---------------- CHECK CLASS COUNTS ----------------

counts = Counter(y.tolist())

print("\nSamples per gesture:")
for label in range(1, 11):
    print(f"Gesture {label}: {counts.get(label, 0)} samples")

missing = [label for label in range(1, 11) if counts.get(label, 0) < 2]

if missing:
    raise ValueError(
        f"These gestures need at least 2 samples each: {missing}. "
        "Collect more examples in Step 4."
    )

if min(counts.values()) < 5:
    print(
        "\nWarning: Some classes have fewer than 5 samples. "
        "Collect more data for a more reliable evaluation."
    )

# ---------------- TRAIN / TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ---------------- TRAIN CLASSIFIER ----------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=list(range(1, 11)),
        zero_division=0
    )
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=list(range(1, 11))))

# ---------------- SAVE MODEL ----------------

artifact = {
    "model": model,
    "feature_count": 63,
    "labels": list(range(1, 11)),
}

joblib.dump(artifact, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Training complete!")
