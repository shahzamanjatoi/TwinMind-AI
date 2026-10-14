from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from amd_inference import TwinMindInference, DATA_DIR

predictor = TwinMindInference()
files = sorted(DATA_DIR.glob("Combined_*.csv"))

all_true = []
all_pred = []

print("Running full-dataset evaluation...")
print("Device:", predictor.device)

for path in files:
    data = pd.read_csv(path)

    if "status" not in data.columns:
        print(f"Skipping {path.name}: no status column")
        continue

    actual = data["status"].astype(str).str.strip().str.lower()
    predictions = predictor.predict(data)
    predicted = predictions["predicted_status"].astype(str).str.lower()

    accuracy = accuracy_score(actual, predicted)

    print(
        f"{path.name}: rows={len(data)}, "
        f"accuracy={accuracy:.4%}"
    )

    all_true.extend(actual.tolist())
    all_pred.extend(predicted.tolist())

labels = predictor.classes

print("\n=== OVERALL RESULTS ===")
print("Total rows:", len(all_true))
print("Accuracy:", f"{accuracy_score(all_true, all_pred):.4%}")

print("\n=== CLASSIFICATION REPORT ===")
print(
    classification_report(
        all_true,
        all_pred,
        labels=labels,
        target_names=labels,
        zero_division=0,
        digits=4,
    )
)

print("\n=== CONFUSION MATRIX ===")
print("Rows = actual class; columns = predicted class")
print("Class order:", labels)
print(confusion_matrix(all_true, all_pred, labels=labels))

print("\n=== PREDICTED CLASS COUNTS ===")
print(pd.Series(all_pred).value_counts().to_string())
