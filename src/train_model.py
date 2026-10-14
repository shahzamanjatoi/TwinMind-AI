
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "ch1_bias",
    "ch1_derivedPk",
    "ch1_direct",
    "ch1_directRMS",
    "ch1_velocityPk",
    "ch1_velocityRMS",
    "ch2_bias",
    "ch2_derivedPk",
    "ch2_direct",
    "ch2_directRMS",
    "ch2_velocityPk",
    "ch2_velocityRMS",
]

matches = list(DATA_DIR.rglob("Combined_Baseline1.csv"))
if not matches:
    raise FileNotFoundError(
        "Combined_Baseline1.csv was not found in the data folder."
    )

data_folder = matches[0].parent
files = sorted(data_folder.glob("Combined_*.csv"))

train_parts = []
test_parts = []
split_summary = []

for file in files:
    df = pd.read_csv(file)

    if "status" not in df.columns:
        print(f"Skipping {file.name}: no status column.")
        continue

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"Skipping {file.name}: missing columns {missing}")
        continue

    df = df.dropna(subset=["status"]).copy()
    df["status"] = df["status"].astype(str).str.strip()

    # Convert the timestamp if possible; otherwise retain row order.
    if "timestamp" in df.columns:
        parsed_time = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True
        )
        if parsed_time.notna().sum() > 1:
            df["_sort_time"] = parsed_time
            df = df.sort_values(
                "_sort_time", kind="stable", na_position="last"
            )

    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df["status"]

    # Exclude records where every sensor measurement is missing.
    valid = X.notna().any(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    if len(X) < 5:
        print(f"Skipping {file.name}: too few usable rows.")
        continue

    # Keep chronological order: first 80% for training,
    # final 20% for testing.
    split_at = int(len(X) * 0.80)
    split_at = max(1, min(split_at, len(X) - 1))

    train_parts.append((X.iloc[:split_at], y.iloc[:split_at]))
    test_parts.append((X.iloc[split_at:], y.iloc[split_at:]))

    split_summary.append({
        "file": file.name,
        "class": str(y.iloc[0]),
        "rows": int(len(X)),
        "train_rows": int(split_at),
        "test_rows": int(len(X) - split_at),
    })

    print(
        f"{file.name}: {len(X):,} usable rows | "
        f"train={split_at:,} | test={len(X) - split_at:,}"
    )

if not train_parts or not test_parts:
    raise ValueError("No recordings could be split for evaluation.")

X_train = pd.concat(
    [part[0] for part in train_parts], ignore_index=True
)
y_train = pd.concat(
    [part[1] for part in train_parts], ignore_index=True
)
X_test = pd.concat(
    [part[0] for part in test_parts], ignore_index=True
)
y_test = pd.concat(
    [part[1] for part in test_parts], ignore_index=True
)

if y_train.nunique() < 2:
    raise ValueError("Training data must contain at least two classes.")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
labels = sorted(set(y_test) | set(predictions))

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    y_test, predictions, labels=labels
)

model_path = MODEL_DIR / "twinmind_fault_classifier.joblib"

joblib.dump({
    "pipeline": model,
    "feature_columns": FEATURES,
    "classes": list(model.named_steps["classifier"].classes_),
    "training_note": (
        "Chronological holdout within each source recording. "
        "This does not establish generalization to unseen machines "
        "or independent recordings."
    ),
}, model_path)

metrics = {
    "validation_method": "chronological 80/20 split within each file",
    "rows_used": int(len(X_train) + len(X_test)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "features": FEATURES,
    "file_splits": split_summary,
    "train_class_counts": {
        str(k): int(v) for k, v in y_train.value_counts().items()
    },
    "test_class_counts": {
        str(k): int(v) for k, v in y_test.value_counts().items()
    },
    "accuracy": float(accuracy),
    "classification_report": report,
    "confusion_matrix_labels": labels,
    "confusion_matrix": matrix.tolist(),
    "evaluation_note": (
        "Chronological holdout reduces random-row leakage but records "
        "within each file may still be correlated. Most fault categories "
        "have only one source recording, so unseen-run generalization "
        "has not been established."
    ),
}

metrics_path = MODEL_DIR / "evaluation_metrics.json"
metrics_path.write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

print("\n" + "=" * 60)
print("CHRONOLOGICAL MODEL EVALUATION")
print("=" * 60)
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")
print(f"Accuracy:      {accuracy:.4f}")

print("\nTest class counts:")
print(y_test.value_counts().to_string())

print("\nClassification report:")
print(classification_report(
    y_test, predictions, labels=labels, zero_division=0
))

print("Confusion matrix (rows=true, columns=predicted):")
print(pd.DataFrame(
    matrix, index=labels, columns=labels
).to_string())

print(f"\nSaved model:   {model_path}")
print(f"Saved metrics: {metrics_path}")
print("\nEvaluation complete.")
