from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from amd_inference import DATA_DIR, MODEL_PATH

SEED = 42
EPOCHS = 25
BATCH_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, torch.get_num_threads()))

checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
FEATURES = checkpoint["features"]
CLASSES = checkpoint["classes"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(CLASSES)}

train_parts, test_parts = [], []

for path in sorted(DATA_DIR.glob("Combined_*.csv")):
    df = pd.read_csv(path)
    if "status" not in df.columns:
        continue

    df["status"] = df["status"].astype(str).str.strip().str.lower()

    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce")
        if parsed.notna().any():
            df = (
                df.assign(_time=parsed, _order=np.arange(len(df)))
                .sort_values(["_time", "_order"], kind="stable", na_position="last")
                .drop(columns=["_time", "_order"])
                .reset_index(drop=True)
            )

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing features: {missing}")

    split = int(len(df) * 0.8)
    train_parts.append(df.iloc[:split].copy())
    test_parts.append(df.iloc[split:].copy())
    print(f"{path.name}: train={split}, test={len(df) - split}")

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(
    train_df[FEATURES].apply(pd.to_numeric, errors="coerce")
)
X_test = imputer.transform(
    test_df[FEATURES].apply(pd.to_numeric, errors="coerce")
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

y_train = train_df["status"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
y_test = test_df["status"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)

if pd.isna(y_train).any() or pd.isna(y_test).any():
    raise ValueError("Unknown labels found in the data.")

class TwinMindNetwork(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(len(FEATURES), 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, len(CLASSES)),
        )

model = TwinMindNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
)

print(f"\nTraining rows: {len(train_df)}")
print(f"Holdout rows: {len(test_df)}")
print("Training fresh model on CPU...")

for epoch in range(EPOCHS):
    model.train()
    running_loss, seen = 0.0, 0

    for xb, yb in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(yb)
        seen += len(yb)

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {running_loss / seen:.5f}")

model.eval()
with torch.no_grad():
    predictions = model(torch.from_numpy(X_test)).argmax(dim=1).numpy()

actual = [CLASSES[i] for i in y_test]
predicted = [CLASSES[i] for i in predictions]

print("\n=== HOLDOUT RESULTS ===")
print("Accuracy:", f"{accuracy_score(actual, predicted):.4%}")
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    actual, predicted, labels=CLASSES, target_names=CLASSES,
    digits=4, zero_division=0
))
print("\n=== CONFUSION MATRIX ===")
print("Rows = actual; columns = predicted")
print("Class order:", CLASSES)
print(confusion_matrix(actual, predicted, labels=CLASSES))

out = Path("models/holdout_experiment")
out.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "features": FEATURES,
    "classes": CLASSES,
    "split_method": "chronological 80/20 within each Combined CSV",
    "seed": SEED,
    "epochs": EPOCHS,
}, out / "holdout_model.pt")

joblib.dump({
    "imputer": imputer,
    "scaler": scaler,
    "features": FEATURES,
    "classes": CLASSES,
}, out / "holdout_preprocessing.joblib")

metrics = {
    "device": "cpu",
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "accuracy": accuracy_score(actual, predicted),
    "features": FEATURES,
    "classes": CLASSES,
    "classification_report": classification_report(
        actual, predicted, labels=CLASSES, target_names=CLASSES,
        output_dict=True, zero_division=0
    ),
    "confusion_matrix": confusion_matrix(
        actual, predicted, labels=CLASSES
    ).tolist(),
    "split_method": "chronological 80/20 within each Combined CSV",
    "warning": "Rows from the same recording may be correlated; architecture is reconstructed."
}
(out / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print(f"\nSaved artifacts to: {out.resolve()}")
