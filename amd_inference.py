
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "twinmind_temporal_model.pt"
PREPROCESS_PATH = BASE_DIR / "models" / "preprocessing.joblib"
DATA_DIR = (
    BASE_DIR
    / "data"
    / "data"
    / "BentlyNevada_System1"
    / "CombinedFiles"
)


class TwinMindNetwork(nn.Sequential):
    def __init__(self, input_features=13, num_classes=6):
        super().__init__(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )


class TwinMindInference:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")
        if not PREPROCESS_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessing bundle not found: {PREPROCESS_PATH}"
            )

         # Use the AMD GPU (ROCm) when available, otherwise fall back to CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=True,
        )
        self.preprocessing = joblib.load(PREPROCESS_PATH)

        self.features = checkpoint["features"]
        self.classes = checkpoint["classes"]

        if self.features != self.preprocessing["features"]:
            raise ValueError(
                "Feature order mismatch between checkpoint and preprocessing."
            )

        self.imputer = self.preprocessing["imputer"]
        self.scaler = self.preprocessing["scaler"]

        self.model = TwinMindNetwork(
            input_features=len(self.features),
            num_classes=len(self.classes),
        ).to(self.device)

        # Verify that the reconstructed layer shapes match the checkpoint.
        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )
        self.model.eval()

    def predict(self, data):
        """Return predicted class, confidence, and probabilities for each row."""
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame.")

        missing = [f for f in self.features if f not in data.columns]
        if missing:
            raise ValueError(f"Missing required sensor features: {missing}")

        # Use only the saved feature order; exclude timestamp and status.
        X = data[self.features].copy()
        X = X.apply(pd.to_numeric, errors="coerce")

        # Apply the saved preprocessing steps in training order.
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        tensor = torch.as_tensor(
            X,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, indices = probabilities.max(dim=1)

        indices = indices.cpu().numpy()
        confidence = confidence.cpu().numpy()
        probabilities = probabilities.cpu().numpy()

        result = pd.DataFrame(
            {
                "predicted_status": [self.classes[i] for i in indices],
                "confidence": confidence,
            },
            index=data.index,
        )

        for i, class_name in enumerate(self.classes):
            result[f"prob_{class_name}"] = probabilities[:, i]

        return result


if __name__ == "__main__":
    predictor = TwinMindInference()

    print("Model loaded successfully.")
    print("Device:", predictor.device)
    print("Feature count:", len(predictor.features))
    print("Classes:", predictor.classes)

    # Smoke-test a few rows from every labeled dataset.
    files = sorted(DATA_DIR.glob("Combined_*.csv"))

    if not files:
        raise FileNotFoundError(f"No Combined_*.csv files found in {DATA_DIR}")

    for path in files:
        data = pd.read_csv(path)
        results = predictor.predict(data.head(5))

        print(f"\n=== {path.name} ===")
        if "status" in data.columns:
            results.insert(
                0,
                "known_status",
                data.head(5)["status"].astype(str).values,
            )
        print(results[["known_status", "predicted_status", "confidence"]]
              .to_string(index=False)
              if "known_status" in results.columns
              else results.to_string(index=False))
