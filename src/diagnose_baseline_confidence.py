from pathlib import Path
import joblib
import pandas as pd

root = Path(__file__).resolve().parents[1]
bundle = joblib.load(root / "models" / "twinmind_fault_classifier.joblib")

model = bundle["pipeline"]
features = bundle["feature_columns"]
path = next((root / "data").rglob("Combined_Baseline1.csv"))

df = pd.read_csv(path)
X = df[features]

probabilities = model.predict_proba(X)
classes = list(model.classes_)

df["predicted_class"] = model.predict(X)
df["max_probability"] = probabilities.max(axis=1)

for name, start, end in [
    ("Before transition", 0, 2523),
    ("After transition", 2523, len(df)),
]:
    segment = df.iloc[start:end]
    print(f"\n{'=' * 60}")
    print(f"{name}: {len(segment)} rows")
    print("Prediction counts:")
    print(segment["predicted_class"].value_counts().to_string())
    print(
        f"\nMean maximum prediction probability: "
        f"{segment['max_probability'].mean():.3f}"
    )
    print(
        f"Median maximum prediction probability: "
        f"{segment['max_probability'].median():.3f}"
    )

    for class_name in ["baseline", "bent_shaft"]:
        if class_name in classes:
            col = classes.index(class_name)
            print(
                f"Mean probability of {class_name}: "
                f"{probabilities[start:end, col].mean():.3f}"
            )

print("\nNote: probability is model confidence, not a guarantee of correctness.")
