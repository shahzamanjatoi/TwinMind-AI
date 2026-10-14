from pathlib import Path
import joblib
import pandas as pd

root = Path(__file__).resolve().parents[1]
model_path = root / "models" / "twinmind_fault_classifier.joblib"

data_path = next(
    (root / "data").rglob("Combined_Baseline1.csv"),
    None
)

if data_path is None:
    raise FileNotFoundError("Combined_Baseline1.csv not found")

bundle = joblib.load(model_path)
model = bundle["pipeline"]
features = bundle["feature_columns"]

df = pd.read_csv(data_path)
X = df[features]

predictions = model.predict(X)
df["predicted_class"] = predictions

print("Actual label: baseline")
print("\nPredicted class counts for the entire recording:")
print(df["predicted_class"].value_counts().to_string())

print("\nPredictions in consecutive blocks of 100 rows:")
for start in range(0, len(df), 100):
    block = df.iloc[start:start + 100]
    counts = block["predicted_class"].value_counts()
    top_class = counts.index[0]
    top_count = int(counts.iloc[0])

    direct_median = block["ch1_direct"].median()
    print(
        f"Rows {start:4d}-{start + len(block) - 1:4d} | "
        f"ch1_direct median={direct_median:.4f} | "
        f"most predicted={top_class} ({top_count}/{len(block)})"
    )

print("\nPrediction counts before and after the transition:")
for name, start, end in [
    ("Before transition", 0, 2523),
    ("After transition", 2523, len(df)),
]:
    segment = df.iloc[start:end]
    print(f"\n{name}: rows {start}-{end - 1}")
    print(segment["predicted_class"].value_counts().to_string())
