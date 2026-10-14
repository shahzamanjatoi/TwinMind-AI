from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parents[1]
path = next((root / "data").rglob("Combined_Baseline1.csv"))
df = pd.read_csv(path)

columns = ["ch1_direct", "ch2_direct"]
threshold = 0.01

print("Baseline1 direct-vibration diagnostic")
print(f"Threshold used for exploration: {threshold}")

for name, start, end in [
    ("Before transition", 0, 2523),
    ("After transition", 2523, len(df)),
]:
    segment = df.iloc[start:end]
    low = segment[columns].lt(threshold).all(axis=1)

    print(f"\n{name}:")
    print(f"Rows: {len(segment)}")
    print(f"Both channels below threshold: {low.sum()}")
    print(f"Percentage: {low.mean() * 100:.2f}%")
    print("Median direct-vibration readings:")
    print(segment[columns].median().to_string())

print("\nThis threshold is exploratory, not a validated machine alarm limit.")
