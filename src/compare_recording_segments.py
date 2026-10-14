from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
matches = list(DATA_DIR.rglob("Combined_Baseline1.csv"))

if not matches:
    raise FileNotFoundError("Combined_Baseline1.csv not found")

data_folder = matches[0].parent

features = [
    "ch1_derivedPk", "ch1_direct", "ch1_directRMS",
    "ch1_velocityPk", "ch1_velocityRMS",
    "ch2_derivedPk", "ch2_direct", "ch2_directRMS",
    "ch2_velocityPk", "ch2_velocityRMS",
]

for path in sorted(data_folder.glob("Combined_*.csv")):
    df = pd.read_csv(path)

    if "status" not in df.columns:
        continue

    df = df.dropna(subset=["status"]).copy()
    if "timestamp" in df.columns:
        parsed = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        if parsed.notna().sum() > 1:
            df["_time"] = parsed
            df = df.sort_values("_time", kind="stable", na_position="last")

    split = int(len(df) * 0.8)
    train_part = df.iloc[:split]
    test_part = df.iloc[split:]

    print("\n" + "=" * 70)
    print(f"FILE: {path.name} | Label: {df['status'].iloc[0]}")
    print(f"Train rows: {len(train_part)} | Test rows: {len(test_part)}")

    available = [f for f in features if f in df.columns]

    print("\nFeature medians: first 80% versus final 20%")
    comparison = pd.DataFrame({
        "first_80_percent": train_part[available].median(),
        "final_20_percent": test_part[available].median(),
    })
    comparison["absolute_difference"] = (
        comparison["first_80_percent"] -
        comparison["final_20_percent"]
    ).abs()

    print(comparison.sort_values(
        "absolute_difference", ascending=False
    ).round(5).to_string())

print("\nRecording comparison complete.")
