
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

matches = list(DATA_DIR.rglob("Combined_Baseline1.csv"))
if not matches:
    raise FileNotFoundError("Could not find Combined_Baseline1.csv")

data_folder = matches[0].parent
files = sorted(data_folder.glob("Combined_*.csv"))

features = [
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

frames = []

for file in files:
    df = pd.read_csv(file)

    if "status" not in df.columns:
        print(f"Skipping {file.name}: no status column")
        continue

    available = [c for c in features if c in df.columns]
    missing = sorted(set(features) - set(available))

    if missing:
        print(f"Warning: {file.name} missing features: {missing}")

    df = df[available + ["status"]].copy()
    df["source_file"] = file.name

    for col in available:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

print("\n=== DATASET OVERVIEW ===")
print(f"Total rows: {len(data):,}")
print("\nRows per label:")
print(data["status"].value_counts().to_string())

print("\n=== MISSING VALUES BY CLASS ===")
print(data.groupby("status")[features].apply(
    lambda group: group.isna().sum()
).to_string())

print("\n=== SENSOR FEATURE SUMMARY BY CLASS ===")
for feature in features:
    if feature not in data.columns:
        continue

    summary = data.groupby("status")[feature].agg(
        ["count", "mean", "std", "median", "min", "max"]
    ).round(5)

    print(f"\n--- {feature} ---")
    print(summary.to_string())

print("\n=== SOURCE FILES PER CLASS ===")
print(data.groupby("status")["source_file"].nunique().to_string())

print("\n=== BASELINE VS BENT SHAFT: MEDIAN COMPARISON ===")
pair = data[data["status"].isin(["baseline", "bent_shaft"])]

if not pair.empty:
    medians = pair.groupby("status")[features].median().T
    if {"baseline", "bent_shaft"}.issubset(medians.columns):
        medians["absolute_median_difference"] = (
            medians["baseline"] - medians["bent_shaft"]
        ).abs()
        print(
            medians.sort_values(
                "absolute_median_difference", ascending=False
            ).round(5).to_string()
        )
    else:
        print("One of the expected labels was not found.")
else:
    print("No baseline/bent_shaft records found.")

print("\nDiagnosis complete.")
