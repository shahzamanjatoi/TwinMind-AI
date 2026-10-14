from pathlib import Path
import pandas as pd

data_dir = Path(__file__).resolve().parents[1] / "data"
files = sorted(data_dir.rglob("Combined_*.csv"))

features = [
    "ch1_direct",
    "ch1_directRMS",
    "ch1_velocityPk",
    "ch1_velocityRMS",
    "ch2_direct",
    "ch2_directRMS",
    "ch2_velocityPk",
    "ch2_velocityRMS",
]

for path in files:
    df = pd.read_csv(path)
    print(f"\n{'=' * 75}")
    print(f"{path.name} | rows={len(df)}")

    available = [c for c in features if c in df.columns]
    print("\nOverall medians:")
    print(df[available].median().round(4).to_string())

    direct = ["ch1_direct", "ch2_direct"]
    if all(c in df.columns for c in direct):
        low = (df[direct] < 0.01).all(axis=1)
        print(
            f"\nRows where BOTH direct channels < 0.01: "
            f"{int(low.sum())} ({low.mean():.1%})"
        )
        if "timestamp" in df.columns and low.any():
            print("First low-reading timestamps:")
            print(df.loc[low, "timestamp"].head(5).to_string(index=False))
