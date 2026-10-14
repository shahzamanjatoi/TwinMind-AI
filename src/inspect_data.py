from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

matches = list(DATA_DIR.rglob("Combined_Baseline1.csv"))

if not matches:
    raise FileNotFoundError(
        "Could not find Combined_Baseline1.csv inside the data folder."
    )

data_folder = matches[0].parent
files = sorted(data_folder.glob("Combined_*.csv"))

if not files:
    raise FileNotFoundError(
        f"No Combined_*.csv files found in {data_folder}"
    )

print(f"Dataset folder: {data_folder}")
print(f"Files found: {len(files)}")

for file in files:
    print("\n" + "=" * 65)
    print(f"FILE: {file.name}")

    try:
        df = pd.read_csv(file)
        print(f"Rows: {len(df):,}")
        print(f"Columns: {list(df.columns)}")

        if "status" in df.columns:
            print("\nStatus counts:")
            print(df["status"].value_counts(dropna=False).to_string())
        else:
            print("WARNING: No 'status' column found.")

        print("\nMissing values:")
        print(df.isna().sum().to_string())

    except Exception as exc:
        print(f"ERROR reading {file.name}: {exc}")

print("\nInspection complete.")
