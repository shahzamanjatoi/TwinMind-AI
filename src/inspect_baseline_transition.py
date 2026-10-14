from pathlib import Path
import pandas as pd

data_dir = Path(__file__).resolve().parents[1] / "data"
path = next(data_dir.rglob("Combined_Baseline1.csv"))
df = pd.read_csv(path)

numeric = [
    c for c in df.select_dtypes(include="number").columns
    if c not in []
]

# Compare the 100 rows immediately before and after the transition.
before = df.iloc[2423:2523]
after = df.iloc[2523:2623]

print("Columns:", df.columns.tolist())
print("\nMissing values in each window:")
print(pd.DataFrame({
    "before": before.isna().sum(),
    "after": after.isna().sum(),
}).to_string())

print("\nNumeric feature medians before vs. after:")
comparison = pd.DataFrame({
    "before_median": before[numeric].median(),
    "after_median": after[numeric].median(),
})
comparison["change"] = (
    comparison["after_median"] - comparison["before_median"]
)
print(comparison.to_string())

for col in ["status", "volts"]:
    if col in df.columns:
        print(f"\n{col}: values before transition")
        print(before[col].value_counts(dropna=False).head(10).to_string())
        print(f"\n{col}: values after transition")
        print(after[col].value_counts(dropna=False).head(10).to_string())

print("\nAll columns around the transition, rows 2515–2539:")
print(df.iloc[2515:2540].to_string(index=True))
