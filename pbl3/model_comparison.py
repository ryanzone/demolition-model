
import pandas as pd
from pathlib import Path

RESULTS_FOLDER = Path(__file__).parent / "model_results"

files = list(RESULTS_FOLDER.glob("*_metrics.csv"))

if not files:
    raise FileNotFoundError("No metrics CSV files found.")

dataframes = []

for file in files:
    df = pd.read_csv(file)

    required = ["Model", "Split", "Accuracy", "Macro F1"]

    if all(col in df.columns for col in required):
        dataframes.append(df[required])
    else:
        print("Skipping invalid file:", file.name)

if not dataframes:
    raise ValueError("No valid metrics files found.")

results = pd.concat(dataframes, ignore_index=True)

results["Split"] = results["Split"].astype(str).str.strip()

results = results.drop_duplicates(
    subset=["Model", "Split"],
    keep="last"
)

validation = results[
    results["Split"].str.lower() == "validation"
].copy()

if validation.empty:
    raise ValueError("No validation results found.")

validation = validation.sort_values(
    by=["Macro F1", "Accuracy"],
    ascending=False
).reset_index(drop=True)

validation["Rank"] = range(1, len(validation) + 1)
validation["Selected Top 5"] = validation["Rank"] <= 5

top5 = validation.head(5)

test_results = results[
    results["Split"].str.lower() == "test"
].copy()

final_results = top5.merge(
    test_results,
    on="Model",
    how="left",
    suffixes=("_Validation", "_Test")
)

validation.to_csv(
    RESULTS_FOLDER / "all_models_comparison.csv",
    index=False
)

top5.to_csv(
    RESULTS_FOLDER / "top_5_models.csv",
    index=False
)

final_results.to_csv(
    RESULTS_FOLDER / "top_5_final_results.csv",
    index=False
)

print("\nALL MODEL RANKINGS")
print(validation.to_string(index=False))

print("\nTOP 5 MODELS")
print(top5.to_string(index=False))

print("\nTOP 5 WITH TEST RESULTS")
print(final_results.to_string(index=False))

print("\nComparison completed!")
