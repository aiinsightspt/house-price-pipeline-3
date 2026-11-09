"""Quality gate: stop the pipeline if the model is not good enough to release."""
import json
import os
import sys

import joblib
import pandas as pd

MAX_MAE = 20  # thousand euros: the largest average error we accept

with open("metrics.json") as f:
    m = json.load(f)

passed = m["mae"] <= MAX_MAE and m["mae"] < m["baseline_mae"]
verdict = "PASSED" if passed else "FAILED"

# Ask the model for the price of three example houses, so that the report shows it answering.
model = joblib.load("model.joblib")
houses = pd.DataFrame([[60, 2, 30], [100, 3, 15], [150, 4, 5]],
                      columns=["size_m2", "rooms", "age_years"])
prices = model.predict(houses)
rows = "\n".join(f"| {s} m², {r} rooms, {a} years old | {p:.1f} |"
                 for (s, r, a), p in zip(houses.values, prices))

report = f"""## Quality gate: {verdict}

| Measure | Value (thousand €) |
|---|---|
| Model error (MAE) on {m['test_houses']} test houses | {m['mae']} |
| Baseline error (always the average price) | {m['baseline_mae']} |
| Largest error accepted | {MAX_MAE} |

### What the model answers

| House | Predicted price (thousand €) |
|---|---|
{rows}
"""
print(report)

# Keep the report for the release notes and, on GitHub, show it on the run page.
with open("report.md", "w") as f:
    f.write(report)
if "GITHUB_STEP_SUMMARY" in os.environ:
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(report)

# A non-zero exit code makes this step fail, and the pipeline stops here.
sys.exit(0 if passed else 1)
