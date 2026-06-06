import json
import os
import sys

baseline_path = "eval/baseline.json"
latest_path = "eval/latest_scores.json"

if not os.path.exists(baseline_path):
    print(f"[Error] Baseline scores file not found at {baseline_path}")
    sys.exit(1)

if not os.path.exists(latest_path):
    print(f"[Error] Latest evaluation scores file not found at {latest_path}.")
    print("Please make sure the evaluation script (eval/run_evals.py) runs successfully first.")
    sys.exit(1)

with open(baseline_path) as f:
    baseline = json.load(f)

with open(latest_path) as f:
    latest = json.load(f)

THRESHOLD = 0.05  # 5% allowed drop
failed = False

for metric in ["faithfulness", "answer_relevancy"]:
    drop = baseline[metric] - latest[metric]
    status = "PASS" if drop <= THRESHOLD else "FAIL"
    if status == "FAIL":
        failed = True
    print(
        f"{metric}: baseline={baseline[metric]:.3f} "
        f"current={latest[metric]:.3f} "
        f"drop={drop:.3f} [{status}]"
    )

sys.exit(1 if failed else 0)