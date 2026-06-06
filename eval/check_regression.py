import json, sys

with open("eval/baseline.json") as f:
    baseline = json.load(f)

with open("eval/latest_scores.json") as f:
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