import json
import glob

files = glob.glob("unsupervised_*.json")

operations = ["_".join(filename[:-5].split("_")[1:]) for filename in files]
operations = list(set(operations))
operations.sort()
operations.remove("AdamicAdarIndex")
operations.remove("JaccardCoefficient")
operations.append("JaccardCoefficient")
operations.append("AdamicAdarIndex")

results = {operation: [] for operation in operations}

for operation in operations:
    with open(f"unsupervised_{operation}.json", "r", encoding="utf-8") as f:
        result = json.load(f)
        results[operation] = [
            scores["average_precision"]
            for scores in result["results"]
            # scores["roc_auc"] for scores in result["results"]
        ]

num_snaps = len(results[list(operations)[0]])

print("distance,", end="")
print(",".join(operations))
for i in range(num_snaps):
    print(i + 1, end="")
    print(",", end="")
    for j, operation in enumerate(operations):
        result = results[operation][i]
        print(result, end="")
        if j != len(operations) - 1:
            print(",", end="")
    print()
