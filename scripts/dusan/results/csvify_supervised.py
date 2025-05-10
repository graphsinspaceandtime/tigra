import json
import glob

files = glob.glob("supervised_*.json")

print(files[0][:-5].split("_"))

operations = [filename[:-5].split("_")[2] for filename in files]
operations = set(operations)

classifiers = [filename[:-5].split("_")[1] for filename in files]
classifiers = set(classifiers)

results = {operation: {} for operation in operations}

for operation in operations:
    results[operation] = {classifier: None for classifier in classifiers}
    for classifier in classifiers:
        with open(
            f"unsupervised_{classifier}_{operation}.json", "r", encoding="utf-8"
        ) as f:
            result = json.load(f)
            results[operation][classifier] = [
                scores["f1"] for scores in result["results"]
            ]

num_snaps = len(results[list(operations)[0]][list(classifiers)[0]])

for operation in operations:
    print(operation)
    print("classifier,", end="")
    print(",".join(classifiers))
    for i in range(num_snaps):
        print(i + 1, end="")
        print(",", end="")
        for j, classifier in enumerate(classifiers):
            result = results[operation][classifier][i]
            print(result, end="")
            if j != len(classifiers) - 1:
                print(",", end="")
        print()
