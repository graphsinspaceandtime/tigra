import os
import json
from typing import Sequence

from tigra.evaluation.linkpred import distance_measure_factory, combine_pair_operation_factory

#from tigra.evaluation.linkpred_classification.skl import skl_classifiers_factory
from tigra.evaluation.clbase.skl import skl_classifiers_factory

distances = [distance_measure.name for distance_measure in distance_measure_factory()]
combiners = [combiner.name for combiner in combine_pair_operation_factory()]
classifiers = [classifier.name for classifier in skl_classifiers_factory()]

def fragments(name: str) -> Sequence[str]:
    return name[:-5].split("~")

def write_unsup_csv(data: dict, dataset_id: str, method: str, metric_name: str):
    metric = []

    # save heading
    metric.append(("Snapshots," + ",".join(list(data.keys()))))

    # save results by snapshot
    max_results = len(data["AdamicAdarIndex"]["results"])
    for snap in range(max_results):
        result_line = str(snap + 1) + ","

        # save roc results for one snapshot
        for distance in distances:
            distance_data = data[distance]
            result_line += str(distance_data["results"][snap][metric_name])
            result_line += ","

        # remove trailing comma
        result_line = result_line[:-1]
        metric.append(result_line)

    print("\n".join(metric))

def write_sup_csv(data: dict, combiner_name: str):
    combiner = []
    combiner.append(("Classifier," + ",".join(list(data[combiner_name].keys()))))

    max_results = len(data[combiner_name]["SVM"]["results"])
    for snap in range(max_results):
        result_line = str(snap + 1) + ","

        # save combiner results for one snapshot
        for distance in classifiers:
            distance_data = data[combiner_name][distance]
            result_line += str(distance_data["results"][snap]["f1"])
            result_line += ","

        # remove trailing comma
        result_line = result_line[:-1]
        combiner.append(result_line)

    print("\n".join(combiner))



results_path = "./results"
results = [
    result_file
    for result_file in os.listdir(results_path)
    if os.path.isfile(os.path.join(results_path, result_file))
    and result_file.endswith(".json")
]

unsup_results = [
    result_file for result_file in results if result_file.startswith("unsup")
]
sup_results = [result_file for result_file in results if result_file.startswith("sup")]

unsup_methods = [fragments(result_file)[1] for result_file in unsup_results]
unsup_methods = list(set(unsup_methods))

unsup_dataset_ids = [fragments(result_file)[2] for result_file in unsup_results]
unsup_dataset_ids = list(set(unsup_dataset_ids))

sup_methods = [fragments(result_file)[1] for result_file in sup_results]
sup_methods = list(set(sup_methods))

sup_dataset_ids = [fragments(result_file)[2] for result_file in sup_results]
sup_dataset_ids = list(set(sup_dataset_ids))

sup_combiners = [fragments(result_file)[4] for result_file in sup_results]
sup_combiners = list(set(sup_combiners))

for method in unsup_methods:
    for dataset_id in unsup_dataset_ids:
        table_data = {}

        for result_file in os.listdir(results_path):
            if not os.path.isfile(os.path.join(results_path, result_file)):
                continue
            if not result_file.endswith(".json"):
                continue
            if not result_file.startswith(f"unsup~{method}~{dataset_id}~"):
                continue

            print(result_file)

            distance = fragments(result_file)[3]

            table_data[distance] = json.load(
                open(os.path.join(results_path, result_file), "r", encoding="utf-8")
            )
            print(result_file)

        print()
        print("ROC AUC")
        write_unsup_csv(table_data, dataset_id, method, "roc_auc")
        print()
        print("Average precision")
        write_unsup_csv(table_data, dataset_id, method, "average_precision")

for method in sup_methods:
    for dataset_id in sup_dataset_ids:
        table_data = {}

        for classifier in classifiers:
            for result_file in os.listdir(results_path):
                if not os.path.isfile(os.path.join(results_path, result_file)):
                    continue
                if not result_file.endswith(".json"):
                    continue
                if not result_file.startswith(f"sup~{method}~{dataset_id}~"):
                    continue

                if classifier not in result_file:
                    continue

                print(fragments(result_file))

                distance = fragments(result_file)[4]

                if distance not in table_data:
                    table_data[distance] = {}

                table_data[distance][classifier] = json.load(
                    open(os.path.join(results_path, result_file), "r", encoding="utf-8")
                )

        print()
        print("AbsoluteDifferenceCombiner")
        write_sup_csv(table_data, "AbsoluteDifferenceCombiner")
        print()
        print()
        print("HadamardCombiner")
        write_sup_csv(table_data, "HadamardCombiner")
        print()
        print()
        print("ConcatinationCombiner")
        write_sup_csv(table_data, "ConcatinationCombiner")
