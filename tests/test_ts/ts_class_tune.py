import sys
import os

sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True

from tigra.embeddings.embedding import FSEmbedding
from tigra.evaluation.classification_eval import eval
from tigra.evaluation.clbase.skl import (
    rf_tuning_factory,
    knn_tuning_factory,
    svm_tuning_factory,
    lr_tuning_factory,
)

emb_folders = [
    # "Plane_10_5_1.0_graph_dn2v",
    # "Plane_20_5_1.0_graph_dn2v",
    "Plane_50_1_1.0_graph_dn2v",
    "Plane_50_5_0.1_graph_dn2v",
    "Plane_50_5_0.05_graph_dn2v",
    "Plane_50_10_1.0_graph_dn2v",
    "Plane_10_5_1.0_graph_stwalk",
    "Plane_20_5_1.0_graph_stwalk",
    "Plane_50_1_1.0_graph_stwalk",
    "Plane_50_5_0.1_graph_stwalk",
    "Plane_50_5_0.05_graph_stwalk",
    "Plane_50_10_1.0_graph_stwalk",
    "Plane_10_5_1.0_graph_deep_hub",
    "Plane_20_5_1.0_graph_deep_hub",
    "Plane_50_1_1.0_graph_deep_hub",
    "Plane_50_5_0.1_graph_deep_hub",
    "Plane_50_5_0.05_graph_deep_hub",
    "Plane_50_10_1.0_graph_deep_hub",
    "Plane_10_5_1.0_graph_deep_lid",
    "Plane_50_5_0.1_graph_deep_lid",
    "Plane_50_5_0.05_graph_deep_lid",
    "Plane_50_10_1.0_graph_deep_lid",
]

header = "model\taccuracy\tprecision\trecall\tf1\n"


factories_to_test = [
    rf_tuning_factory,
    knn_tuning_factory,
    svm_tuning_factory,
    lr_tuning_factory,
]

if __name__ == "__main__":

    with open("rezultati.txt", "a", encoding="utf-8") as f:

        for folder_name in emb_folders:

            if not os.path.isdir(folder_name):
                continue

            emb = FSEmbedding(folder_name)
            f.write(f"=== {folder_name} ===\n")
            f.flush()
            os.fsync(f.fileno())

            for factory_func in factories_to_test:
                classifiers = factory_func()

                for classifier in classifiers:

                    param_name, param_val = "None", "None"
                    if classifier.name == "RF":
                        param_name, param_val = (
                            "n_estimators",
                            classifier.model.n_estimators,
                        )
                    elif classifier.name == "KNN":
                        param_name, param_val = (
                            "n_neighbors",
                            classifier.model.n_neighbors,
                        )
                    elif classifier.name == "SVM":
                        param_name, param_val = "kernel", classifier.model.kernel
                    elif classifier.name == "LOGREG":
                        param_name, param_val = "penalty", classifier.model.penalty

                    (score_accuracy, score_precision, score_recall, score_f1) = eval(
                        emb, classifier
                    )

                    line = f"{classifier.name}\t{param_name}={param_val}\t{score_accuracy}\t{score_precision}\t{score_recall}\t{score_f1}\n"
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
