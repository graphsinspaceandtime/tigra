import sys
import os
import csv
import random
import numpy as np
import networkx as nx


sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph


try:
    from tigra.data.bank import Databank
except ImportError:
    try:
        from data.bank import Databank
    except ImportError:
        from tigra.data.loaders import Databank

from tigra.evaluation.linkpred import (
    combine_pair_operation_factory,
    SupervisedLinkPrediction,
    Predicter,
)


from tigra.evaluation.clbase.skl import (
    skl_classifiers_factory,
    rf_tuning_factory,
    knn_tuning_factory,
    svm_tuning_factory,
    lr_tuning_factory,
)


from tigra.embeddings.rndw.rlwalk import DynRLWalk


class RamEmbedding:
    """
    Wraps a list of embeddings to mimic FSEmbedding interface for Predicter.
    """

    def __init__(self, embedding_list):
        self.embeddings = embedding_list

    def num_embeddings(self):
        return len(self.embeddings)

    def get_embedding(self, index):
        if 0 <= index < len(self.embeddings):
            return self.embeddings[index]
        return None

    def __getitem__(self, index):
        return self.embeddings[index]

    def __len__(self):
        return len(self.embeddings)


if __name__ == "__main__":
    DATASET_NAME = "radoslaw-email"
    OUTPUT_FILE = "rl_supervised_tuning.csv"

    print(f"--- Loading {DATASET_NAME} ---")
    databank = Databank()
    es = databank.get_data(DATASET_NAME)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution="month", directed=False)
    dg.info()

    print("\n[1/3] Generating RL Embeddings...")
    rl_model = DynRLWalk(
        dim=50,
        epsilon=0.2,
        alpha=0.1,
        reward_bias=True,
        num_walks=10,
        walk_length=40,
        num_workers=1,
    )
    raw_embeddings = rl_model.construct(dg)
    embed_wrapper = RamEmbedding(raw_embeddings)
    print(f"      Generated {len(raw_embeddings)} snapshots.")

    csv_headers = [
        "Classifier",
        "Param_Name",
        "Param_Value",
        "Combiner",
        "Snapshot",
        "F1",
        "Acc",
        "Prec",
        "Rec",
        "Pos_Prec",
        "Pos_Rec",
        "Neg_Prec",
        "Neg_Rec",
    ]

    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    print(f"\n[2/3] Starting Supervised Grid Search -> {OUTPUT_FILE}")

    best_global_f1 = -1.0
    best_global_config = {}

    factories_to_test = [rf_tuning_factory, knn_tuning_factory, lr_tuning_factory]

    for factory_func in factories_to_test:
        classifiers = factory_func()
        for classifier in classifiers:
            for combine_pair_operation in combine_pair_operation_factory():
                linkpred = SupervisedLinkPrediction(combine_pair_operation, classifier)
                predicter = Predicter(linkpred, embed_wrapper, dg)

                p_name = "None"
                p_val = "None"
                if classifier.name == "RF":
                    p_name, p_val = "n_estimators", classifier.model.n_estimators
                elif classifier.name == "KNN":
                    p_name, p_val = "n_neighbors", classifier.model.n_neighbors
                elif classifier.name == "LOGREG":
                    p_name, p_val = "penalty", classifier.model.penalty

                print(
                    f"      Testing {classifier.name} [{p_name}={p_val}] + {combine_pair_operation.name}"
                )

                try:
                    predicter.predict(print_results=False)
                    with open(OUTPUT_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        current_f1s = []
                        for idx, res in enumerate(predicter.results, start=1):
                            writer.writerow(
                                [
                                    classifier.name,
                                    p_name,
                                    p_val,
                                    combine_pair_operation.name,
                                    idx,
                                    res.get("f1", 0),
                                    res.get("accuracy", 0),
                                    res.get("precision", 0),
                                    res.get("recall", 0),
                                    res.get("pos_prec", 0),
                                    res.get("pos_rec", 0),
                                    res.get("neg_prec", 0),
                                    res.get("neg_rec", 0),
                                ]
                            )
                            current_f1s.append(res.get("f1", 0))

                    if current_f1s:
                        avg_f1 = np.mean(current_f1s)
                        if avg_f1 > best_global_f1:
                            best_global_f1 = avg_f1
                            best_global_config = {
                                "classifier": classifier.name,
                                "param": f"{p_name}={p_val}",
                                "combiner": combine_pair_operation.name,
                                "avg_f1": best_global_f1,
                            }
                except Exception as e:
                    print(f"      [ERROR] {e}")

    print("\n" + "#" * 60)
    if best_global_config:
        print(
            f"WINNER: {best_global_config['classifier']} ({best_global_config['param']})"
        )
        print(f"Combiner: {best_global_config['combiner']}")
        print(f"Avg F1: {best_global_config['avg_f1']:.5f}")
    print("#" * 60)
