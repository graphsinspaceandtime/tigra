import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.loaders import load_event_sequence


from tigra.evaluation.linkpred import (
    Distance,
    EuclideanDistance,
    UnsupervisedLinkPrediction,
    Predicter,
)


from tigra.embeddings.ae.autoencoders import (
    DistanceDynAE,
    HubDistanceDynAE,
    LIDDistanceDynAE,
)

from tigra.evaluation.lideval import NCLIDEstimator

if __name__ == "__main__":
    DATASET = "radoslaw-email"
    TIME_RES = "month"
    DIM = 50
    EPOCHS = 300
    LR = 0.01

    print(f"--- Loading {DATASET} ---")
    try:
        databank = Databank()
        es = databank.get_data(DATASET)
    except:
        es, _ = load_event_sequence(DATASET)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=TIME_RES, directed=False)
    graph_sequence = dg.get_graph_sequence()

    print("PRE-COMPUTING LID SCORES")
    global_precomputed_lids = []
    for i, G_wrapper in enumerate(graph_sequence):
        G = G_wrapper.G
        if len(G.nodes) > 0:
            lid_estimator = NCLIDEstimator(G, alpha=1.0)
            lid_estimator.estimate_lids()
            nodelist = list(G.nodes())
            lid_scores = [lid_estimator.get_lid(n) for n in nodelist]
            global_precomputed_lids.append(
                torch.tensor(lid_scores, dtype=torch.float32)
            )
        else:
            global_precomputed_lids.append(torch.tensor([], dtype=torch.float32))

    models_to_test = [
        (
            "Distance_Basic",
            DistanceDynAE(dim=DIM, hidden_dim=DIM * 2, epochs=EPOCHS, lr=LR),
        ),
        (
            "Distance_Hub",
            HubDistanceDynAE(
                dim=DIM, hidden_dim=DIM * 2, epochs=EPOCHS, lr=LR, hub_alpha=1.0
            ),
        ),
        (
            "Distance_LID",
            LIDDistanceDynAE(
                dim=DIM,
                hidden_dim=DIM * 2,
                epochs=EPOCHS,
                lr=LR,
                lid_alpha=1.0,
                precomputed_lids=global_precomputed_lids,
            ),
        ),
    ]

    print(" STARTING UNSUPERVISED LINK PREDICTION")

    distance_metric = EuclideanDistance()

    final_results = {}

    for model_name, ae in models_to_test:
        print(f"Evaluating Model: {model_name}")

        start_t = time.time()
        ae.make_embedding(dg)
        print(f"Embeddings generated in {time.time() - start_t:.2f}s")

        link_predictor = UnsupervisedLinkPrediction(distance=distance_metric)
        evaluator = Predicter(linkpred=link_predictor, embedding=ae, dg=dg)

        evaluator.predict(print_results=True, take_top_k=False)

        valid_roc = [
            res["roc_auc"]
            for res in evaluator.results
            if res.get("roc_auc") is not None
        ]
        valid_ap = [
            res["average_precision"]
            for res in evaluator.results
            if res.get("average_precision") is not None
        ]

        mean_roc = np.mean(valid_roc) if valid_roc else 0.0
        mean_ap = np.mean(valid_ap) if valid_ap else 0.0

        final_results[model_name] = {"roc_auc": mean_roc, "avg_prec": mean_ap}

        print(
            f"\n {model_name} -> Mean ROC-AUC: {mean_roc:.4f} | Mean Avg Precision: {mean_ap:.4f}"
        )

    print("LINK PREDICTION")

    print(f"{'Model Name':<20} | {'Mean ROC-AUC':<15} | {'Mean Avg Precision':<15}")

    for model_name, metrics in final_results.items():
        print(
            f"{model_name:<20} | {metrics['roc_auc']:.4f}          | {metrics['avg_prec']:.4f}"
        )
