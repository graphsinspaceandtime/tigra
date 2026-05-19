import sys
import os
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tigra.core.dgraph import DynamicGraph
from data.bank import Databank
from tigra.core.greval import GraphReconstructionEval
from tigra.core.loaders import load_event_sequence


from tigra.embeddings.ae.autoencoders import (
    AbstractDynAE,
    GCNLayer,
    EuclideanGAEModel,
    DistanceDynAE,
    HubDistanceDynAE,
    LIDDistanceDynAE,
)


try:
    from tigra.evaluation.lideval import NCLIDEstimator
except ImportError:
    pass


if __name__ == "__main__":
    DATASET = "radoslaw-email"
    TIME_RES = "month"
    EPOCHS = 300
    LR = 0.01

    DIMS_TO_TEST = [25, 50, 100]

    print(f"--- Loading {DATASET} ---")
    try:
        databank = Databank()
        es = databank.get_data(DATASET)
    except:
        es, _ = load_event_sequence(DATASET)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=TIME_RES, directed=False)
    graph_sequence = dg.get_graph_sequence()

    print(" PRE-COMPUTING LID SCORES ONCE FOR ALL SNAPSHOTS...")

    global_precomputed_lids = []
    for i, G_wrapper in enumerate(graph_sequence):
        G = G_wrapper.G
        print(
            f"    -> Estimating Natural Communities for Snapshot {i} ({len(G.nodes)} nodes)..."
        )

        lid_estimator = NCLIDEstimator(G, alpha=1.0)
        lid_estimator.estimate_lids()

        nodelist = list(G.nodes())
        lid_scores = [lid_estimator.get_lid(n) for n in nodelist]

        global_precomputed_lids.append(torch.tensor(lid_scores, dtype=torch.float32))

    print("Pre-computation complete\n")

    models_to_test = [
        (
            "Distance_Basic",
            lambda d, h: DistanceDynAE(dim=d, hidden_dim=h, epochs=EPOCHS, lr=LR),
        ),
        (
            "Distance_Hub_a1.0",
            lambda d, h: HubDistanceDynAE(
                dim=d, hidden_dim=h, epochs=EPOCHS, lr=LR, hub_alpha=1.0
            ),
        ),
        (
            "Distance_LID_a1.0",
            lambda d, h: LIDDistanceDynAE(
                dim=d,
                hidden_dim=h,
                epochs=EPOCHS,
                lr=LR,
                lid_alpha=1.0,
                precomputed_lids=global_precomputed_lids,
            ),
        ),
    ]

    overall_results = {name: {} for name, _ in models_to_test}

    print("THE ULTIMATE TEST")

    for model_name, model_constructor in models_to_test:
        print(f"TESTING MODEL: {model_name}")

        for d in DIMS_TO_TEST:
            h_dim = d * 2
            ae = model_constructor(d, h_dim)

            start_t = time.time()
            ae.make_embedding(dg)
            end_t = time.time()

            avg_f1 = []
            for i, graph_wrapper in enumerate(graph_sequence):
                emb_obj = ae.embeddings[i]
                if emb_obj is None:
                    continue

                gre = GraphReconstructionEval(
                    graph_wrapper, emb_obj, report_warnings=False
                )
                avg_f1.append(gre.getF1())

            mean_f1 = np.mean(avg_f1)
            overall_results[model_name][d] = mean_f1
            print(
                f"{model_name} | Dim {d:<3} | F1: {mean_f1:.4f} | Time: {end_t - start_t:.2f}s"
            )

    print("FINAL COMPARISON RESULTS")

    model_names = [name for name, _ in models_to_test]
    header = f"{'Dim':<5} | " + " | ".join([f"{name:<20}" for name in model_names])
    print(header)
    print("-" * len(header))

    for d in DIMS_TO_TEST:
        row_str = f"{d:<5} | "
        scores = [overall_results[name].get(d, 0.0) for name in model_names]
        best_score = max(scores) if scores else 0

        for score in scores:
            marker = "!" if score == best_score and score > 0 else "  "
            row_str += f"{score:.4f} {marker}             | "

        print(row_str[:-3])
