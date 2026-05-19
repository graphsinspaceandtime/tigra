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
)


if __name__ == "__main__":
    DATASET = "radoslaw-email"
    TIME_RES = "month"
    EPOCHS = 300
    LR = 0.01
    DIMS_TO_TEST = [10, 25, 50, 100]

    print(f"--- Loading {DATASET} ---")
    try:
        databank = Databank()
        es = databank.get_data(DATASET)
    except:
        es, _ = load_event_sequence(DATASET)

    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=TIME_RES, directed=False)
    graph_sequence = dg.get_graph_sequence()

    for d in DIMS_TO_TEST:
        h_dim = d * 2
        print(f"\nTesting DistanceDynAE | Dim: {d} | Hidden: {h_dim}")

        ae = DistanceDynAE(dim=d, hidden_dim=h_dim, epochs=EPOCHS, lr=LR)

        start_t = time.time()
        ae.make_embedding(dg)
        end_t = time.time()

        avg_f1 = []
        for i, graph_wrapper in enumerate(graph_sequence):
            emb_obj = ae.embeddings[i]
            if emb_obj is None:
                continue

            gre = GraphReconstructionEval(graph_wrapper, emb_obj, report_warnings=False)
            avg_f1.append(gre.getF1())

        mean_f1 = np.mean(avg_f1)
        print(f"Dim {d:<3} | Mean F1: {mean_f1:.4f} | Time: {end_t - start_t:.2f}s")
