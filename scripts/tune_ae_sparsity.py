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
from tigra.core.greval import GraphReconstructionEval
from tigra.core.loaders import load_event_sequence
from tigra.embeddings.ae.autoencoders import DistanceDynAE


class TunedDistanceDynAE(DistanceDynAE):
    def __init__(
        self,
        dim=32,
        hidden_dim=64,
        epochs=300,
        lr=0.01,
        device=None,
        hub_alpha=1.0,
        pos_cap=50.0,
    ):
        super().__init__(dim, hidden_dim, epochs, lr, device)
        self.hub_alpha = hub_alpha
        self.pos_cap = pos_cap

    def compute_loss(self, recon_logits, target_adj, G, nodelist):
        num_edges = target_adj.sum().item()
        num_nodes = target_adj.shape[0]

        raw_pos_weight = (num_nodes**2 - num_edges) / (num_edges + 1e-9)
        pos_weight_val = (
            min(raw_pos_weight, self.pos_cap) if self.pos_cap else raw_pos_weight
        )

        base_weights = torch.ones_like(target_adj)
        base_weights[target_adj == 1] = pos_weight_val

        raw_loss = F.binary_cross_entropy_with_logits(
            recon_logits, target_adj, reduction="none"
        )

        degrees = torch.tensor(
            [G.degree(n) for n in nodelist], device=self.device, dtype=torch.float32
        )
        deg_outer = torch.outer(degrees, degrees)
        normalized_hub = deg_outer / (deg_outer.max() + 1e-9)
        hub_multiplier = 1.0 + (self.hub_alpha * normalized_hub)

        return (raw_loss * base_weights * hub_multiplier).mean()


if __name__ == "__main__":
    DATASETS = [
        ("radoslaw-email", "month"),
        ("CollegeMsg", "month"),
        ("fb-messages", "month"),
    ]
    DIM = 50

    CONFIGS = [
        {"name": "Default (No Cap, LR=0.01)", "lr": 0.01, "cap": None, "epochs": 300},
        {"name": "Tuned 1 (Cap=50, LR=0.005)", "lr": 0.005, "cap": 50.0, "epochs": 400},
        {"name": "Tuned 2 (Cap=20, LR=0.005)", "lr": 0.005, "cap": 20.0, "epochs": 400},
    ]

    print("SPARSITY TUNING")

    for DATASET, TIME_RES in DATASETS:
        print(f"\n DATASET: {DATASET}")
        try:
            databank = Databank()
            es = databank.get_data(DATASET)
        except:
            es, _ = load_event_sequence(DATASET)

        dg = DynamicGraph()
        dg.init_from_event_sequence(es, time_resolution=TIME_RES, directed=False)
        graph_sequence = dg.get_graph_sequence()

        for cfg in CONFIGS:
            ae = TunedDistanceDynAE(
                dim=DIM,
                hidden_dim=DIM * 2,
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                hub_alpha=1.0,
                pos_cap=cfg["cap"],
            )
            ae.make_embedding(dg)

            avg_f1 = []
            for i, graph_wrapper in enumerate(graph_sequence):
                if ae.embeddings[i] is None:
                    continue
                gre = GraphReconstructionEval(
                    graph_wrapper, ae.embeddings[i], report_warnings=False
                )
                avg_f1.append(gre.getF1())

            print(f"   -> {cfg['name']:<30} | F1: {np.mean(avg_f1):.4f}")
