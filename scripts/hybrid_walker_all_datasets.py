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
from tigra.embeddings.rndw.rndwbase import RWEmbEgoDelta, RandomWalkSampler

from tigra.embeddings.ae.autoencoders import DistanceDynAE

# TODO: Modify, tune, ...


class ExportingDistanceDynAE(DistanceDynAE):
    def __init__(self, dim=32, hidden_dim=64, epochs=300, lr=0.01):
        super().__init__(dim, hidden_dim, epochs, lr)
        self.probability_matrices = {}
        self.current_snap_idx = 0

    def train_snapshot(self, G):
        features, adj_norm, target_adj, nodelist = self._prepare_input(G)

        model = self.model_class(features.shape[1], self.hidden_dim, self.dim).to(
            self.device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon_logits, z = model(features, adj_norm)
            loss = self.compute_loss(recon_logits, target_adj, G, nodelist)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            z_final = model.encode(features, adj_norm)
            logits = model.decode(z_final)

            probs = torch.sigmoid(logits).cpu().numpy()
            np.fill_diagonal(probs, 0)

            K = 10
            if probs.shape[1] > K:
                thresholds = np.sort(probs, axis=1)[:, -K]

                probs[probs < thresholds[:, None]] = 0.0

            row_sums = probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            probs = probs / row_sums

            self.probability_matrices[self.current_snap_idx] = {
                "matrix": probs,
                "nodelist": nodelist,
            }

        self.current_snap_idx += 1
        return z_final.cpu().numpy()

    @property
    def model_class(self):
        from tigra.embeddings.ae.autoencoders import EuclideanGAEModel

        return EuclideanGAEModel


class EuclideanHybridSampler(RandomWalkSampler):
    def __init__(
        self, graph, nodes, ae_prob_data, alpha=0.8, num_walks=10, walk_length=32
    ):
        super().__init__(graph, nodes)
        self.nw = num_walks
        self.wl = walk_length
        self.alpha = alpha

        if ae_prob_data is not None:
            self.ae_prob_matrix = ae_prob_data["matrix"]
            self.nodelist = ae_prob_data["nodelist"]
            self.node_to_local_idx = {n: i for i, n in enumerate(self.nodelist)}
        else:
            self.ae_prob_matrix = None
            self.nodelist = []
            self.node_to_local_idx = {}

    def select_next_node(self, start_node, current_node, prev_node, neighbours):
        if len(neighbours) > 0 and (
            np.random.rand() < self.alpha or self.ae_prob_matrix is None
        ):
            total_hub = sum([self.G.degree[n] for n in neighbours])
            if total_hub == 0:
                return np.random.choice(neighbours)
            probabilities = [self.G.degree[n] / total_hub for n in neighbours]
            return np.random.choice(neighbours, 1, p=probabilities)[0]

        else:
            if (
                self.ae_prob_matrix is None
                or current_node not in self.node_to_local_idx
            ):
                return np.random.choice(list(self.G.nodes()))

            curr_idx = self.node_to_local_idx[current_node]
            ae_probs = self.ae_prob_matrix[curr_idx].copy()

            if ae_probs.sum() == 0:
                return np.random.choice(list(self.G.nodes()))

            ae_probs = ae_probs / ae_probs.sum()
            next_idx = np.random.choice(len(ae_probs), p=ae_probs)

            return self.nodelist[next_idx]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class DynEuclideanHybridWalk(RWEmbEgoDelta):
    def __init__(self, dim, probability_matrices, alpha=0.8):
        super().__init__(dim)
        self.probability_matrices = probability_matrices
        self.alpha = alpha
        self.current_snapshot_idx = 0

    def create_random_walk_sampler(self, graph, nodes):
        ae_prob_data = self.probability_matrices.get(self.current_snapshot_idx, None)
        sampler = EuclideanHybridSampler(graph, nodes, ae_prob_data, alpha=self.alpha)
        self.current_snapshot_idx += 1
        return sampler


if __name__ == "__main__":
    DATASETS = [
        ("radoslaw-email", "month"),
        ("ia-hospital-ward-proximity-attr", "day"),
        ("ia-contacts_hypertext2009", "day"),
        ("ia-enron-employees", "3-months"),
        ("ia-primary-school-proximity-attr", "day"),
        ("fb-forum", "month"),
        ("email-Eu-core-temporal", "month"),
        ("CollegeMsg", "month"),
        ("fb-messages", "month"),
    ]

    DIM = 50
    WALK_ALPHA = 0.95

    print(" THE HYBRID WALKER")

    for DATASET, TIME_RES in DATASETS:
        print("\n" + "=" * 80)
        print(f" DATASET: {DATASET} (Res: {TIME_RES})")
        print("=" * 80)

        try:
            try:
                databank = Databank()
                es = databank.get_data(DATASET)
            except:
                es, _ = load_event_sequence(DATASET)

            dg = DynamicGraph()
            dg.init_from_event_sequence(es, time_resolution=TIME_RES, directed=False)
            graph_sequence = dg.get_graph_sequence()

            print("Training Autoencoder to generate teleportation maps...")
            start_t = time.time()
            ae = ExportingDistanceDynAE(
                dim=DIM, hidden_dim=DIM * 2, epochs=300, lr=0.01
            )
            ae.make_embedding(dg)
            print(f"Maps generated in {time.time() - start_t:.2f}s")

            print(f"Running Hybrid Random Walk (Alpha={WALK_ALPHA})...")
            start_t = time.time()

            walker = DynEuclideanHybridWalk(
                dim=DIM, probability_matrices=ae.probability_matrices, alpha=WALK_ALPHA
            )
            walker.make_embedding(dg)
            print(f"Walker finished in {time.time() - start_t:.2f}s")

            avg_f1 = []
            for i, graph_wrapper in enumerate(graph_sequence):
                emb_obj = walker.embeddings[i]
                if emb_obj is None:
                    continue

                for node, vec in emb_obj.embedding.items():
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        emb_obj.embedding[node] = vec / norm

                gre = GraphReconstructionEval(
                    graph_wrapper, emb_obj, report_warnings=False
                )
                avg_f1.append(gre.getF1())

            print(f"Hybrid Walker Final F1: {np.mean(avg_f1):.4f}")

        except Exception as e:
            print(f"Failed to process dataset {DATASET}. Error: {e}")
