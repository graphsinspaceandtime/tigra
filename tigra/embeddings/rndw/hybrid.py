import numpy as np
import networkx as nx
import torch
from tigra.embeddings.rndw.rndwbase import RWEmbEgoDelta, RandomWalkSampler


class EuclideanHybridSampler(RandomWalkSampler):
    """
    TIGRA Random Walker that uses standard network edges for EXPLOITATION,
    and Euclidean Autoencoder Probabilities for EXPLORATION (Teleportation).
    """

    def __init__(
        self,
        graph: nx.Graph,
        nodes,
        ae_prob_matrix: np.ndarray,
        node_to_idx: dict,
        alpha=0.8,  # 80% walk on real edges, 20% AE teleportation
        num_walks=10,
        walk_length=32,
        num_workers=1,
    ):
        super().__init__(graph, nodes)
        self.nw = num_walks
        self.wl = walk_length
        self.num_workers = num_workers
        self.alpha = alpha

        self.ae_prob_matrix = ae_prob_matrix
        self.node_to_idx = node_to_idx
        self.idx_to_node = {v: k for k, v in node_to_idx.items()}

        # --- Node Masking (Prevent walking into the void) ---
        if self.ae_prob_matrix is not None:
            valid_nodes = set(self.G.nodes())
            valid_indices = set(
                self.node_to_idx[n] for n in valid_nodes if n in self.node_to_idx
            )

            invalid_indices = [
                i for i in range(len(self.idx_to_node)) if i not in valid_indices
            ]

            # Zero out probabilities for nodes that don't exist in THIS snapshot
            self.ae_prob_matrix[:, invalid_indices] = 0.0

            # Re-normalize rows to sum to 1.0
            row_sums = self.ae_prob_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.ae_prob_matrix = self.ae_prob_matrix / row_sums

    def select_next_node(self, start_node, current_node, prev_node, neighbours):
        # 1. EXPLOITATION: Walk along actual graph edges (Hub-weighted)
        if len(neighbours) > 0 and (
            np.random.rand() < self.alpha or self.ae_prob_matrix is None
        ):
            total_hub = sum([self.G.degree[n] for n in neighbours])
            if total_hub == 0:
                return np.random.choice(neighbours)

            probabilities = [self.G.degree[n] / total_hub for n in neighbours]
            return np.random.choice(neighbours, 1, p=probabilities)[0]

        # 2. EXPLORATION: Teleport using Euclidean Latent Proximity!
        else:
            if current_node not in self.node_to_idx:
                return np.random.choice(list(self.G.nodes()))

            curr_idx = self.node_to_idx[current_node]
            ae_probs = self.ae_prob_matrix[curr_idx]

            if ae_probs.sum() == 0:
                return np.random.choice(list(self.G.nodes()))

            next_idx = np.random.choice(len(ae_probs), p=ae_probs)
            return self.idx_to_node[next_idx]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class DynEuclideanHybridWalk(RWEmbEgoDelta):
    """
    TIGRA Wrapper: Feeds the pre-computed Autoencoder probability matrices
    into the Hybrid Sampler sequentially.
    """

    def __init__(
        self,
        dim,
        probability_matrices: dict,
        node_to_idx: dict,
        alpha=0.8,
        walk_length=30,
    ):
        super().__init__(dim)
        self.probability_matrices = probability_matrices
        self.node_to_idx = node_to_idx
        self.alpha = alpha
        self.walk_length = walk_length
        self.current_snapshot_idx = 0

    def create_random_walk_sampler(self, graph: nx.Graph, nodes):
        # Fetch the AE probability matrix for time t
        ae_prob_matrix = self.probability_matrices.get(self.current_snapshot_idx, None)

        sampler = EuclideanHybridSampler(
            graph,
            nodes,
            ae_prob_matrix=ae_prob_matrix,
            node_to_idx=self.node_to_idx,
            alpha=self.alpha,
            walk_length=self.walk_length,
        )

        self.current_snapshot_idx += 1
        return sampler
