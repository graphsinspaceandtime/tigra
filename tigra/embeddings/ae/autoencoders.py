import sys
import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from tigra.embeddings.embedding import Embedding, SingletonEmbedding

from tigra.evaluation.lideval import NCLIDEstimator


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        support = self.linear(x)
        return torch.spmm(adj, support)


class BasicGAEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(BasicGAEModel, self).__init__()

        self.gc1 = GCNLayer(input_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, latent_dim)

    def encode(self, x, adj):
        hidden = F.relu(self.gc1(x, adj))
        z = self.gc2(hidden, adj)
        return z

    def decode(self, z):
        return torch.mm(z, z.t())

    def forward(self, x, adj):
        z = self.encode(x, adj)
        adj_recon = self.decode(z)
        return adj_recon, z


class AbstractDynAE(Embedding):
    def __init__(self, dim=32, hidden_dim=64, epochs=200, lr=0.01, device=None):
        super().__init__(dim)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.node_to_idx = {}
        self.num_global_nodes = 0

        print(f"DynAE initialized on: {self.device}")

    def _build_global_vocab(self, snapshots):
        """Finds all unique nodes across all time steps."""
        all_nodes = set()
        for G_raw in snapshots:
            G = G_raw.G if hasattr(G_raw, "G") else G_raw
            all_nodes.update(G.nodes())

        global_nodes_list = sorted(list(all_nodes))
        self.node_to_idx = {node: idx for idx, node in enumerate(global_nodes_list)}
        self.num_global_nodes = len(global_nodes_list)

    def _prepare_input(self, G):
        """Converts NX graph to PyTorch sparse tensors."""
        nodelist = list(G.nodes())
        adj = nx.adjacency_matrix(G, nodelist=nodelist)

        adj_loop = adj + sp.eye(len(nodelist), format="csr")
        rowsum = np.array(adj_loop.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        adj_normalized = adj_loop.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

        adj_coo = adj_normalized.tocoo()
        indices = np.vstack((adj_coo.row, adj_coo.col))
        values = adj_coo.data
        adj_tensor = torch.sparse_coo_tensor(
            torch.LongTensor(indices),
            torch.FloatTensor(values),
            torch.Size(adj_coo.shape),
            device=self.device,
        )

        features = torch.zeros(
            (len(nodelist), self.num_global_nodes), device=self.device
        )
        for idx, node_id in enumerate(nodelist):
            features[idx, self.node_to_idx[node_id]] = 1.0

        adj_orig = nx.adjacency_matrix(G, nodelist=nodelist)
        adj_dense = torch.tensor(adj_orig.todense(), dtype=torch.float32).to(
            self.device
        )

        return features, adj_tensor, adj_dense, nodelist

    def compute_loss(self, recon_adj, target_adj, G, nodelist):
        num_edges = target_adj.sum().item()
        num_nodes = target_adj.shape[0]

        pos_weight_val = (num_nodes**2 - num_edges) / (num_edges + 1e-9)
        pos_weight = torch.tensor(pos_weight_val, device=self.device)

        loss = F.binary_cross_entropy_with_logits(
            recon_adj, target_adj, pos_weight=pos_weight
        )
        return loss

    def train_snapshot(self, G):
        """Trains the GAE on a single snapshot."""
        features, adj_norm, target_adj, nodelist = self._prepare_input(G)

        model = BasicGAEModel(features.shape[1], self.hidden_dim, self.dim).to(
            self.device
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            recon_adj, z = model(features, adj_norm)

            loss = self.compute_loss(recon_adj, target_adj, G, nodelist)

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            z_final = model.encode(features, adj_norm)

        return z_final.cpu().numpy()

    def construct(self, dynamic_graph):
        generated_embeddings = []

        if hasattr(dynamic_graph, "get_graph_sequence"):
            snapshots = dynamic_graph.get_graph_sequence()
        else:
            snapshots = [
                dynamic_graph.get_matrix_as_graph(t)
                for t in range(dynamic_graph.get_num_snaps())
            ]

        self._build_global_vocab(snapshots)

        for t, G_raw in enumerate(snapshots):
            G = G_raw.G if hasattr(G_raw, "G") else G_raw

            if len(G.nodes) < 2:
                generated_embeddings.append(None)
                continue

            z_final = self.train_snapshot(G)

            singleton = SingletonEmbedding()
            for idx, node_id in enumerate(G.nodes()):
                singleton.add_embedding_vector(node_id, z_final[idx].tolist())

            generated_embeddings.append(singleton)

        return generated_embeddings


class BasicDynAE(AbstractDynAE):
    pass


class TunedHubDynAE(AbstractDynAE):
    def __init__(
        self, dim=32, hidden_dim=64, epochs=200, lr=0.01, device=None, hub_alpha=1.0
    ):
        super().__init__(dim, hidden_dim, epochs, lr, device)
        self.hub_alpha = hub_alpha

    def compute_loss(self, recon_adj, target_adj, G, nodelist):
        num_edges = target_adj.sum().item()
        num_nodes = target_adj.shape[0]
        pos_weight_val = (num_nodes**2 - num_edges) / (num_edges + 1e-9)

        base_weight_matrix = torch.ones_like(target_adj)
        base_weight_matrix[target_adj == 1] = pos_weight_val

        raw_loss = F.binary_cross_entropy_with_logits(
            recon_adj, target_adj, reduction="none"
        )

        degrees = torch.tensor(
            [G.degree(n) for n in nodelist], device=self.device, dtype=torch.float32
        )
        deg_outer = torch.outer(degrees, degrees)

        max_deg_interaction = deg_outer.max() + 1e-9
        normalized_hub_score = deg_outer / max_deg_interaction

        hub_multiplier = 1.0 + (self.hub_alpha * normalized_hub_score)

        final_weighted_loss = raw_loss * base_weight_matrix * hub_multiplier

        return final_weighted_loss.mean()


class HubDynAE(AbstractDynAE):
    """
    Extended Autoencoder that penalizes the model based on Node Degree (Hubness).
    """

    def compute_loss(self, recon_adj, target_adj, G, nodelist):
        raw_loss = F.binary_cross_entropy_with_logits(
            recon_adj, target_adj, reduction="none"
        )

        degrees = torch.tensor(
            [G.degree(n) for n in nodelist], device=self.device, dtype=torch.float32
        )

        hub_weights = torch.log1p(torch.outer(degrees, degrees))
        hub_weights = hub_weights / (hub_weights.mean() + 1e-9)

        custom_weighted_loss = raw_loss * hub_weights

        return custom_weighted_loss.mean()


class LIDDynAE(AbstractDynAE):
    """
    Penalizes the Autoencoder for missing edges within complex,
    high-dimensional local neighborhoods (Natural Communities).
    """

    def __init__(
        self, dim=32, hidden_dim=64, epochs=200, lr=0.01, device=None, lid_alpha=1.0
    ):
        super().__init__(dim, hidden_dim, epochs, lr, device)
        self.lid_alpha = lid_alpha
        self.current_lid_tensor = None

    def train_snapshot(self, G):
        lid_estimator = NCLIDEstimator(G, alpha=1.0)
        lid_estimator.estimate_lids()

        nodelist = list(G.nodes())
        lid_scores = [lid_estimator.get_lid(n) for n in nodelist]

        self.current_lid_tensor = torch.tensor(
            lid_scores, device=self.device, dtype=torch.float32
        )

        return super().train_snapshot(G)

    def compute_loss(self, recon_adj, target_adj, G, nodelist):
        num_edges = target_adj.sum().item()
        num_nodes = target_adj.shape[0]
        pos_weight_val = (num_nodes**2 - num_edges) / (num_edges + 1e-9)

        base_weights = torch.ones_like(target_adj)
        base_weights[target_adj == 1] = pos_weight_val

        raw_loss = F.binary_cross_entropy_with_logits(
            recon_adj, target_adj, reduction="none"
        )

        lid_outer = torch.outer(self.current_lid_tensor, self.current_lid_tensor)

        max_lid = lid_outer.max() + 1e-9
        normalized_lid = lid_outer / max_lid

        lid_multiplier = 1.0 + (self.lid_alpha * normalized_lid)

        final_loss = raw_loss * base_weights * lid_multiplier

        return final_loss.mean()


class EuclideanGAEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EuclideanGAEModel, self).__init__()
        self.gc1 = GCNLayer(input_dim, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, latent_dim)

        self.bias = nn.Parameter(torch.tensor([0.0]))

    def encode(self, x, adj):
        hidden = torch.tanh(self.gc1(x, adj))
        z = self.gc2(hidden, adj)
        return z

    def decode(self, z):
        z_sq = torch.sum(z**2, dim=1, keepdim=True)
        dist_sq = z_sq + z_sq.t() - 2.0 * torch.mm(z, z.t())

        dist_sq = torch.clamp(dist_sq, min=0.0)

        logits = self.bias - dist_sq
        return logits

    def forward(self, x, adj):
        z = self.encode(x, adj)
        logits = self.decode(z)
        return logits, z


class DistanceDynAE(AbstractDynAE):
    """
    Autoencoder explicitly designed to optimize Euclidean Distance,
    """

    def train_snapshot(self, G):
        features, adj_norm, target_adj, nodelist = self._prepare_input(G)

        model = EuclideanGAEModel(features.shape[1], self.hidden_dim, self.dim).to(
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

        return z_final.cpu().numpy()


class HubDistanceDynAE(DistanceDynAE):
    """
    Euclidean Autoencoder with Hub Penalty.
    """

    def __init__(
        self, dim=32, hidden_dim=64, epochs=300, lr=0.01, device=None, hub_alpha=1.0
    ):
        super().__init__(dim, hidden_dim, epochs, lr, device)
        self.hub_alpha = hub_alpha

    def compute_loss(self, recon_logits, target_adj, G, nodelist):
        num_edges = target_adj.sum().item()
        num_nodes = target_adj.shape[0]
        pos_weight_val = (num_nodes**2 - num_edges) / (num_edges + 1e-9)
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


class LIDDistanceDynAE(DistanceDynAE):
    """
    Euclidean Autoencoder with LID (Natural Community) Penalty.
    """

    def __init__(
        self,
        dim=32,
        hidden_dim=64,
        epochs=300,
        lr=0.01,
        device=None,
        lid_alpha=1.0,
        precomputed_lids=None,
    ):
        super().__init__(dim, hidden_dim, epochs, lr, device)
        self.lid_alpha = lid_alpha
        self.precomputed_lids = precomputed_lids
        self.current_snap_idx = 0

    def train_snapshot(self, G):
        self.current_lid_tensor = self.precomputed_lids[self.current_snap_idx].to(
            self.device
        )
        self.current_snap_idx += 1

        return super().train_snapshot(G)

    def compute_loss(self, recon_logits, target_adj, G, nodelist):
        num_edges = target_adj.sum().item()
        num_nodes = target_adj.shape[0]
        pos_weight_val = (num_nodes**2 - num_edges) / (num_edges + 1e-9)
        base_weights = torch.ones_like(target_adj)
        base_weights[target_adj == 1] = pos_weight_val

        raw_loss = F.binary_cross_entropy_with_logits(
            recon_logits, target_adj, reduction="none"
        )

        lid_outer = torch.outer(self.current_lid_tensor, self.current_lid_tensor)
        normalized_lid = lid_outer / (lid_outer.max() + 1e-9)
        lid_multiplier = 1.0 + (self.lid_alpha * normalized_lid)

        return (raw_loss * base_weights * lid_multiplier).mean()
