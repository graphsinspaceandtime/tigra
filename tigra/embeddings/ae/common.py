from abc import ABC, abstractmethod

import networkx as nx

from tigra.core.dgraph import DynamicGraph

from torch import nn
from torch.nn import functional as F
from typing import Sequence
import numpy as np
import torch

class PenaltyTensorInitializer(ABC):
    @abstractmethod
    def initialize(self, x_cur_batch, **kwargs) -> torch.Tensor:
        pass

class BetaPenaltyTensorInitializer(PenaltyTensorInitializer):
    beta: float

    def __init__(self, beta: float):
        self.beta = beta

    def initialize(self, x_cur_batch, **kwargs):
        y_batch = torch.ones(
            x_cur_batch.shape, device=x_cur_batch.device
        )  # penalty tensor for x_cur_batch
        y_batch[x_cur_batch != 0] = self.beta
        return y_batch

class GraphLIDPenaltyTensorInitializer(PenaltyTensorInitializer):
    dynamic_graph: DynamicGraph
    k: int
    cutoff: int

    def __init__(self, dynamic_graph: DynamicGraph, k: int = 10, cutoff: int = 10):
        self.dynamic_graph = dynamic_graph
        self.k = k
        self.cutoff = cutoff
        self.snapshot_lid_cache = {}

        for idx, graph in enumerate(self.dynamic_graph.get_nx_graph_sequence()):
            lid_tensor = self._compute_snapshot_lid(graph)
            self.snapshot_lid_cache[idx] = lid_tensor
    
    def _compute_lid_node(self, G, node) -> float:
        """Compute graph-based LID for a single node using k nearest distances"""
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=self.cutoff)
        dists = sorted(lengths.values())
        if len(dists) <= self.k:
            return 0.0

        dists = dists[1 : self.k + 1]  # exclude distance to itself (0)
        d_k = dists[-1]
        return -self.k / sum(np.log(d / d_k) for d in dists)
    
    def _compute_snapshot_lid(self, G):
        lid_dict = {v: self._compute_lid_node(G, v) for v in G.nodes()}
        lid = torch.tensor([lid_dict[v] for v in sorted(lid_dict.keys())], dtype=torch.float32)
        # normalize
        lid = (lid - lid.mean()) / (lid.std() + 1e-8)
        return lid



    def initialize(self, x_cur_batch, **kwargs):
        snapshot_index = kwargs.get("snapshot_index", None)
        batch_node_ids = kwargs.get("batch_node_ids", None)

        if snapshot_index is None:
            raise ValueError("snapshot_index must be provided for GraphLIDPenaltyTensorInitializer")
        if batch_node_ids is None:
            raise ValueError("batch_node_ids must be provided for GraphLIDPenaltyTensorInitializer")
        if snapshot_index not in self.snapshot_lid_cache:
            raise ValueError(f"Snapshot index {snapshot_index} not found in cache")

        snapshot_lid = self.snapshot_lid_cache[snapshot_index]

        y_batch = snapshot_lid[batch_node_ids].to(x_cur_batch.device)

        return y_batch

class BatchGenerator:
    node_list: list
    node_num: int
    batch_size: int
    look_back: int
    penalty_initializer: PenaltyTensorInitializer
    shuffle: bool
    has_cuda: bool

    def __init__(
        self, node_list, batch_size, look_back, penalty_initializer: PenaltyTensorInitializer, shuffle=True, has_cuda=False
    ):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.look_back = look_back
        self.penalty_initializer = penalty_initializer
        self.shuffle = shuffle
        self.has_cuda = has_cuda

    def generate(self, graph_list):
        # Number of graphs
        graph_num = len(graph_list)
        # Training size = graphs that can be learned (excluding look_back graphs)
        train_size = graph_num - self.look_back
        assert train_size > 0
        # Total number of nodes in all training graphs
        all_node_num = self.node_num * train_size
        # Number of batches (each batch has batch_size samples)
        batch_num = all_node_num // self.batch_size
        if all_node_num % self.batch_size != 0:
            batch_num += 1
        # Create a list of all node indices in all training graphs
        node_indices = np.arange(all_node_num)

        # Shuffle the node indices if required
        if self.shuffle:
            np.random.shuffle(node_indices)
        counter = 0
        while True:
            # Get the indices for the current batch
            batch_indices = node_indices[
                self.batch_size
                * counter : min(all_node_num, self.batch_size * (counter + 1))
            ]
            # Initialize tensor for previous graphs
            x_pre_batch = torch.zeros((self.batch_size, self.look_back, self.node_num))
            x_pre_batch = x_pre_batch.cuda() if self.has_cuda else x_pre_batch

            # Initialize tensor for current graph
            x_cur_batch = torch.zeros(
                (self.batch_size, self.node_num), device=x_pre_batch.device
            )

            for idx, record_id in enumerate(batch_indices):
                graph_idx = record_id // self.node_num
                node_idx = record_id % self.node_num
                for step in range(self.look_back):
                    # graph is a scipy.sparse.lil_matrix
                    pre_tensor = torch.tensor(
                        graph_list[graph_idx + step][node_idx, :].toarray(),
                        device=x_pre_batch.device,
                    )
                    x_pre_batch[idx, step, :] = pre_tensor
                # graph is a scipy.sparse.lil_matrix
                cur_tensor = torch.tensor(
                    graph_list[graph_idx + self.look_back][node_idx, :].toarray(),
                    device=x_pre_batch.device,
                )
                x_cur_batch[idx] = cur_tensor

            # Initialize penalty tensor for current graph
            y_batch = self.penalty_initializer.initialize(x_cur_batch, batch_node_ids=batch_indices, snapshot_index=graph_idx + self.look_back)
            counter += 1
            yield x_pre_batch, x_cur_batch, y_batch

            if counter == batch_num:
                if self.shuffle:
                    np.random.shuffle(node_indices)
                counter = 0


class BatchPredictor:
    node_list: list
    node_num: int
    batch_size: int
    has_cuda: bool

    def __init__(self, node_list, batch_size, has_cuda=False):
        self.node_list = node_list
        self.node_num = len(node_list)
        self.batch_size = batch_size
        self.has_cuda = has_cuda

    def get_predict_res(
        self,
        graph_list,
        model,
        batch_indices,
        counter,
        look_back,
        embedding_mat,
        x_pred,
    ):
        batch_size = len(batch_indices)
        x_pre_batches = torch.zeros((batch_size, look_back, self.node_num))
        x_pre_batches = x_pre_batches.cuda() if self.has_cuda else x_pre_batches

        for idx, node_idx in enumerate(batch_indices):
            for step in range(look_back):
                # graph is a scipy.sparse.lil_matrix
                pre_tensor = torch.tensor(
                    graph_list[step][node_idx, :].toarray(), device=x_pre_batches.device
                )
                x_pre_batches[idx, step, :] = pre_tensor
        # DynAE uses 2D tensor as its input
        if model.method_name == "DynAE":
            x_pre_batches = x_pre_batches.reshape(batch_size, -1)
        embedding_mat_batch, x_pred_batch = model(x_pre_batches)
        if counter:
            embedding_mat = torch.cat((embedding_mat, embedding_mat_batch), dim=0)
            x_pred = torch.cat((x_pred, x_pred_batch), dim=0)
        else:
            embedding_mat = embedding_mat_batch
            x_pred = x_pred_batch
        return embedding_mat, x_pred

    def predict(self, model, graph_list):
        look_back = len(graph_list)
        counter = 0
        embedding_mat, x_pred = 0, 0
        batch_num = self.node_num // self.batch_size

        while counter < batch_num:
            batch_indices = range(
                self.batch_size * counter, self.batch_size * (counter + 1)
            )
            embedding_mat, x_pred = self.get_predict_res(
                graph_list,
                model,
                batch_indices,
                counter,
                look_back,
                embedding_mat,
                x_pred,
            )
            counter += 1
        # has a remaining batch
        if self.node_num % self.batch_size != 0:
            remain_indices = range(self.batch_size * counter, self.node_num)
            embedding_mat, x_pred = self.get_predict_res(
                graph_list,
                model,
                remain_indices,
                counter,
                look_back,
                embedding_mat,
                x_pred,
            )
        return embedding_mat, x_pred


class DynamicEmbedder:
    full_node_list: Sequence[int]
    node_num: int
    batch_generator: BatchGenerator
    device: torch.device

    def __init__(
        self,
        node_list: Sequence[int],
        model: nn.Module,
        loss,
        batch_generator: BatchGenerator,
        device: torch.device,
    ):
        self.model = model
        self.loss = loss
        self.batch_generator = batch_generator
        self.device = device

        self.full_node_list = node_list
        self.node_num = len(self.full_node_list)  # node num

    def prepare(self, lr=1e-3, weight_decay=0.0):
        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.8, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        optimizer.zero_grad()
        return self.model, self.loss, optimizer

    def get_batch_info(self, adj_list, model):
        graph_num = len(adj_list)
        batch_size = self.batch_generator.batch_size

        train_size = graph_num - self.batch_generator.look_back
        element_num = self.node_num * train_size

        batch_num = element_num // batch_size
        if element_num % batch_size != 0:
            batch_num += 1
        return batch_size, batch_num, train_size

    def get_model_res(self, model, generator):
        batch_size = self.batch_generator.batch_size

        x_pre_batches, x_cur_batch, y_batch = next(generator)
        # DynAE uses 2D tensor as its input
        if model.method_name == "DynAE":
            x_pre_batches = x_pre_batches.reshape(batch_size, -1)
        _, x_pred_batch = model(x_pre_batches)
        loss_input_list = [x_pred_batch, x_cur_batch, y_batch]

        return loss_input_list
