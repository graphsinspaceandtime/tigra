from typing import List, Union
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.CTGCN.baseline.gcrn import GCRN as _GCRN
from tigra.embeddings.sota.CTGCN.metrics import NegativeSamplingLoss
from tigra.embeddings.sota.CTGCN.helper import DataLoader
import numpy as np
from tigra.embeddings.sota.CTGCN.embedding import (
    SupervisedEmbedding,
    UnsupervisedEmbedding,
)


class Gnn(Embedding):
    n_units: List[int]
    feature_dim: int
    hidden_dim: int
    feature_pre: bool
    layer_num: int
    dropout: float
    duration: int
    rnn_type: str
    bias: bool
    has_cuda: bool
    epochs: int
    learning_rate: float
    batch_size: int

    def __init__(
        self,
        dim: int,
        n_units: List[int],
        feature_dim: int,
        hidden_dim: int,
        feature_pre,
        layer_num,
        dropout,
        duration,
        rnn_type,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        bias=True,
        has_cuda=False,
    ):
        super(Gnn, self).__init__(dim)
        self.n_units = n_units
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        self.duration = duration
        self.rnn_type = rnn_type
        self.bias = bias
        self.has_cuda = has_cuda
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def __construct(
        self,
        data: DynamicSparseMatrixGraph,
        graphs,
        prediction_index,
        num_nodes,
        node_list,
        node_pair_list,
        neg_freq_list,
        neg_num,
        Q,
    ) -> SingletonEmbedding:

        model = _GCRN(
            input_dim=node_list,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.dim,
            feature_pre=self.feature_pre,
            layer_num=self.layer_num,
            dropout=self.dropout,
            bias=self.bias,
            duration=self.duration,
        )

        loss = NegativeSamplingLoss(node_pair_list, neg_freq_list, neg_num=neg_num, Q=Q)

        trainer = UnsupervisedEmbedding(
            node_list=node_list,
            model=model,
            loss=loss,
            has_cuda=self.has_cuda,
            base_path="",
            origin_folder="",
            embedding_folder="",
            model_folder="",
        )
        adj_list = []
        x_list = []
        edge_list = []
        node_dist_list = []

        embedding = trainer.learn_embedding(
            adj_list,
            x_list,
            edge_list=edge_list,
            node_dist_list=node_dist_list,
            epoch=self.epochs,
            batch_size=self.batch_size,
            lr=self.learning_rate,
            shuffle=True,
            export=False,
            model_file=None,
        )

        singleton = SingletonEmbedding()

        learn_graph = data.get_matrix_as_graph(prediction_index)
        node_list = learn_graph.G.nodes()

        embedding = embedding or []

        for node in node_list:
            singleton.add_embedding_vector(node, embedding[node].tolist())

        return singleton

    def construct(self, data) -> List[Union[SingletonEmbedding, None]]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                "{} works with DynamicSparseMatrixGraph objects.".format(
                    type(self).__name__
                )
            )

        graphs = data.get_scipy_sparse_matrices()
        num_graphs = len(graphs)
        num_nodes = graphs[0].shape[0]
        node_list = list(range(num_nodes))
        node_pair_list = []
        node_freq_list = []
        neg_sample_num = 0
        Q = 0

        embs: List[Union[SingletonEmbedding, None]] = []
        for prediction_index in range(num_graphs):
            embs.append(
                self.__construct(
                    data,
                    graphs,
                    prediction_index,
                    num_nodes,
                    node_list,
                    node_pair_list,
                    node_freq_list,
                    neg_sample_num,
                    Q,
                )
            )

        return embs
