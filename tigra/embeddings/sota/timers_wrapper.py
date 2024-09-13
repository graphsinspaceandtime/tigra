from typing import List
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.CTGCN.baseline.timers import our_timers
import numpy as np


class Timers(Embedding):
    theta: float
    update: bool

    def __init__(self, dim: int, theta=0.17, update=True):
        super(Timers, self).__init__(dim // 2)
        self.theta = theta
        self.update = update

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                "{} works with DynamicSparseMatrixGraph objects.".format(
                    type(self).__name__
                )
            )

        graphs = data.get_scipy_sparse_matrices()

        U_cur, V_cur = our_timers(self.dim, graphs, self.theta, self.update)
        embedding = np.hstack((U_cur, V_cur))

        singleton = SingletonEmbedding()
        for node in range(graphs[0].shape[0]):
            singleton.add_embedding_vector(node, embedding[node])

        return [singleton]
