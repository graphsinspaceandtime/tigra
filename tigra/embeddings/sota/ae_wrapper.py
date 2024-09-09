from typing import List
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.dynae.dynae import DynAE
from tigra.embeddings.sota.dynae.dynrnn import DynRNN


class DynamicAutoEncoder(Embedding):
    def __init__(self, dim, lookback):
        self._d = dim
        self._lookback = lookback

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                "{} works with DynamicSparseMatrixGraph objects.".format(
                    type(self).__name__
                )
            )

        graphs = data.get_scipy_sparse_matrices()

        # TODO: Use hyperparameters instead of hardcoding
        dynae = DynAE(
            d=self._d,
            beta=5,
            n_prev_graphs=self._lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_units=[
                500,
                300,
            ],
            rho=0.3,
            n_iter=4,
            xeta=1e-4,
            n_batch=100,
        )

        dynae.learn_embedding(graphs)

        embedding = dynae.get_embedding()

        singleton = SingletonEmbedding()

        for node in range(graphs[0].shape[0]):
            singleton.add_embedding_vector(node, embedding[node])

        return [singleton]


class DynamicRecurrentNeuralNetwork(Embedding):
    def __init__(self, dim, lookback):
        self._d = dim
        self._lookback = lookback

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                "{} works with DynamicSparseMatrixGraph objects.".format(
                    type(self).__name__
                )
            )

        graphs = data.get_scipy_sparse_matrices()

        # TODO: Use hyperparameters instead of hardcoding
        dynrnn = DynRNN(
            d=self._d,
            beta=5,
            n_prev_graphs=self._lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_enc_units=[500, 300],
            n_dec_units=[500, 300],
            rho=0.3,
            n_iter=2,
            xeta=1e-3,
            n_batch=100,
        )

        dynrnn.learn_embedding(graphs)

        embedding = dynrnn.get_embedding()

        singleton = SingletonEmbedding()

        for node in range(graphs[0].shape[0]):
            singleton.add_embedding_vector(node, embedding[node])

        return [singleton]
