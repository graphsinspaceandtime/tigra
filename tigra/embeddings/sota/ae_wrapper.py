import sys
from time import time
from typing import List

sys.path.append("../../../..")

from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph

# from dynamicgem.embedding.dynAERNN import DynAERNN as DynAERNNDynGem

from .DynamicGEM.dynamicgem.embedding.dynAERNN import DynAERNN as DynAERNNDynGem


class DynAERNN(Embedding):
    # TODO: What is length???
    def __init__(self, dim, lookback=2, length=7):
        super().__init__(dim)
        self.lookback = lookback
        self.length = length

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                f"{type(self).__name__} works with DynamicAggregateGraph objects."
            )

        # TODO: Check graph direction

        # TODO: Does everything need to be hardcoded?
        embedding = DynAERNNDynGem(
            d=self.dim,
            beta=5,
            n_prev_graphs=self.lookback,
            nu1=1e-6,
            nu2=1e-6,
            n_aeunits=[500, 300],
            n_lstmunits=[500, self.dim],
            rho=0.3,
            n_iter=10,
            xeta=1e-3,
            n_batch=100,
            modelfile=[
                "/tmp/intermediate_enc_model_dynAERNN.json",
                "/tmp/intermediate_dec_model_dynAERNN.json",
            ],
            weightfile=[
                "/tmp/intermediate_enc_weights_dynAERNN.hdf5",
                "/tmp/intermediate_dec_weights_dynAERNN.hdf5",
            ],
            savefilesuffix="testing",
        )

        embs = []

        t1 = time()
        for temp_var in range(self.lookback + 1, self.length + 1):
            emb, _ = embedding.learn_embeddings(
                data.get_scipy_sparse_matrices()[:temp_var]
            )
            embs.append(emb)
        print(embedding._method_name + ":\n\tTraining time: %f" % (time() - t1))

        self._embs = embs

        return []
