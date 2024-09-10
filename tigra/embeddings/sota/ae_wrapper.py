from typing import List
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.dynae.dynae import DynAE, OurDynAE, OurPyTDynAE
from tigra.embeddings.sota.dynae.dnn_utils import (
    DynGraph2VecLoss,
    BatchGenerator,
    BatchPredictor,
)
from tigra.embeddings.sota.dynae.dynrnn import DynRNN
import scipy as sp
import torch


class OurDynamicAutoEncoder(Embedding):
    def __init__(self, dim, look_back):
        self.dim = dim
        self.look_back = look_back

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                "{} works with DynamicSparseMatrixGraph objects.".format(
                    type(self).__name__
                )
            )

        graphs = data.get_scipy_sparse_matrices()

        num_nodes = graphs[0].shape[0]
        epoch = 4
        lr = 1e-3
        weight_decay = 0.0

        self.model = OurPyTDynAE(num_nodes, self.dim, 2, [500, 300])
        self.loss = DynGraph2VecLoss(beta=5, nu1=1e-6, nu2=1e-6)
        self.batch_generator = BatchGenerator(
            node_list=[None] * num_nodes,
            batch_size=4,
            look_back=self.look_back,
            beta=5,
            shuffle=True,
            has_cuda=False,
        )
        self.batch_predictor = BatchPredictor(
            node_list=[None] * num_nodes,
            batch_size=4,
            has_cuda=False,
        )

        batch_size, batch_num, train_size = self.__get_batch_info(graphs)

        for i in range(epoch):
            for j in range(batch_num):
                generator = self.batch_generator.generate(graphs)
                loss_input_list = self.__get_model_res(generator)
                loss = self.loss(model, loss_input_list)

        # TODO: Use hyperparameters instead of hardcoding
        # num_nodes = 167
        # dynae = OurDynAE(num_nodes, self._d, [500, 300], 1e-6, 1e-6, "relu")

        # dynae.learn_embedding(graphs)

        # embedding = dynae.get_embedding()

        # singleton = SingletonEmbedding()

        # for node in range(graphs[0].shape[0]):
        #     singleton.add_embedding_vector(node, embedding[node])

        # return [singleton]

    def __get_batch_info(self, adj_list):
        graph_num = len(adj_list)
        batch_size = self.batch_generator.batch_size
        train_size = graph_num - self.batch_generator.look_back
        element_num = self.node_num * train_size
        batch_num = element_num // batch_size
        if element_num % batch_size != 0:
            batch_num += 1
        return batch_size, batch_num, train_size

    def __get_model_res(self, generator):
        batch_size = self.batch_generator.batch_size
        x_pre_batches, x_cur_batch, y_batch = next(generator)
        # DynAE uses 2D tensor as its input
        x_pre_batches = x_pre_batches.reshape(batch_size, -1)
        _, x_pred_batch = self.model(x_pre_batches)
        loss_input_list = [x_pred_batch, x_cur_batch, y_batch]
        return loss_input_list


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
