from abc import ABC, abstractmethod

from typing import Sequence, Union, List

from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph

class AutoEncoderBase(Embedding):
    """
    Base class for auto encoder embedding methods
    """
    name: str
    look_back: int
    beta: float
    nu1: float
    nu2: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    last_only: bool
    bias: bool
    shuffle: bool
    has_cuda: bool
    verbose: int

    def __init__(
        self,
        dim,
        look_back: int,
        beta: float,
        nu1: float,
        nu2: float,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        last_only=True,
        bias=True,
        shuffle=True,
        has_cuda=False,
        verbose=0,
        **kawrgs,  # collect redundant arguments
    ):
        super().__init__(dim)
        self.look_back = look_back
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.last_only = last_only
        self.bias = bias
        self.shuffle = shuffle
        self.has_cuda = has_cuda
        self.verbose = verbose

    @abstractmethod
    def __construct(self, data: DynamicSparseMatrixGraph, prediction_index: int, num_nodes: int, node_list: Sequence[int]):
        pass

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

        embs: List[Union[SingletonEmbedding, None]] = []

        if self.last_only:
            embs = [
                self.__construct(data, num_graphs - 1, num_nodes, node_list)
            ]
        else:
            embs = [None] * self.look_back
            for prediction_index in range(self.look_back, num_graphs):
                embs.append(
                    self.__construct(
                        data, prediction_index, num_nodes, node_list
                    )
                )

        return embs


class SymetricAEBase(AutoEncoderBase):
    n_units: Sequence[int]

    def __init__(self, dim, look_back, beta, nu1, nu2, batch_size, epochs, learning_rate, weight_decay, n_units: Sequence[int], last_only=True, bias=True, shuffle=True, has_cuda=False, verbose=0, **kawrgs):
        super().__init__(dim, look_back, beta, nu1, nu2, batch_size, epochs, learning_rate, weight_decay, last_only, bias, shuffle, has_cuda, verbose, **kawrgs)
        self.n_units = n_units

class AsymetricAEBase(AutoEncoderBase):
    ae_units: Sequence[int]
    rnn_units: Sequence[int]

    def __init__(self, dim, look_back, beta, nu1, nu2, batch_size, epochs, learning_rate, weight_decay, ae_units: Sequence[int], rnn_units: Sequence[int], last_only=True, bias=True, shuffle=True, has_cuda=False, verbose=0, **kawrgs):
        super().__init__(dim, look_back, beta, nu1, nu2, batch_size, epochs, learning_rate, weight_decay, last_only, bias, shuffle, has_cuda, verbose, **kawrgs)
        self.ae_units = ae_units
        self.rnn_units = rnn_units
