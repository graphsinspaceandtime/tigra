from typing import List, Union
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.CTGCN.baseline.dynAE import (
    BatchGenerator,
    BatchPredictor,
    DynGraph2VecLoss,
    DynamicEmbedding,
)
from tigra.embeddings.sota.CTGCN.baseline.dynAERNN import DynAERNN as _DynAERNN


class DynAERNN(Embedding):
    ae_units: List[int]
    rnn_units: List[int]
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
        dim: int,
        ae_units: List[int],
        rnn_units: List[int],
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
        super(DynAERNN, self).__init__(dim)
        self.ae_units = ae_units
        self.rnn_units = rnn_units
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

    def __construct(
        self,
        data: DynamicSparseMatrixGraph,
        graphs,
        prediction_index,
        num_nodes,
        node_list,
    ) -> SingletonEmbedding:
        start_index = prediction_index - self.look_back
        train_graphs = graphs[start_index : prediction_index + 1]

        model = _DynAERNN(
            input_dim=num_nodes,
            output_dim=self.dim,
            look_back=self.look_back,
            ae_units=self.ae_units,
            rnn_units=self.rnn_units,
            bias=self.bias,
        )
        loss = DynGraph2VecLoss(beta=self.beta, nu1=self.nu1, nu2=self.nu2)
        batch_generator = BatchGenerator(
            node_list=node_list,
            batch_size=self.batch_size,
            look_back=self.look_back,
            beta=self.beta,
            shuffle=self.shuffle,
            has_cuda=self.has_cuda,
        )
        batch_predictor = BatchPredictor(
            node_list=node_list, batch_size=self.batch_size, has_cuda=self.has_cuda
        )

        trainer = DynamicEmbedding(
            node_list=node_list,
            model=model,
            loss=loss,
            batch_generator=batch_generator,
            batch_predictor=batch_predictor,
            has_cuda=self.has_cuda,
            base_path="",
            origin_folder="",
            embedding_folder="",
        )
        embedding = trainer.learn_embedding(
            train_graphs,
            epoch=self.epochs,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            export=False,
            verbose=self.verbose,
            model_file=None,
        )

        singleton = SingletonEmbedding()

        # SVC, 27.10.2024
        learn_graph = data.get_matrix_as_graph(prediction_index)
        node_list = learn_graph.G.nodes()

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

        embs: List[Union[SingletonEmbedding, None]] = []

        if self.last_only:
            embs = [
                self.__construct(data, graphs, num_graphs - 1, num_nodes, node_list)
            ]
        else:
            embs = [None] * self.look_back
            for prediction_index in range(self.look_back, num_graphs):
                embs.append(
                    self.__construct(
                        data, graphs, prediction_index, num_nodes, node_list
                    )
                )

        return embs
