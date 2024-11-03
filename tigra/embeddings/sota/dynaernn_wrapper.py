from typing import List
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
    ae_units: int
    rnn_units: int
    look_back: int
    beta: float
    nu1: float
    nu2: float
    batch_size: int
    epochs: int
    learning_rate: float
    bias: bool
    shuffle: bool
    has_cuda: bool
    verbose: int

    def __init__(
        self,
        dim,
        ae_units,
        rnn_units,
        look_back,
        beta,
        nu1,
        nu2,
        batch_size,
        epochs,
        learning_rate,
        bias=True,
        shuffle=True,
        has_cuda=False,
        verbose: 0,
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
        self.bias = bias
        self.shuffle = shuffle
        self.has_cuda = has_cuda
        self.verbose = verbose

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                "{} works with DynamicSparseMatrixGraph objects.".format(
                    type(self).__name__
                )
            )

        graphs = data.get_scipy_sparse_matrices()
        num_graphs = len(graphs)
        start_index = num_graphs - 1 - self.look_back
        graphs = graphs[start_index:]
        num_nodes = graphs[0].shape[0]
        node_list = list(range(num_nodes))

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
            graphs, epoch=self.epochs, lr=self.learning_rate, export=False, verbose=self.verbose
        )

        singleton = SingletonEmbedding()

        # SVC, 27.10.2024
        learn_graph = data.get_matrix_as_graph(-1)
        node_list = learn_graph.G.nodes()

        for node in node_list:
            singleton.add_embedding_vector(node, embedding[node].tolist())

        return [singleton]
