from typing import List, Union
from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.embeddings.sota.CTGCN.baseline.dynAE import (
    DynamicEmbedding,
)
from tigra.embeddings.sota.CTGCN.baseline.dynGEM import (
    DynGEM as _DynGEM,
    DynGEMLoss,
    DynGEMBatchGenerator,
    DynGEMBatchPredictor,
)


class DynGEM(Embedding):
    n_units: List[int]
    alpha: float
    beta: float
    nu1: float
    nu2: float
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    bias: bool
    shuffle: bool
    has_cuda: bool
    verbose: int

    def __init__(
        self,
        dim: int,
        n_units: List[int],
        alpha: float,
        beta: float,
        nu1: float,
        nu2: float,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        bias=True,
        shuffle=True,
        has_cuda=False,
        verbose=0,
    ):
        super(DynGEM, self).__init__(dim)
        self.n_units = n_units
        self.alpha = alpha
        self.beta = beta
        self.nu1 = nu1
        self.nu2 = nu2
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.bias = bias
        self.shuffle = shuffle
        self.has_cuda = has_cuda
        self.verbose = verbose

    def construct(self, data) -> List[Union[SingletonEmbedding, None]]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                "{} works with DynamicSparseMatrixGraph objects.".format(
                    type(self).__name__
                )
            )

        graphs = data.get_scipy_sparse_matrices()
        num_nodes = graphs[0].shape[0]
        node_list = list(range(num_nodes))

        model = _DynGEM(
            input_dim=num_nodes,
            output_dim=self.dim,
            n_units=self.n_units,
            bias=self.bias,
        )
        loss = DynGEMLoss(alpha=self.alpha, beta=self.beta, nu1=self.nu1, nu2=self.nu2)
        batch_generator = DynGEMBatchGenerator(
            node_list=node_list,
            batch_size=self.batch_size,
            beta=self.beta,
            shuffle=self.shuffle,
            has_cuda=self.has_cuda,
        )
        batch_predictor = DynGEMBatchPredictor(
            node_list=node_list, batch_size=self.batch_size, has_cuda=self.has_cuda
        )

        embeddings = []

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

        for i in range(len(graphs)):
            embedding = trainer.learn_embedding(
                [graphs[i]],
                epoch=self.epochs,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                export=False,
                verbose=self.verbose,
                model_file=None,
            )

            singleton = SingletonEmbedding()

            learn_graph = data.get_matrix_as_graph(i)
            nl = learn_graph.G.nodes()

            for node in nl:
                singleton.add_embedding_vector(node, embedding[node].tolist())

            embeddings.append(singleton)

        return embeddings
