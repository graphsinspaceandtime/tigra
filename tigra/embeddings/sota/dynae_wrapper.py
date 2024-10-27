from typing import List
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.CTGCN.baseline.dynAE import (
    DynAE as _DynAE,
    DynGraph2VecLoss,
    BatchGenerator,
    BatchPredictor,
    DynamicEmbedding,
)


class DynAE(Embedding):
    n_units: int
    look_back: int
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

    def __init__(
        self,
        dim,
        n_units,
        look_back,
        beta,
        nu1,
        nu2,
        batch_size,
        epochs,
        learning_rate,
        weight_decay,
        bias=True,
        shuffle=True,
        has_cuda=False,
    ):
        super(DynAE, self).__init__(dim)
        self.n_units = n_units
        self.look_back = look_back
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

    def construct(self, data) -> List[SingletonEmbedding]:
        if not isinstance(data, DynamicSparseMatrixGraph):
            raise Exception(
                "{} works with DynamicSparseMatrixGraph objects.".format(
                    type(self).__name__
                )
            )

        graphs = data.get_scipy_sparse_matrices()
        #print("DEBUG, len(graphs)", len(graphs))
        #for i in range(0, len(graphs)):
        #    print("DEBUG, num_nodes", graphs[i].shape[0])
        
        num_nodes = graphs[0].shape[0]
        node_list = list(range(num_nodes))

        model = _DynAE(
            input_dim=num_nodes,
            output_dim=self.dim,
            look_back=self.look_back,
            n_units=self.n_units,
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
            graphs,
            epoch=self.epochs,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            export=False,
        )

        singleton = SingletonEmbedding()

        # SVC, 27.10.2024
        last_graph = data.get_last_matrix_as_graph()
        node_list = last_graph.G.nodes()

        for node in node_list:
            singleton.add_embedding_vector(node, embedding[node].tolist())

        return [singleton]
