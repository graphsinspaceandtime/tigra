from typing import List, Union, Optional
import networkx as nx
from tigra.embeddings.embedding import Embedding, SingletonEmbedding
from tigra.core.dgraph import DynamicSparseMatrixGraph, DynamicGraph, Graph
from tigra.embeddings.sota.CTGCN.baseline.dynAE import (
    DynAE as _DynAE,
    DynamicEmbedding,
    BatchPredictor,
)
from tigra.embeddings.ae.base import DynGraph2VecLoss, CombinedDynGraphLoss
from tigra.embeddings.ae.common import (
    BatchGenerator,
    BetaPenaltyTensorInitializer,
    GraphLIDPenaltyTensorInitializer,
    PenaltyTensorInitializer,
    DynamicEmbeddingWithLID,
)


class DynAE(Embedding):
    n_units: List[int]
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
    penalty_type: str
    lid_k: int
    lid_cutoff: int
    use_lid_loss: bool
    lambda_lid: float

    def __init__(
        self,
        dim: int,
        n_units: List[int],
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
        penalty_type: str = "beta",  # "beta" or "lid"
        lid_k: int = 10,
        lid_cutoff: int = 10,
        use_lid_loss: bool = False,  # Use GraphLIDLoss as additional loss
        lambda_lid: float = 0.1,  # Weight for LID loss component
        **kwargs,  # collect redundant arguments
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
        self.last_only = last_only
        self.bias = bias
        self.shuffle = shuffle
        self.has_cuda = has_cuda
        self.verbose = verbose
        self.penalty_type = penalty_type
        self.lid_k = lid_k
        self.lid_cutoff = lid_cutoff
        self.use_lid_loss = use_lid_loss
        self.lambda_lid = lambda_lid

    def _create_penalty_initializer(
        self, data: DynamicSparseMatrixGraph
    ) -> PenaltyTensorInitializer:
        """Create penalty initializer based on penalty_type"""
        if self.penalty_type == "beta":
            return BetaPenaltyTensorInitializer(beta=self.beta)
        elif self.penalty_type == "lid":
            # Convert DynamicSparseMatrixGraph to DynamicGraph for LID computation
            # IMPORTANT: Don't remove isolated nodes, as batch generator uses full matrix indices
            num_snapshots = data.get_num_snaps()
            graph_list = []
            for idx in range(num_snapshots):
                M = data.get_scipy_sparse_matrices()[idx]
                G = nx.from_scipy_sparse_array(M, create_using=nx.Graph)
                graph = Graph(data.name + f"_t{idx}", nx_graph=G)
                # Don't remove isolated nodes - keep node IDs consistent with matrix
                graph_list.append(graph)

            dynamic_graph = DynamicGraph(list_of_graphs=graph_list)

            return GraphLIDPenaltyTensorInitializer(
                dynamic_graph=dynamic_graph,
                k=self.lid_k,
                cutoff=self.lid_cutoff,
            )
        else:
            raise ValueError(
                f"Unknown penalty_type: {self.penalty_type}. Must be 'beta' or 'lid'."
            )

    def __construct(
        self,
        data: DynamicSparseMatrixGraph,
        graphs,
        prediction_index,
        num_nodes,
        node_list,
        penalty_initializer: Optional[PenaltyTensorInitializer] = None,
    ) -> SingletonEmbedding:
        start_index = prediction_index - self.look_back
        train_graphs = graphs[start_index : prediction_index + 1]

        model = _DynAE(
            input_dim=num_nodes,
            output_dim=self.dim,
            look_back=self.look_back,
            n_units=self.n_units,
            bias=self.bias,
        )
        # Choose loss function based on use_lid_loss parameter
        if self.use_lid_loss:
            # Combined loss: reconstruction + LID regularization
            loss = CombinedDynGraphLoss(
                nu1=self.nu1, nu2=self.nu2, lambda_lid=self.lambda_lid
            )
        else:
            # Standard reconstruction loss only
            loss = DynGraph2VecLoss(nu1=self.nu1, nu2=self.nu2)

        # Create penalty initializer if not provided
        if penalty_initializer is None:
            penalty_initializer = self._create_penalty_initializer(data)

        # Updated: use penalty_initializer instead of beta
        batch_generator = BatchGenerator(
            node_list=node_list,
            batch_size=self.batch_size,
            look_back=self.look_back,
            penalty_initializer=penalty_initializer,
            shuffle=self.shuffle,
            has_cuda=self.has_cuda,
        )
        batch_predictor = BatchPredictor(
            node_list=node_list, batch_size=self.batch_size, has_cuda=self.has_cuda
        )

        # Choose trainer class based on whether we need LID loss support
        if self.use_lid_loss:
            trainer = DynamicEmbeddingWithLID(
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
        else:
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

        # Create penalty initializer once for all snapshots
        penalty_initializer = self._create_penalty_initializer(data)

        embs: List[Union[SingletonEmbedding, None]] = []

        if self.last_only:
            embs = [
                self.__construct(
                    data, graphs, num_graphs - 1, num_nodes, node_list, penalty_initializer
                )
            ]
        else:
            embs = [None] * self.look_back
            for prediction_index in range(self.look_back, num_graphs):
                embs.append(
                    self.__construct(
                        data, graphs, prediction_index, num_nodes, node_list, penalty_initializer
                    )
                )

        return embs
