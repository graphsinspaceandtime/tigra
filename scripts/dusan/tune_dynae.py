import sys

sys.path.append("../..")

from tests.common import choose_graph
from tigra.embeddings.hypertuner import DynamicEmbedding_Tuner
from tigra.embeddings.utils import n_nodes_to_ae_layers
from tigra.core.dgraph import DynamicSparseMatrixGraph


es, meta = choose_graph(name="radoslaw-email")

dsmg = DynamicSparseMatrixGraph(
    "", es, time_resolution=meta["resolution"], directed=False
)
last_graph = dsmg.get_last_matrix_as_graph()
num_snaps = dsmg.get_num_snaps()

params_grid = {
    "dim": [10, 25, 50, 100],
    "look_back": list(range(1, num_snaps)),
}

tuner = DynamicEmbedding_Tuner(
    graph=last_graph,
    data=dsmg,
    params_desc="",
    method="DynAE",
    param_grid=params_grid,
    epochs=1000,
    batch_size=16,
    n_nodes=dsmg.number_of_nodes,
)

tuner.tune()
tuner.print_report()
