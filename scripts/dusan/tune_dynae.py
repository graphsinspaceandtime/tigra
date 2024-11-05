import sys

sys.path.append("../..")

from tests.common import choose_graph
from tigra.embeddings.hypertuner import DynamicEmbedding_Tuner
from tigra.embeddings.utils import n_nodes_to_ae_layers
from tigra.core.dgraph import DynamicSparseMatrixGraph
from data.bank import Databank

db = Databank(path=None if sys.argv[1] else sys.argv[1])

dataset_names = db.get_keys()

for dataset_name in dataset_names:
    print(f"dataset: {dataset_name}")
    es = db.get_data(dataset_name)
    meta = db.get_metadata(dataset_name)

    dsmg = DynamicSparseMatrixGraph(
        dataset_name, es, time_resolution=meta["resolution"], directed=False
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
        has_cuda=True,
    )

    tuner.tune()
    tuner.print_report()
