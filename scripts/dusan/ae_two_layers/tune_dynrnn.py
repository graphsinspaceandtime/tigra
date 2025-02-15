import sys

sys.path.append("../..")

from tigra.embeddings.hypertuner import DynamicEmbedding_Tuner
from tigra.core.dgraph import DynamicSparseMatrixGraph
from data.bank import Databank, Dataset

db = Databank(sys.argv[1])

dataset_name: Dataset = sys.argv[2]  # type: ignore

print(f"dataset: {dataset_name}")
es = db.get_data(dataset_name)
meta = db.get_metadata(dataset_name)

dsmg = DynamicSparseMatrixGraph(
    dataset_name,
    es,
    time_resolution=meta["resolution"],
    directed=False,
    cumulative=meta["cumulative"],
)

last_graph = dsmg.get_last_matrix_as_graph()
num_snaps = dsmg.get_num_snaps()

params_grid = {
    "dim": (
        [10, 25, 50, 100]
        if dataset_name != "ia-contacts_hypertext2009"
        else [10, 25, 50]
    ),
    "look_back": list(range(1, num_snaps)),
}

tuner = DynamicEmbedding_Tuner(
    graph=last_graph,
    data=dsmg,
    num_runs=5,
    params_desc="",
    method="DynRNN",
    param_grid=params_grid,
    epochs=1000,
    batch_size=16,
    n_nodes=dsmg.number_of_nodes,
    has_cuda=True,
    pick_n_layers_from_middle=2,
)

tuner.tune()
tuner.print_report()
