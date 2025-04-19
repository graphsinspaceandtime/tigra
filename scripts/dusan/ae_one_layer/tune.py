import sys
import typing

from tigra.embeddings.de_hypertuner import DynamicEmbedding2_Tuner
from tigra.embeddings.utils import n_nodes_to_ae_layers
from tigra.data.loaders import get_dataset_by_id
from tigra.core.dgraph import DynamicSparseMatrixGraph

dataset_id = sys.argv[1].strip()
dataset_base_path = sys.argv[2].strip()
method: typing.Literal["DynAE", "DynRNN", "DynAERNN", "DynGEM"] = sys.argv[3].strip()  # type: ignore
if method not in ["DynAE", "DynRNN", "DynAERNN", "DynGEM"]:
    print("method must be one of 'DynAE', 'DynRNN', 'DynAERNN', 'DynGEM'")


Data = get_dataset_by_id(dataset_id)

data = Data(base_path=dataset_base_path)

data.download()

dsmg = DynamicSparseMatrixGraph(
    data.name,
    data.get_event_sequence(),
    time_resolution=data.resolution,
    directed=False,
    cumulative=data.cumulative,
)
print("Loaded dataset")

last_graph = dsmg.get_last_matrix_as_graph()
num_snaps = dsmg.get_num_snaps()

dims = [10, 25, 50, 100]
if data.id == "ia_contacts_hypertext_2009":
    dims = [10, 25, 50]

look_backs = list(range(1, num_snaps))

configs = []

default_parameters = {
    "beta": 5,
    "nu1": 1e-6,
    "nu2": 1e-6,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "epochs": 1000,
}

if method == "DynGEM":
    default_parameters["alpha"] = 1e-6

for dim in dims:
    for look_back in look_backs:
        n_units = n_nodes_to_ae_layers(dsmg.number_of_nodes, dim)
        for n_unit in n_units:
            config = {
                "dim": dim,
                "look_back": look_back,
                **default_parameters,
            }
            if method == "DynAERNN":
                config["ae_units"] = [n_unit]
                config["rnn_units"] = [n_unit]
            else:
                config["n_units"] = [n_unit]
            configs.append(config)
        if len(n_units) > 1:
            if method == "DynAERNN":
                configs.append({
                    "dim": dim,
                    "look_back": look_back,
                    "ae_units": n_units,
                    "rnn_units": n_units,
                    **default_parameters,
                })
            else:
                configs.append({
                    "dim": dim,
                    "look_back": look_back,
                    "n_units": n_units,
                    **default_parameters,
                })


print(f"NUM OF CONFIGS: {len(configs)}")


tuner = DynamicEmbedding2_Tuner(
    graph=last_graph,
    data=dsmg,
    num_runs=5,
    params_desc="",
    method=method,
    configs=configs,
    has_cuda=False,
)

print("STARTED TUNING")

tuner.tune()
tuner.print_report()
