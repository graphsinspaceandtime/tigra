import sys

sys.path.append("../../..")

import typing

import torch

from tigra.embeddings.de_hypertuner import DynamicEmbedding3_Tuner
from tigra.embeddings.utils import n_nodes_to_ae_layers
from tigra.data.loaders import get_dataset_by_id
from tigra.core.dgraph import DynamicSparseMatrixGraph

# dataset_id = sys.argv[1].strip()
dataset_id = "radoslaw_email"
dataset_base_path = sys.argv[1].strip()

Data = get_dataset_by_id(dataset_id)

data = Data(base_path=dataset_base_path)

print("Downloading dataset...")

data.download()

print("Getting event sequence...")

es = data.get_event_sequence(shrink=True)

dsmg = DynamicSparseMatrixGraph(
    data.name,
    es,
    time_resolution=data.resolution,
    directed=False,
    cumulative=data.cumulative,
)

print("Loaded dataset")

last_graph = dsmg.get_last_matrix_as_graph()
num_snaps = dsmg.get_num_snaps()

default_parameters = {
    "beta": 5,
    "nu1": 1e-6,
    "nu2": 1e-6,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "epochs": 1000,
}

layers = n_nodes_to_ae_layers(dsmg.number_of_nodes, 100)

# Create configurations: beta penalty, LID penalty, and LID penalty + LID loss
configs = [
    # Baseline: Beta penalty only
    {
        "dim": 100,
        "look_back": 1,
        "n_units": layers,
        "penalty_type": "beta",
        **default_parameters,
    },

    # LID penalty only (moderate)
    {
        "dim": 100,
        "look_back": 1,
        "n_units": layers,
        "penalty_type": "lid",
        "lid_k": 10,
        "lid_cutoff": 15,
        **default_parameters,
    },

    # LID penalty + LID loss (combined approach)
    {
        "dim": 100,
        "look_back": 1,
        "n_units": layers,
        "penalty_type": "lid",
        "lid_k": 10,
        "lid_cutoff": 15,
        "use_lid_loss": True,
        "lambda_lid": 0.1,
        **default_parameters,
    },

    # LID penalty + stronger LID loss
    {
        "dim": 100,
        "look_back": 1,
        "n_units": layers,
        "penalty_type": "lid",
        "lid_k": 10,
        "lid_cutoff": 15,
        "use_lid_loss": True,
        "lambda_lid": 0.5,
        **default_parameters,
    },
]

print(f"NUM OF CONFIGS: {len(configs)}")

tuner = DynamicEmbedding3_Tuner(
    graph=last_graph,
    data=dsmg,
    num_runs=1,
    params_desc="Testing beta penalty vs LID penalty vs LID penalty+loss",
    configs=configs,
    has_cuda=torch.cuda.is_available(),
)

print("STARTED TUNING")

tuner.tune(progress_bar=True)
tuner.print_report()
tuner.save_report(f"report_{dataset_id}_DynAE_lid_comparison.json")
