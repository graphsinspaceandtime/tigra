import sys
import json

from tigra.embeddings.utils import n_nodes_to_ae_layers
from tigra.data.loaders import get_dataset_by_id

configs = json.load(open("../dynae_config.json", "r", encoding="utf-8"))
dataset_configs: dict = configs["dataset_configs"]

for dataset, config in dataset_configs.items():
    Data = get_dataset_by_id(dataset)
    data = Data("../data")
    es = data.get_event_sequence()
    n_nodes = es.num_nodes()
    dim = config["dim"]
    print(dataset + " " + str(n_nodes_to_ae_layers(n_nodes, dim)))
