import sys
from tigra.embeddings.sota.dynae_wrapper import DynAE
from tigra.embeddings.utils import n_nodes_to_ae_layers
from tigra.data.loaders import get_dataset_by_id
from tigra.core.dgraph import DynamicSparseMatrixGraph

dataset_base_path = sys.argv[1].strip()
dataset_id = sys.argv[2].strip()
dim = int(sys.argv[3])
look_back = int(sys.argv[4])
layer_size = int(sys.argv[5])

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

n_nodes = dsmg.number_of_nodes

# n_units = n_nodes_to_ae_layers(n_nodes=n_nodes, embedding_size=dim)

# n_unit = n_units[len(n_units) // 2]

n_unit = layer_size

print(f"n_unit: {n_unit}")

config = {
    "beta": 5,
    "nu1": 1e-6,
    "nu2": 1e-6,
    "batch_size": 16,
    "epochs": 1000,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dim": dim,
    "look_back": look_back,
    "n_units": [n_unit],
}

dynae = DynAE(has_cuda=False, **config)

dynae.make_embedding(data=dsmg)

last_embedding = dynae.embeddings[-1]

if last_embedding == None:
    raise Exception("bad, last_embedding is none")

last_embedding.save_to_file(f"dynae_{data.id}.emb")
