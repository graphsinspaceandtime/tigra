from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.sota.dyngem_wrapper import DynGEM
from tigra.embeddings.utils import n_nodes_to_ae_layers
from tests.common import choose_graph

es, meta = choose_graph()

dsmg = DynamicSparseMatrixGraph(
    "radoslaw",
    es,
    time_resolution="day" if meta["resolution"] == None else meta["resolution"],
    cumulative=meta["cumulative"],
    directed=False,
)

dim = 25

n_units = n_nodes_to_ae_layers(dsmg.number_of_nodes, dim)
# n_units = [128]

snaps = dsmg.get_num_snaps()
look_back = snaps // 2

print(f"n_units: {n_units}")
print(f"dim: {dim}")
print(f"snaps: {snaps}")
print(f"look_back: {look_back}")

method = DynGEM(
    dim=dim,
    n_units=n_units,
    alpha=1e-6,
    beta=10,
    nu1=1e-6,
    nu2=1e-6,
    batch_size=256,
    epochs=10,
    learning_rate=1e-3,
)
method.make_embedding(dsmg)

gre = GraphReconstructionEval(dsmg.get_last_matrix_as_graph(), method.embeddings[-1])
print("Precision", gre.get_avg_precision())
print("Recall", gre.get_avg_recall())
print("F1", gre.getF1())
