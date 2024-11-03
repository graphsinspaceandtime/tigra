import sys

sys.path.append("../..")

from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.dynae_wrapper import DynAE
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.utils import n_nodes_to_ae_layers, iterate_valid_embeddings
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

dsmg = DynamicSparseMatrixGraph("radoslaw", es, time_resolution="month", directed=False)

dim = 25

n_units = n_nodes_to_ae_layers(dsmg.number_of_nodes, dim)
# n_units = [128]

snaps = dsmg.get_num_snaps()
look_back = 5

print(f"n_units: {n_units}")
print(f"dim: {dim}")
print(f"snaps: {snaps}")
print(f"look_back: {look_back}")


# method = DynAE(
#     dim=DIM,
#     n_units=n_units,
#     look_back=7,
#     beta=5,
#     nu1=1e-6,
#     nu2=1e-6,
#     batch_size=16,
#     epochs=10,
#     learning_rate=0.001,
#     weight_decay=1e-4,
# )
# method.make_embedding(dsmg)

# for i in range(len(method.embeddings)):
#     if method.embeddings[i] == None:
#         continue

#     gre = GraphReconstructionEval(dsmg.get_matrix_as_graph(i), method.embeddings[i])
#     print(len(method.embeddings[i].embedding))
#     print("Precision", gre.get_avg_precision())
#     print("Recall", gre.get_avg_recall())
#     print("F1", gre.getF1())

method = DynAE(
    dim=dim,
    n_units=n_units,
    look_back=look_back,
    beta=5,
    nu1=1e-6,
    nu2=1e-6,
    batch_size=16,
    epochs=10,
    learning_rate=0.001,
    weight_decay=1e-4,
    last_only=False,
)
method.make_embedding(dsmg)

for i, singleton_embedding in iterate_valid_embeddings(method.embeddings):
    print("=" * 25)
    print(" " * 10, end="")
    print(i)
    print("=" * 25)
    gre = GraphReconstructionEval(dsmg.get_matrix_as_graph(i), singleton_embedding)
    print(len(singleton_embedding.embedding))
    print("Precision", gre.get_avg_precision())
    print("Recall", gre.get_avg_recall())
    print("F1", gre.getF1())

"""
for i in range(8):
    gr = dsmg.get_matrix_as_graph(i)
    print(i)
    print(len(gr.G.nodes()))
    print(len(gr.G.edges()))
    gr.degree_distribution(to_print=True)
"""
