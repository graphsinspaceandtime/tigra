import sys

sys.path.append("../..")

from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.dynae_wrapper import DynAE
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.utils import n_nodes_to_ae_layers
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

dsmg = DynamicSparseMatrixGraph("radoslaw", es, time_resolution="month")

DIM = 25

n_units = n_nodes_to_ae_layers(dsmg.number_of_nodes, DIM)
print(n_units)


method = DynAE(
    dim=DIM,
    n_units=n_units,
    look_back=7,
    beta=5,
    nu1=1e-6,
    nu2=1e-6,
    batch_size=16,
    epochs=10,
    learning_rate=0.001,
    weight_decay=1e-4,
)
method.make_embedding(dsmg)

gre = GraphReconstructionEval(dsmg.get_matrix_as_graph(-1), method.embeddings[0])
print(len(method.embeddings[0].embedding))
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
