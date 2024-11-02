from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.sota.dynaernn_wrapper import DynAERNN
from tigra.embeddings.utils import n_nodes_to_ae_layers
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

dsmg = DynamicSparseMatrixGraph("radoslaw", es, time_resolution="month", directed=False)

DIM = 25

n_units = n_nodes_to_ae_layers(dsmg.number_of_nodes, DIM)
print(n_units)

method = DynAERNN(
    dim=DIM,
    ae_units=n_units,
    rnn_units=n_units,
    look_back=3,
    beta=5,
    nu1=1e-6,
    nu2=1e-6,
    batch_size=16,
    epochs=500,
    learning_rate=0.001,
)
method.make_embedding(dsmg)

gre = GraphReconstructionEval(dsmg.get_last_matrix_as_graph(), method.embeddings[0])
print("Precision", gre.get_avg_precision())
print("Recall", gre.get_avg_recall())
print("F1", gre.getF1())
