from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.core.greval import GraphReconstructionEval
from tigra.embeddings.sota.dynaernn_wrapper import DynAERNN
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

dsmg = DynamicSparseMatrixGraph("radoslaw", es, time_resolution="month")

method = DynAERNN(
    dim=10,
    ae_units=[500, 300],
    rnn_units=[500, 300],
    look_back=5,
    beta=5,
    nu1=1e-6,
    nu2=1e-6,
    batch_size=200,
    epochs=10,
    learning_rate=0.001,
)
method.make_embedding(dsmg)

gre = GraphReconstructionEval(dsmg.get_last_matrix_as_graph(), method.embeddings[0])
print("Precision", gre.get_avg_precision())
print("Recall", gre.get_avg_recall())
print("F1", gre.getF1())
