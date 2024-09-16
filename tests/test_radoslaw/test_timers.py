import sys

sys.path.append("../..")

from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.timers_wrapper import Timers
from tigra.core.greval import GraphReconstructionEval
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

# ! Timers needs float values in matrix
dsmg = DynamicSparseMatrixGraph(
    "radoslaw", es, time_resolution="month", value_type="float"
)

method = Timers(10)
method.make_embedding(dsmg)

gre = GraphReconstructionEval(dsmg.get_last_matrix_as_graph(), method.embeddings[0])
print("Precision", gre.get_avg_precision())
print("Recall", gre.get_avg_recall())
print("F1", gre.getF1())
