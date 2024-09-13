import sys

sys.path.append("../..")

from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.timers_wrapper import Timers
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")

# ! Timers needs float values in matrix
dsmg = DynamicSparseMatrixGraph(
    "radoslaw", es, time_resolution="month", value_type="float"
)

method = Timers(10)
method.make_embedding(dsmg)
print(method.embeddings[0])
