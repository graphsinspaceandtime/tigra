import sys
import os

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicSparseMatrixGraph
from tigra.embeddings.sota.ae_wrapper import DynAERNN
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")
print("num-events=", es.num_events())
print("time-span =", es.get_timespan_in_days())

dsmg = DynamicSparseMatrixGraph(
    "radoslaw-email", es, time_resolution="month", directed=True
)

aernn = DynAERNN(10)
aernn.make_embedding(dsmg)

print(aernn._embs)
