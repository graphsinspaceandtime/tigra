import sys

sys.path.append("..")

from data.bank import Databank
from tigra.core.dgraph import DynamicGraph

databank = Databank()
es = databank.get_data("radoslaw-email")

dg = DynamicGraph()
dg.init_from_event_sequence(es, time_resolution="month")

matrices = dg.get_scipy_sparse_matrices()

print(matrices)
