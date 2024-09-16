from data.bank import Databank
from tigra.core.dgraph import DynamicSparseMatrixGraph

db = Databank()
es = db.get_data("radoslaw-email")

dg = DynamicSparseMatrixGraph("test", es, time_resolution="day")

print(dg.get_neigbours(0))
