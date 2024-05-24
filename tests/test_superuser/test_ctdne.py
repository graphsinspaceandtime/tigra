import sys
import os

workers = int(os.environ.get("NUM_CPUS", 4))

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicAggregateGraph
from tigra.embeddings.sota.ctdne_wrapper import Ctdne
from data.bank import Databank

databank = Databank()
es = databank.get_data("sx-superuser")
print("num-events=", es.num_events())
print("time-span=", es.get_timespan_in_days())

dag = DynamicAggregateGraph("superuser", es, directed=False)
print("DIRECTED = ", dag.get_aggregate_graph().is_directed())

ctdne = Ctdne(5)
ctdne.make_embedding(dag)
print(ctdne.is_singleton_embedding())
print(ctdne.embeddings[0])
