import sys

sys.path.append("..")
sys.dont_write_bytecode = True

from tigra.core.loaders import load_netrep_dataset
from tigra.core.dgraph import DynamicAggregateGraph
from tigra.embeddings.sota.ctdne_wrapper import Ctdne

es = load_netrep_dataset("../data/netrep/ia-radoslaw-email.edges")
print("num-events=", es.num_events())
print("time-span =", es.get_timespan_in_days())

dag = DynamicAggregateGraph("radoslaw", es, directed=False)
print("DIRECTED = ", dag.get_aggregate_graph().is_directed())

ctdne = Ctdne(5)
ctdne.make_embedding(dag)
print(ctdne.is_singleton_embedding())
print(ctdne.embeddings[0])
