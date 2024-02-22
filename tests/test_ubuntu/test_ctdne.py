import sys
from os import path

sys.path.append(path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.loaders import load_netrep_dataset
from tigra.core.dgraph import DynamicAggregateGraph
from tigra.embeddings.sota.ctdne_wrapper import Ctdne
from tigra.shared import get_data_prefix

es = load_netrep_dataset(
    get_data_prefix() / "netrep" / "ia-sx-askubuntu-a2q.edges",
    has_weight=False,
)
print("num-events=", es.num_events())
print("time-span=", es.get_timespan_in_days())

dag = DynamicAggregateGraph("askubuntu", es, directed=False)
print("DIRECTED = ", dag.get_aggregate_graph().is_directed())

ctdne = Ctdne(5)
ctdne.make_embedding(dag)
print(ctdne.is_singleton_embedding())
print(ctdne.embeddings[0])
