import sys
import os

sys.path.append(os.path.join("..", ".."))
sys.dont_write_bytecode = True

from tigra.core.dgraph import DynamicGraph, DynamicAggregateGraph
from data.bank import Databank

databank = Databank()
es = databank.get_data("radoslaw-email")
print("num-events=", es.num_events())
print("time-span =", es.get_timespan_in_days())

res = ["day", "week", "month", "3-months", "6-months", "year"]
for r in res:
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=r, cumulative=True, directed=False)
    print()
    print(r)
    dg.info()


"""
print("\n\n\nDynamic aggregate graph test")

dag = DynamicAggregateGraph("radoslaw", es, directed=False)
g = dag.get_aggregate_graph()
g.print_graph()


print("\n\n\n")

es = load_netrep_dataset("../data/netrep/soc-epinions-trust-dir.edges")
print("num-events=", es.num_events())
print("time-span =", es.get_timespan_in_days())

res = ["month", "3-months", "6-months", "year"]
for r in res:
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, time_resolution=r)
    print(r)
    dg.info()
"""
