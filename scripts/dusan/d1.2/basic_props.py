import sys

sys.path.append("../../..")

from data.bank import Databank
from tigra.core.dgraph import DynamicGraph, DynamicAggregateGraph
import math

db = Databank()


first_table = []
second_table = []


def basic_props(dataset):
    es = db.get_data(dataset)
    meta = db.get_metadata(dataset)

    dg = DynamicGraph()
    dg.init_from_event_sequence(
        es,
        time_resolution="day",
        directed=False,
        cumulative=meta["cumulative"],
    )

    dag = DynamicAggregateGraph(name=dataset, event_sequence=es, directed=False)

    num_nodes = dag.agg_graph.num_nodes()
    num_temporal_edges = len(es.get_events())
    num_edges = dag.agg_graph.num_edges()
    time_span = math.ceil(es.get_timespan_in_days())
    resolution = meta["resolution"].replace("-", " ")

    first_table.append(
        f"{dataset} & {num_nodes} & {num_temporal_edges} & {num_edges} \\\\"
    )
    first_table.append("\\hline")

    second_table.append(f"{dataset} & {time_span} & {resolution} \\\\")
    second_table.append("\\hline")


for i, dataset in enumerate(db.get_keys()):
    print(
        "Calculating for dataset " + str(i + 1) + " out of " + str(len(db.get_keys()))
    )
    basic_props(dataset)

print()
print()
print()
print("\n".join(first_table))
print()
print()
print()
print("\n".join(second_table))
