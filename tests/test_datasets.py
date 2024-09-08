from tigra.core.dgraph import DynamicGraph, DynamicAggregateGraph

from data.bank import Databank

databank = Databank()


def form_graphs(es, tr="month"):
    dg = DynamicGraph()
    dg.init_from_event_sequence(es, tr, directed=False)
    dg.info()

    dag = DynamicAggregateGraph("DAG", es, directed=False)
    ag = dag.get_aggregate_graph()
    print("#n", len(ag.G.nodes()), " #l", len(ag.G.edges()))


for dataset in databank.get_keys():
    print("dataset:", dataset)
    es = databank.get_data(dataset)

    print("num-events=", es.num_events())
    print("time-span =", es.get_timespan_in_days())
    form_graphs(es)
    print()
