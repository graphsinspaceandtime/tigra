from tests.common import choose_graph
from tigra.core.dgraph import DynamicGraph

es, meta = choose_graph()

dg = DynamicGraph()
dg.init_from_event_sequence(
    es,
    time_resolution="day" if meta["resolution"] == None else meta["resolution"],
    directed=False,
    cumulative=meta["cumulative"],
)

labels = {}

for i, g in enumerate(dg.get_graph_sequence()):
    for node in g.G.nodes:
        # print(f"G[{i}] N[{node}] Label[{g.get_node_label(node)}]")
        if node not in labels:
            labels[node] = g.get_node_label(node)
        else:
            lab = labels[node]
            if lab != g.get_node_label(node):
                print("WARNING")
