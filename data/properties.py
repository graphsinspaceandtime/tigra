import sys
import statistics
import typing


sys.path.append("..")

import data.bank as databank
from tigra.core.dgraph import DynamicGraph, DynamicAggregateGraph, EventSequence
from networkx import Graph


def debug_log(msg: str):
    print("[DEBUG] {}".format(msg))


def get_nxgraph_props(
    g: Graph,
) -> typing.Tuple[int, int, int, int, float, float, float]:
    degs = [p[1] for p in g.degree]
    if len(degs) < 2:
        debug_log("length is less than two")
    number_of_nodes = g.number_of_nodes()
    number_of_edges = g.number_of_edges()
    max_deg = max(degs)
    min_deg = min(degs)
    avg_deg = statistics.mean(degs)
    median_deg = statistics.median(degs)
    std_deg = statistics.stdev(degs) if len(degs) > 1 else 0

    return (
        number_of_nodes,
        number_of_edges,
        max_deg,
        min_deg,
        avg_deg,
        median_deg,
        std_deg,
    )


def get_properties(
    name: databank.Dataset, es: EventSequence
) -> typing.List[typing.Tuple[int, int, int, int, float, float, float]]:
    graph_props = []

    # get static graph properties
    # TODO: some might be directed
    g = DynamicAggregateGraph(name, es, directed=False)
    graph_props.append(get_nxgraph_props(g.get_aggregate_graph().G))

    # get dynamic graph properties (discrete graphs split by intervals)
    g = DynamicGraph()
    # TODO: some might be directed
    g.init_from_event_sequence(es, time_resolution="month", directed=False)

    for i, G in enumerate(g.get_nx_graph_sequence()):
        print(f"g-{i}")
        graph_props.append(get_nxgraph_props(G))

    return graph_props


def write_properties(key: databank.Dataset):
    es = databank.get_data(key)
    graph_props = get_properties(key, es)
    with open("{}.csv".format(key), "w") as f:
        f.write(
            "code,num_of_nodes,num_of_edges,max_deg,min_deg,avg_deg,median_deg,std_deg"
        )
        # handle windows line endings as a special case
        if sys.platform == "win32":
            f.write("\r")
        f.write("\n")

        for i, g in enumerate(graph_props):
            f.write(
                ",".join(
                    [
                        "static" if i == 0 else "snap_{}".format(i),
                        *[str(gp) for gp in list(g)],
                    ]
                )
            )
            # handle windows line endings as a special case
            if sys.platform == "win32":
                f.write("\r")
            f.write("\n")


if __name__ == "__main__":
    # for key in databank.get_keys():
    #     print("Calculating properties for {}...".format(key))
    #     write_properties(key)
    write_properties("sx-stackoverflow")
